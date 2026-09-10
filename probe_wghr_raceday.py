# -*- coding: utf-8 -*-
"""마체중 tier 판정 — 경마 시행일(금·토·일)에 실행할 것.

묻는 것 하나: **발주 전에 마체중(wgHr)이 채워지는가.**
  채워진다  → schema_v2.py 의 X_wgHr / X_wgHr_delta 를 tier="B" 로 바꾼다
              (73피처 주말 실시간 모델에 마체중 축이 하나 생긴다)
  안 채워진다 → tier="G" 유지 (리플레이 전용)

결론 (2026-09-10 제주 실측 — 이 스크립트로 확정):
  4R(발주 14:30) 이 **T-65분 0/8 → T-61분 8/8** 로 전환. 같은 시점 3R(14:00)은 T-36분에
  이미 7/7. → **각 경주 발주 약 T-61~65분에 채워진다. tier="B" 확정.**
  기존 문서의 "T-60~90분"은 아래쪽 끝만 맞았다(T-65 에는 아직 비어 있다).

이미 확인된 것 (2026-09-10 D-1 실측):
  다가올 경주일 5건 전부 원장(API4_3)·API25_1 **양쪽 wgHr 충전 0/423두.**
  → 'D-1 에는 없다' 는 확정. 남은 것은 '당일 발주 전' 인지 '발주 후' 인지.
  두 엔드포인트의 wgHr 값은 지난 경주일 4건 365두에서 **완전 일치**했으므로
  (불일치 0) 어느 쪽으로 서빙해도 학습값과 같은 숫자가 온다.

왜 probe_realtime.py 로 안 되는가:
  그쪽은 한 시점 스냅샷을 '종료/대기' 로만 갈라 찍는다. 발주까지 **몇 분 남았는지**를
  기록하지 않아 "T-60 에 채워졌다" 를 증명할 수 없다. 저장도 안 한다.
  이 스크립트는 경주마다 T-minus 를 같이 남기고 snapshots/ 에 누적한다.

쓰는 법 — 첫 경주 발주 2시간 전부터 30~40분 간격으로 3~4번:
    PYTHONUTF8=1 python probe_wghr_raceday.py
마지막에 판정이 출력된다. 한 번만 돌려도 '대기 중인 경주가 채워져 있다' 가 잡히면
그 자리에서 tier=B 가 증명된다.
"""
import io, json, os, sys
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')
from kra_client import KRA, KRAError

LOG = 'snapshots/wghr_raceday.jsonl'
PLACEHOLDER = {'', '-', '0', '0()', ' (0%)', '(0%)', '0.0'}


def FILLED(v):
    """이 API 는 빈 값을 NULL 이 아니라 플레이스홀더로 준다. truthy 검사로 세면 안 된다."""
    s = str(v).strip()
    return s not in PLACEHOLDER and not s.startswith('0()')


def hhmm_to_min(s):
    s = str(s or '').strip()
    return int(s[:2]) * 60 + int(s[2:]) if len(s) == 4 and s.isdigit() else None


def main():
    day = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime('%Y%m%d')
    now = datetime.now()
    now_min = now.hour * 60 + now.minute
    kra = KRA()
    print(f'=== {day} · 관측 {now:%H:%M} ===\n')

    obs, verdict_rows = [], []
    for meet, nm in ((1, '서울'), (2, '제주'), (3, '부경')):
        try:
            plan = kra.fetch('race_plan', meet=meet, rc_date=day)
        except KRAError as e:
            print(f'[{nm}] racePlan 실패: {e}'); continue
        if not plan:
            continue
        try:
            led = kra.fetch('race_result', meet=meet, rc_date=day)
        except KRAError:
            led = []
        try:
            ew = kra.fetch('entry_weight', meet=meet, rc_date=day)
        except KRAError:
            ew = []

        print(f'[{nm}] {len(plan)}경주')
        print(f"   {'R':>3} {'발주':>6} {'T-minus':>8} {'상태':>5} | "
              f"{'원장 wgHr':>12} {'API25_1 wgHr':>14}")
        for r in sorted(plan, key=lambda x: int(x.get('rcNo', 0))):
            rno = int(r['rcNo'])
            st = hhmm_to_min(r.get('schStTime'))
            tminus = None if st is None else st - now_min
            lrows = [x for x in led if str(x.get('rcNo')) == str(rno)]
            erows = [x for x in ew if str(x.get('rcNo')) == str(rno)]
            lf = sum(1 for x in lrows if FILLED(x.get('wgHr')))
            ef = sum(1 for x in erows if FILLED(x.get('wgHr')))
            state = '대기' if (tminus is not None and tminus > 0) else '지남'
            stxt = '  -  ' if st is None else f'{st//60:02d}:{st%60:02d}'
            ttxt = '   ?  ' if tminus is None else f'{tminus:+5d}분'
            print(f'   {rno:>3} {stxt:>6} {ttxt:>8} {state:>5} | '
                  f'{lf:>5}/{len(lrows):<6} {ef:>6}/{len(erows):<7}')
            rec = dict(day=day, obs=now.strftime('%Y%m%d_%H%M'), meet=meet, rcNo=rno,
                       sch=r.get('schStTime'), tminus=tminus, state=state,
                       led_filled=lf, led_n=len(lrows), ew_filled=ef, ew_n=len(erows))
            obs.append(rec)
            if state == '대기' and (lf or ef):
                verdict_rows.append(rec)
        print()

    if not obs:
        print('오늘 편성된 경주가 없다. 경마 시행일(금·토·일)에 실행할 것.')
        return

    os.makedirs('snapshots', exist_ok=True)
    with io.open(LOG, 'a', encoding='utf-8') as f:
        for r in obs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print('=' * 70)
    if verdict_rows:
        best = max(verdict_rows, key=lambda r: r['tminus'])
        print(f'★ 발주 전 충전 확인 — 최대 여유 T-{best["tminus"]}분 '
              f'({best["meet"]}장 {best["rcNo"]}R: 원장 {best["led_filled"]}/{best["led_n"]}, '
              f'API25_1 {best["ew_filled"]}/{best["ew_n"]})')
        print('  → schema_v2.py 의 X_wgHr / X_wgHr_delta 를 tier="B" 로 바꿔라.')
        print('     tier 값은 파케이 내용을 바꾸지 않으므로 재빌드가 필요 없다.')
        print('     73피처 A/B 는 이미 재둔 것이 있다 (basemodel/experiments/ab_wghr.py B75 팔).')
    else:
        waiting = [r for r in obs if r['state'] == '대기']
        if not waiting:
            print('판정 보류 — 관측 시점에 대기 중인 경주가 없었다(전 경주 발주 지남).')
            print('  → 첫 경주 발주 **전**에 다시 돌려라.')
        else:
            m = max(r['tminus'] for r in waiting)
            print(f'아직 미충전 — 대기 경주 {len(waiting)}개(최대 T-{m}분) 전부 wgHr 0.')
            print('  → 발주가 가까워지면 다시 돌려라. 첫 경주 발주 직후까지 계속 0 이면 tier="G" 확정.')
    print(f'\n관측 {len(obs)}경주 기록 → {LOG}  ({kra.calls}콜)')


if __name__ == '__main__':
    main()
