# -*- coding: utf-8 -*-
"""
경주 종료 → API 반영 지연 실측 프로브.

미해결 질문 #1 을 닫기 위한 유일한 수단. 공개 문서에 반영 시점이 없으므로 직접 잰다.
경마 시행일(기본 금·토·일, 2026년은 월·화·수 예외 편성도 있음)에 실행할 것.

    python probe_latency.py                 # 오늘, 서울·부경·제주 전부, 5분 간격
    python probe_latency.py 20260828 1 60   # 날짜/경마장/간격(초) 지정

동작: 해당 경주일의 경주번호별 데이터 등장 시각을 기록해
      "경주 종료 시각 → API 노출 시각" 지연을 산출한다.
      결과는 latency_<날짜>.csv 로 append 되므로 중단해도 이어붙는다.
"""
import sys, time, io, csv, os, datetime, json
from kra_client import KRA, KRAError, MEETS

def now():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def main():
    date  = sys.argv[1] if len(sys.argv) > 1 else datetime.date.today().strftime('%Y%m%d')
    meets = [int(sys.argv[2])] if len(sys.argv) > 2 else [1, 2, 3]
    every = int(sys.argv[3]) if len(sys.argv) > 3 else 300

    kra = KRA()
    out = f'latency_{date}.csv'
    new = not os.path.exists(out)
    f = io.open(out, 'a', encoding='utf-8-sig', newline='')
    w = csv.writer(f)
    if new:
        w.writerow(['observed_at', 'meet', 'rc_no', 'n_rows', 'ord_filled',
                    'rcTime_filled', 'odds_filled', 'weather', 'track'])
        f.flush()

    seen = {}   # (meet, rc_no) -> 최초 관측 시각
    print(f'[{now()}] 폴링 시작 date={date} meets={[MEETS[m] for m in meets]} 간격={every}s')
    print('  Ctrl+C 로 중단. 결과 →', out)
    try:
        while True:
            for meet in meets:
                try:
                    rows = kra.fetch('race_result', rc_date=date, meet=meet)
                except KRAError as e:
                    print(f'[{now()}] {MEETS[meet]}: {e}')
                    continue
                if not rows:
                    print(f'[{now()}] {MEETS[meet]}: 아직 0행')
                    continue
                by_race = {}
                for r in rows:
                    by_race.setdefault(r.get('rcNo'), []).append(r)
                for rc_no in sorted(by_race, key=lambda x: (x is None, x)):
                    rs = by_race[rc_no]
                    filled = lambda k: sum(1 for r in rs if r.get(k) not in (None, '', '-', 0))
                    key = (meet, rc_no)
                    first = key not in seen
                    if first:
                        seen[key] = now()
                        print(f'[{now()}] ★ 신규 노출  {MEETS[meet]} {rc_no}R  {len(rs)}두 '
                              f'착순={filled("ord")} 기록={filled("rcTime")} 배당={filled("winOdds")} '
                              f'날씨={rs[0].get("weather")} 주로={rs[0].get("track")}')
                    w.writerow([now(), MEETS[meet], rc_no, len(rs), filled('ord'),
                                filled('rcTime'), filled('winOdds'),
                                rs[0].get('weather'), rs[0].get('track')])
                f.flush()
            time.sleep(every)
    except KeyboardInterrupt:
        print(f'\n[{now()}] 중단. 총 {kra.calls}콜.')
        print('최초 노출 시각:')
        for (meet, rc_no), t in sorted(seen.items()):
            print(f'   {MEETS[meet]:6s} {rc_no}R  →  {t}')
        print(f'\n{out} 을 경주별 실제 발주/종료 시각(race.kra.co.kr 경주시간표)과 대조해')
        print('"경주 종료 → API 노출" 지연을 산출할 것.')
    finally:
        f.close()

if __name__ == '__main__':
    main()
