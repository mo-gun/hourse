# -*- coding: utf-8 -*-
"""
이번 주말 경주 데이터셋 빌더.

만드는 것 (data/ 폴더):
  1. entries_<주말>.csv      이번 주말 출마표 — 발주 전 확정 필드만 (예측 입력)
  2. entries_<주말>_full.csv 이번 주말 원본 전체 필드 (빈 결과 필드 포함)
  3. result_<날짜>_<장>.csv  가장 최근 완료 경주일 — 결과 포함 전체 (정답지 예시)
  4. jockeys_<주말>.csv      이번 주말 출전 기수별 통산 성적 — 원장에서 컷오프 기준 집계
  5. trainers_<주말>.csv     동일, 조교사
  6. README.md               각 파일에 뭐가 들었고 어느 필드가 언제 채워지는지

기수/조교사 성적은 **별도 API를 쓰지 않고** 경주기록 원장에서
"컷오프 날짜 이전" 행만으로 계산한다 → 미래 정보 누수가 원천적으로 없다.
(KRA 의 기수/조교사 전적 API 는 '오늘 기준 누적 스냅샷'이라 과거 예측에 쓰면 누수가 난다.)
"""
import io, os, csv, collections, datetime, sys
from kra_client import KRA, KRAError

OUT = 'data'
HIST_FROM = 2023          # 기수/조교사 집계에 쓸 과거 범위
BLANKISH = ('', '-', '0', '0()', ' (0%)', '(0%)')

# 발주 전 확정 — 예측 입력으로 쓸 수 있는 필드
PRE_RACE = ['rcDate', 'rcDay', 'meet', 'rcNo', 'rcName', 'rcDist', 'rank', 'budam', 'prizeCond',
            'ageCond', 'sexCond', 'chulNo', 'hrNo', 'hrName', 'hrNameEn', 'name', 'age', 'sex',
            'ilsu', 'rating', 'wgBudam', 'jkNo', 'jkName', 'trNo', 'trName', 'owNo', 'owName',
            'chaksun1', 'chaksun2', 'chaksun3', 'chaksun4', 'chaksun5']
# 당일 또는 경주 후에만 채워짐
POST_RACE = ['wgHr', 'weather', 'track', 'winOdds', 'plcOdds', 'ord', 'rcTime', 'diffUnit']


def blank(v):
    if v is None:
        return True
    s = str(v).strip()
    return s in BLANKISH or s.startswith('0(') or s.startswith(' (0')


def write_csv(path, rows, cols=None):
    if not rows:
        print(f'  (건너뜀, 0행) {path}')
        return 0
    if cols is None:
        cols = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with io.open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in cols})
    print(f'  {path}  {len(rows):,}행 × {len(cols)}열')
    return len(rows)


def upcoming_dates(today=None):
    """오늘 기준 이번 주 금·토·일."""
    today = today or datetime.date.today()
    fri = today + datetime.timedelta(days=(4 - today.weekday()) % 7)
    return [(fri + datetime.timedelta(days=i)).strftime('%Y%m%d') for i in range(3)]


def latest_completed(kra, before, lookback=21):
    """before 이전에서 결과(ord)가 채워진 가장 최근 (날짜, 경마장) 찾기."""
    d = datetime.datetime.strptime(before, '%Y%m%d').date()
    for _ in range(lookback):
        d -= datetime.timedelta(days=1)
        ds = d.strftime('%Y%m%d')
        for meet in (1, 2, 3):
            try:
                rows = kra.race_result(rc_date=ds, meet=meet)
            except KRAError:
                continue
            if rows and sum(1 for r in rows if not blank(r.get('ord'))) > len(rows) * 0.8:
                return ds, meet, rows
    return None, None, []


def person_stats(hist, id_key, name_key, cutoff, keep_ids):
    """원장에서 cutoff(YYYYMMDD) '이전' 행만으로 인물별 전적 집계 — 누수 없음."""
    agg = collections.defaultdict(lambda: {'출주': 0, '1착': 0, '2착': 0, '3착': 0,
                                           '착순합': 0, '착순건': 0, '이름': ''})
    for r in hist:
        if str(r.get('rcDate', '')) >= cutoff:
            continue                       # 컷오프 이후는 미래 → 제외
        pid = str(r.get(id_key, '')).strip()
        if not pid or pid not in keep_ids:
            continue
        o = r.get('ord')
        a = agg[pid]
        a['이름'] = r.get(name_key) or a['이름']
        a['출주'] += 1
        if not blank(o):
            try:
                o = int(o)
            except (TypeError, ValueError):
                continue
            a['착순합'] += o
            a['착순건'] += 1
            if o == 1: a['1착'] += 1
            elif o == 2: a['2착'] += 1
            elif o == 3: a['3착'] += 1
    out = []
    for pid, a in agg.items():
        n = a['출주'] or 1
        out.append({
            'id': pid, '이름': a['이름'], '출주수': a['출주'],
            '1착': a['1착'], '2착': a['2착'], '3착': a['3착'],
            '승률%': round(a['1착'] * 100 / n, 1),
            '복승률%': round((a['1착'] + a['2착']) * 100 / n, 1),
            '삼복승률%': round((a['1착'] + a['2착'] + a['3착']) * 100 / n, 1),
            '평균착순': round(a['착순합'] / a['착순건'], 2) if a['착순건'] else '',
        })
    return sorted(out, key=lambda x: (-x['출주수'], x['이름']))


def main():
    kra = KRA()
    os.makedirs(OUT, exist_ok=True)
    dates = upcoming_dates()
    tag = f'{dates[0]}-{dates[-1][4:]}'
    print(f'대상 주말: {dates}')

    # ── 1) 이번 주말 출마표 ─────────────────────────────
    print('\n[1] 이번 주말 출마표 수집')
    entries = []
    for d in dates:
        for meet in (1, 2, 3):
            try:
                rows = kra.race_result(rc_date=d, meet=meet)
            except KRAError as e:
                print(f'  {d} meet={meet}: {e}')
                continue
            if rows:
                print(f'  {d} {rows[0]["meet"]}: {len(rows)}두')
                entries += rows
    entries.sort(key=lambda r: (str(r.get('rcDate')), str(r.get('meet')),
                                int(r.get('rcNo') or 0), int(r.get('chulNo') or 0)))
    write_csv(f'{OUT}/entries_{tag}.csv', entries, PRE_RACE)
    write_csv(f'{OUT}/entries_{tag}_full.csv', entries)

    # ── 2) 가장 최근 완료 경주일 (정답지 예시) ───────────
    print('\n[2] 최근 완료 경주일 (결과 포함)')
    ds, meet, done = latest_completed(kra, dates[0])
    if done:
        mname = done[0]['meet']
        print(f'  {ds} {mname} — {len(done)}두')
        write_csv(f'{OUT}/result_{ds}_{mname}.csv', done)

    # ── 3) 기수·조교사 통산 성적 (누수 없는 집계) ────────
    print(f'\n[3] 과거 원장 수집 ({HIST_FROM}~) — 기수/조교사 집계용')
    hist = []
    this_year = int(dates[0][:4])
    for y in range(HIST_FROM, this_year + 1):
        n0 = len(hist)
        for m in range(1, 13):
            for meet in (1, 2, 3):
                try:
                    hist += kra.race_result(rc_month=f'{y}{m:02d}', meet=meet)
                except KRAError:
                    pass
        print(f'  {y}: +{len(hist)-n0:,}행 (누적 {len(hist):,}, {kra.calls}콜)')

    cutoff = dates[0]                       # 이번 주말 첫날 '이전'까지만 사용
    jk_ids = {str(r.get('jkNo', '')).strip() for r in entries}
    tr_ids = {str(r.get('trNo', '')).strip() for r in entries}
    jk = person_stats(hist, 'jkNo', 'jkName', cutoff, jk_ids)
    tr = person_stats(hist, 'trNo', 'trName', cutoff, tr_ids)
    print(f'\n[4] 인물 집계 (컷오프 {cutoff} 이전만 사용 → 누수 없음)')
    write_csv(f'{OUT}/jockeys_{tag}.csv', jk)
    write_csv(f'{OUT}/trainers_{tag}.csv', tr)

    # ── 4) README ───────────────────────────────────────
    pre_ok = [k for k in PRE_RACE if entries and
              sum(1 for r in entries if not blank(r.get(k))) > len(entries) * 0.5]
    readme = f'''# 이번 주말 경주 데이터 ({dates[0]} ~ {dates[-1]})

생성 `{datetime.date.today()}` · 출처 한국마사회 공공데이터 `API4_3/raceResult_3` · API 호출 {kra.calls}회

## 핵심 — 결과 API가 "아직 안 뛴 경기"의 출마표도 준다

`경주기록 정보`는 이름과 달리 **경주 전 엔트리부터 행이 존재**한다.
결과 필드만 비어 있고, 발주 전 확정 정보는 이미 다 들어 있다.
**출마표용 API(`API154/racePlan`)를 따로 신청하지 않아도 경기 전 예측이 가능하다.**

## 파일

| 파일 | 내용 | 행수 |
|---|---|---|
| `entries_{tag}.csv` | 이번 주말 출마표 — **발주 전 확정 필드만**. 예측 모델 입력용 | {len(entries):,} |
| `entries_{tag}_full.csv` | 위와 같은 행, **원본 전체 필드**(빈 결과 필드 포함) | {len(entries):,} |
| `result_{ds}_{done[0]["meet"] if done else ""}.csv` | 최근 완료 경주일 — **결과 포함 전체 필드**. 정답지 예시 | {len(done):,} |
| `jockeys_{tag}.csv` | 이번 주말 출전 기수별 통산 성적 | {len(jk):,} |
| `trainers_{tag}.csv` | 이번 주말 출전 조교사별 통산 성적 | {len(tr):,} |

## 필드가 채워지는 시점

**발주 전 확정** — 예측 입력으로 쓸 수 있다:
`{', '.join(pre_ok)}`

**당일 또는 경주 후에만** — 경기 전 예측에서는 못 쓴다:
`{', '.join(POST_RACE)}` + 구간기록 전체(`se*` `bu*` `je*` `sj*`)

> `wgHr`(마체중) `track`(주로상태) `weather`(날씨)는 **당일** 확정이다.
> 전날 예측 모델을 만들 거라면 이 셋을 빼거나 예보/직전값으로 대체해야 한다.
> 배당률(`winOdds` `plcOdds`)은 **확정치라 경주 후에 들어온다** — 발주 전 예측의 입력이 될 수 없고,
> 성능 비교의 **시장 베이스라인**으로만 쓴다.

## 기수·조교사 성적을 만든 방법 (중요)

KRA 의 기수/조교사 전적 API는 **오늘 기준 누적 스냅샷**이라, 과거 경주 행에 붙이면
모델이 미래를 안 채로 예측하게 된다(누수).

그래서 이 파일은 **경주기록 원장 {HIST_FROM}~{this_year}년치({len(hist):,}행)에서
`rcDate < {cutoff}` 인 행만으로 직접 집계**했다. 이번 주말을 예측하는 시점에
실제로 알 수 있었던 정보만 들어 있다.

학습 데이터를 만들 때도 같은 방식이어야 한다 — 각 (말, 경주일)마다
**그 경주일 이전** 행만으로 rolling 집계할 것.

## 재현

```bash
python build_weekend.py
```
'''
    io.open(f'{OUT}/README.md', 'w', encoding='utf-8').write(readme)
    print(f'  {OUT}/README.md')
    print(f'\n총 {kra.calls}콜 사용 (개발계정 한도 3,000/일)')


if __name__ == '__main__':
    main()
