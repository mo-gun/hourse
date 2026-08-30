# -*- coding: utf-8 -*-
"""
출마표 스냅샷 · 변동 감시 · 공개 시점 실측.

세 가지 질문에 답한다.
  1. 게이트(출주번호)는 언제부터 조회되나?      → `python watch_entries.py horizon`
  2. 공개된 출마표가 발주까지 바뀌나?            → `python watch_entries.py snap` 반복 후 `diff`
  3. 다음 주말 출마표는 정확히 몇 시에 뜨나?      → `python watch_entries.py wait`

스냅샷은 snapshots/entries_<주말>_<타임스탬프>.json 으로 쌓인다.
diff 는 가장 최근 두 스냅샷을 비교해 출전취소·기수교체·게이트변경·부담중량변경을 뽑는다.
"""
import sys, os, io, json, time, datetime, collections
from kra_client import KRA, KRAError

SNAP = 'snapshots'
KEYFIELDS = ['chulNo', 'jkNo', 'jkName', 'wgBudam', 'rating', 'trNo', 'trName']


def weekend(offset_weeks=0, today=None):
    t = today or datetime.date.today()
    fri = t + datetime.timedelta(days=(4 - t.weekday()) % 7, weeks=offset_weeks)
    return [(fri + datetime.timedelta(days=i)).strftime('%Y%m%d') for i in range(3)]


def pull(kra, dates):
    rows = []
    for d in dates:
        for meet in (1, 2, 3):
            try:
                rows += kra.race_result(rc_date=d, meet=meet)
            except KRAError:
                pass
    return rows


def rowkey(r):
    """말 1두를 경주 안에서 식별 — 게이트가 바뀌어도 추적되도록 마번 기준."""
    return (str(r['rcDate']), str(r['meet']), str(r['rcNo']), str(r['hrNo']))


def cmd_snap():
    kra = KRA()
    dates = weekend()
    rows = pull(kra, dates)
    os.makedirs(SNAP, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = f'{dates[0]}-{dates[-1][4:]}'
    path = f'{SNAP}/entries_{tag}_{stamp}.json'
    io.open(path, 'w', encoding='utf-8').write(json.dumps(
        {'taken_at': datetime.datetime.now().isoformat(timespec='seconds'),
         'dates': dates, 'rows': rows}, ensure_ascii=False))
    races = len({(r['rcDate'], r['meet'], r['rcNo']) for r in rows})
    print(f'스냅샷 저장 {path}')
    print(f'  {len(rows):,}두 / {races}개 경주 / {kra.calls}콜')
    return path


def cmd_diff():
    files = sorted(f for f in os.listdir(SNAP) if f.endswith('.json')) if os.path.isdir(SNAP) else []
    if len(files) < 2:
        print(f'스냅샷이 {len(files)}개뿐. `python watch_entries.py snap` 을 시간 간격을 두고 2회 이상 실행할 것.')
        return
    a = json.load(io.open(f'{SNAP}/{files[-2]}', encoding='utf-8'))
    b = json.load(io.open(f'{SNAP}/{files[-1]}', encoding='utf-8'))
    print(f'비교: {files[-2]}  ({a["taken_at"]})')
    print(f'  →   {files[-1]}  ({b["taken_at"]})\n')
    A = {rowkey(r): r for r in a['rows']}
    B = {rowkey(r): r for r in b['rows']}
    gone = [A[k] for k in A if k not in B]
    new = [B[k] for k in B if k not in A]
    chg = []
    for k in A.keys() & B.keys():
        d = {f: (A[k].get(f), B[k].get(f)) for f in KEYFIELDS
             if str(A[k].get(f)) != str(B[k].get(f))}
        if d:
            chg.append((B[k], d))
    print(f'출전취소(사라짐) {len(gone)}두 · 추가 {len(new)}두 · 항목변경 {len(chg)}두\n')
    for r in gone[:20]:
        print(f'  [취소] {r["rcDate"]} {r["meet"]} {r["rcNo"]}R  {r["chulNo"]}번 {r["hrName"]} ({r["jkName"]})')
    for r in new[:20]:
        print(f'  [추가] {r["rcDate"]} {r["meet"]} {r["rcNo"]}R  {r["chulNo"]}번 {r["hrName"]} ({r["jkName"]})')
    for r, d in chg[:30]:
        bits = ', '.join(f'{f}: {o}→{n}' for f, (o, n) in d.items())
        print(f'  [변경] {r["rcDate"]} {r["meet"]} {r["rcNo"]}R  {r["hrName"]:10s} {bits}')
    if not (gone or new or chg):
        print('  변동 없음 — 이 구간에서 출마표는 안정적이었다.')


def cmd_horizon():
    kra = KRA()
    today = datetime.date.today()
    print(f'오늘 {today} ({"월화수목금토일"[today.weekday()]})\n')
    print(f'{"날짜":10s} {"요일":>3s} {"D+":>4s}  {"서울":>5s} {"제주":>5s} {"부경":>5s}  {"게이트채움":>8s}')
    print('-' * 60)
    found = []
    for off in range(0, 29):
        d = today + datetime.timedelta(days=off)
        ds = d.strftime('%Y%m%d')
        cnt, gates = {}, 0
        tot = 0
        for meet, nm in [(1, '서울'), (2, '제주'), (3, '부경')]:
            try:
                rows = kra.fetch('race_result', num_rows=1000, rc_date=ds, meet=meet)
            except KRAError:
                rows = []
            cnt[nm] = len(rows)
            tot += len(rows)
            gates += sum(1 for r in rows if str(r.get('chulNo', '')).strip() not in ('', '0', '-'))
        if not tot:
            continue
        found.append(off)
        print(f'{ds:10s} {"월화수목금토일"[d.weekday()]:>3s} {"D+"+str(off):>4s}  '
              f'{cnt["서울"]:>5d} {cnt["제주"]:>5d} {cnt["부경"]:>5d}  {gates*100//tot:>7d}%')
    print()
    if found:
        print(f'조회 가능한 최장 선행일수: D+{max(found)}  (그 이후는 미편성/미공개)')
    print(f'{kra.calls}콜 사용')


def cmd_wait(every=600, limit=72 * 60):
    """다음 주말 출마표가 API에 처음 뜨는 순간을 기록한다."""
    kra = KRA()
    dates = weekend(1)
    print(f'감시 대상: 다음 주말 {dates} · {every}s 간격 · 최대 {limit//60}시간')
    t0 = time.time()
    while time.time() - t0 < limit * 60:
        rows = pull(kra, dates)
        stamp = datetime.datetime.now()
        if rows:
            print(f'\n[{stamp:%Y-%m-%d %H:%M:%S}] ★ 출마표 등장 — {len(rows):,}두')
            os.makedirs(SNAP, exist_ok=True)
            io.open(f'{SNAP}/first_seen_{dates[0]}.json', 'w', encoding='utf-8').write(json.dumps(
                {'first_seen': stamp.isoformat(timespec='seconds'), 'dates': dates,
                 'n_rows': len(rows), 'rows': rows}, ensure_ascii=False))
            byd = collections.Counter((r['rcDate'], r['meet']) for r in rows)
            for k, v in sorted(byd.items()):
                print(f'   {k[0]} {k[1]}: {v}두')
            print(f'\n기록 저장 → {SNAP}/first_seen_{dates[0]}.json')
            print(f'선행일수 = D+{(datetime.datetime.strptime(dates[0], "%Y%m%d").date() - stamp.date()).days}')
            return
        print(f'[{stamp:%H:%M:%S}] 아직 없음 ({kra.calls}콜)')
        time.sleep(every)
    print('시간 초과.')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'snap'
    if cmd == 'snap':      cmd_snap()
    elif cmd == 'diff':    cmd_diff()
    elif cmd == 'horizon': cmd_horizon()
    elif cmd == 'wait':    cmd_wait(int(sys.argv[2]) if len(sys.argv) > 2 else 600)
    else:                  print(__doc__)
