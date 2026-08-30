# -*- coding: utf-8 -*-
"""
새로 신청한 API의 인증키 반영을 기다렸다가, 살아나면 즉시 데이터를 받아 CSV로 떨군다.

data.go.kr 은 활용신청 승인 직후에도 게이트웨이에 키가 전파되기까지 시간이 걸린다
(그 사이 응답은 HTTP 403 + SERVICE_KEY_IS_NOT_REGISTERED_ERROR).

    python wait_raceplan.py            # 최대 2시간, 5분 간격
    python wait_raceplan.py 30 240     # 30초 간격, 240분까지
"""
import sys, time, io, os, csv, json, datetime, urllib.request, urllib.parse, ssl

_CTX = ssl.create_default_context(); _CTX.check_hostname = False; _CTX.verify_mode = ssl.CERT_NONE
BASE = 'https://apis.data.go.kr/B551015'
OUT = 'data'

TARGETS = {
    # 이름                 경로                                    이번 주말 조회 파라미터
    'race_plan':         ('API154/racePlan',                      lambda d, m: {'rccrs_cd': str(m), 'race_dt': d}),
    'race_detail':       ('racedetailresult/getracedetailresult',  lambda d, m: {'meet': str(m), 'rc_date': d}),
    'ai_race_detail':    ('API156/raceRsutDtl',                    lambda d, m: {'meet': str(m), 'rc_date': d}),
    'same_day_total':    ('Race_Result_total',                     lambda d, m: {'meet': str(m), 'rc_date': d}),
}


def key():
    if os.environ.get('KRA_API_KEY_ENCODED'):
        return os.environ['KRA_API_KEY_ENCODED']
    for line in io.open('.env', encoding='utf-8'):
        if line.startswith('KRA_API_KEY_ENCODED'):
            return line.split('=', 1)[1].strip()
    raise SystemExit('.env 에 KRA_API_KEY_ENCODED 없음')


K = key()


def call(path, params):
    p = dict(params); p.setdefault('numOfRows', '500'); p.setdefault('pageNo', '1'); p.setdefault('_type', 'json')
    url = f'{BASE}/{path}?serviceKey={K}&' + urllib.parse.urlencode(p)
    req = urllib.request.Request(url, headers={'User-Agent': 'kra-client/1.0', 'Accept': '*/*'})
    try:
        return 200, urllib.request.urlopen(req, timeout=40, context=_CTX).read().decode('utf-8', 'replace')
    except urllib.error.HTTPError as e:
        return e.code, (e.read().decode('utf-8', 'replace') if e.fp else '')
    except Exception as e:
        return 0, str(e)[:200]


def status(path, params):
    code, body = call(path, params)
    if 'SERVICE_KEY_IS_NOT_REGISTERED' in body: return 'NOKEY', None
    if 'SERVICE_ACCESS_DENIED' in body:         return 'DENIED', None
    if 'NO_OPENAPI_SERVICE' in body:            return 'NOPATH', None
    if 'LIMITED_NUMBER' in body:                return 'QUOTA', None
    if code != 200:                             return f'HTTP{code}', None
    try:
        b = json.loads(body)['response']['body']
    except Exception:
        return 'OTHER', None
    items = (b.get('items') or {}).get('item') or []
    items = items if isinstance(items, list) else [items]
    return 'OK', items


def dates():
    t = datetime.date.today()
    fri = t + datetime.timedelta(days=(4 - t.weekday()) % 7)
    return [(fri + datetime.timedelta(days=i)).strftime('%Y%m%d') for i in range(3)]


def dump(name, rows):
    if not rows:
        print(f'    {name}: 0행'); return
    cols = []
    for r in rows:
        for k in r:
            if k not in cols: cols.append(k)
    os.makedirs(OUT, exist_ok=True)
    ds = dates()
    p = f'{OUT}/{name}_{ds[0]}-{ds[-1][4:]}.csv'
    with io.open(p, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore'); w.writeheader()
        for r in rows: w.writerow({c: r.get(c, '') for c in cols})
    print(f'    저장 {p}  {len(rows):,}행 × {len(cols)}열')


def harvest(name, path, mk):
    rows = []
    for d in dates():
        for meet in (1, 2, 3):
            st, items = status(path, mk(d, meet))
            if st == 'OK' and items:
                rows += items
                print(f'    {d} meet={meet}: {len(items)}행')
            elif st != 'OK':
                print(f'    {d} meet={meet}: {st}')
    dump(name, rows)
    return len(rows)


def main():
    every = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 120     # 분
    d0 = dates()[0]
    pending = dict(TARGETS)
    t0 = time.time()
    print(f'대기 시작 — 대상 {list(pending)} · {every}s 간격 · 최대 {limit}분')
    while pending and (time.time() - t0) < limit * 60:
        for name in list(pending):
            path, mk = pending[name]
            st, _ = status(path, mk(d0, 1))
            stamp = datetime.datetime.now().strftime('%H:%M:%S')
            if st == 'OK':
                print(f'[{stamp}] ★ {name} 활성화 — 수집 시작 ({path})')
                harvest(name, path, mk)
                del pending[name]
            else:
                print(f'[{stamp}]   {name}: {st}')
        if pending:
            time.sleep(every)
    if pending:
        print(f'\n시간 초과. 아직 미반영: {list(pending)}')
        print('data.go.kr > 마이페이지 > 개발계정 에서 승인 상태를 확인할 것.')
    else:
        print('\n전부 수집 완료.')


if __name__ == '__main__':
    main()
