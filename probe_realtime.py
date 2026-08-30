# -*- coding: utf-8 -*-
"""
실시간성 실측 — 경마 시행일 발주 시간대에 실행할 것.

측정 대상:
  1) 경주 전에 배당률(=인기도)이 조회되는가?           → 결론: 안 됨 (발주 후에만)
  2) 당일 필드(마체중·날씨·주로)가 언제 채워지는가?      → 마체중 ~T-1h, 날씨/주로 ~T-10m
  3) 경주 종료 → API 반영 지연

주의: 이 API는 빈 값을 NULL 이 아니라 플레이스홀더로 준다.
      wgHr='0()'  track=' (0%)'  weather='-'  ord/winOdds='0'
      단순 truthy 검사로 세면 '채워진 것처럼' 보인다. FILLED() 를 쓸 것.
"""
import urllib.request, urllib.parse, ssl, json, sys
from datetime import datetime
sys.stdout.reconfigure(encoding='utf-8')

CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE
KEY = [l.split('=', 1)[1].strip() for l in open('.env', encoding='utf-8')
       if l.startswith('KRA_API_KEY_ENCODED')][0]
BASE = 'https://apis.data.go.kr/B551015'

PLACEHOLDER = {'', '-', '0', '0()', ' (0%)', '(0%)', '0.0'}
def FILLED(v):
    s = str(v).strip()
    return s not in PLACEHOLDER and not s.startswith('0()')

def call(path, **p):
    q = {'serviceKey': KEY, '_type': 'json', 'numOfRows': '300', 'pageNo': '1'}
    q.update({k: str(v) for k, v in p.items() if v is not None})
    url = f'{BASE}/{path}?' + '&'.join(
        f'{k}={v if k == "serviceKey" else urllib.parse.quote(str(v))}' for k, v in q.items())
    raw = urllib.request.urlopen(
        urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}),
        timeout=30, context=CTX).read().decode('utf-8', 'replace')
    if not raw.lstrip().startswith('{'):
        return []
    it = (json.loads(raw).get('response', {}).get('body', {}) or {}).get('items') or {}
    it = it.get('item', []) if isinstance(it, dict) else []
    return [it] if isinstance(it, dict) else it

TODAY = datetime.now().strftime('%Y%m%d')
NOW   = datetime.now().strftime('%H:%M')
COLS  = ['wgHr', 'track', 'weather', 'rating', 'ord', 'winOdds']

print(f'=== {TODAY} {NOW} 기준 실측 ===\n')
for meet, name in [(1, '서울'), (2, '제주'), (3, '부산경남')]:
    plan = call('API72_2/racePlan_2', meet=meet, rc_date=TODAY)
    if not plan:
        print(f'[{name}] 오늘 경주 없음\n'); continue
    print(f'[{name}] {len(plan)}경주')
    print('   R  발주   상태 | ' + ' '.join(f'{c:>10}' for c in COLS) + ' | 단승배당')
    for r in sorted(plan, key=lambda x: int(x.get('rcNo', 0))):
        rno = int(r['rcNo'])
        st  = str(r.get('schStTime') or '')
        st  = f'{st[:2]}:{st[2:]}' if len(st) == 4 else st or '  -  '
        rows = call('API4_3/raceResult_3', meet=meet, rc_date=TODAY, rc_no=rno)
        if not rows: continue
        n = len(rows)
        cells = ' '.join(f'{sum(1 for x in rows if FILLED(x.get(c))):>5}/{n:<4}' for c in COLS)
        odds = call('API160_1/integratedInfo_1', meet=meet, rc_date=TODAY, rc_no=rno, pool='WIN')
        print(f'  {rno:>2} {st} {"종료" if st < NOW else "대기"} | {cells} | {len(odds):>4}행')
    print()

print("""판독 —
  · 배당률 0행 + ord 0/n  → 경주 전. 인기도는 공공 API 로 얻을 수 없다.
  · wgHr 만 채워짐        → 발주 1시간 안쪽. 마체중은 당일 예측 입력으로 쓸 수 있다.
  · weather/track 채워짐  → 발주 10분 안쪽.
  · 배당률 n행            → 경주 종료 + API 반영 완료.""")
