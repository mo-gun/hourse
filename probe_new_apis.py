# -*- coding: utf-8 -*-
"""미신청 API 후보 실호출 — 승인 필요 여부/엔드포인트 생존 확인."""
import urllib.request, urllib.parse, ssl, json, sys, re
sys.stdout.reconfigure(encoding='utf-8')
CTX = ssl.create_default_context(); CTX.check_hostname=False; CTX.verify_mode=ssl.CERT_NONE
KEY = [l.split('=',1)[1].strip() for l in open('.env',encoding='utf-8') if l.startswith('KRA_API_KEY_ENCODED')][0]
BASE='https://apis.data.go.kr/B551015'

CAND = [
  ('말훈련내역',        'trainstate/gettrainstate',            {'tr_date_fr':'20260801','tr_date_to':'20260827'}),
  ('말훈련내역(alt)',   'API185/trainState',                   {}),
  ('경주마장제정보',    'jangjeinfo/getjangjeinfo',            {}),
  ('씨수말정보',        'API33/stallionInfo',                  {}),
  ('경주마명단',        'racehorselist/getracehorselist',      {}),
  ('현직기수정보',      'currentjockeyInfo/getcurrentjockeyinfo', {}),
  ('퇴직기수정보',      'retiredjockeyInfo/getretiredjockeyinfo', {}),
  ('경주별상세성적표',  'racedetailresult/getracedetailresult', {'meet':'1'}),
  ('당일경주결과종합',  'API214_1/RaceDetailResult_1',         {}),
  ('발매금액정보',      'API224/salesAmount',                  {}),
  ('경마경주정보',      'API187/HorseRaceInfo',                {'ym_fr':'202601','ym_to':'202608'}),
  ('경주마상세정보',    'API8_2/raceHorseInfo_2',              {}),
  ('마필종합상세정보',  'API19_1/totalHorseInfo_1',            {}),
  ('기수상세정보',      'API12_1/jockeyInfo_1',                {}),
]
for name, path, extra in CAND:
    q = {'serviceKey':KEY,'_type':'json','numOfRows':'3','pageNo':'1'}; q.update(extra)
    url = f'{BASE}/{path}?' + '&'.join(f'{k}={v if k=="serviceKey" else urllib.parse.quote(str(v))}' for k,v in q.items())
    try:
        raw = urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'}), timeout=25, context=CTX).read().decode('utf-8','replace')
    except Exception as e:
        code = getattr(e,'code',None)
        print(f'{name:20s} {path:42s} → HTTP {code or type(e).__name__}'); continue
    if raw.lstrip().startswith('{'):
        d=json.loads(raw); h=d.get('response',{}).get('header',{})
        b=d.get('response',{}).get('body',{}) or {}
        print(f'{name:20s} {path:42s} → {h.get("resultCode")} {str(h.get("resultMsg"))[:34]:36s} total={b.get("totalCount")}')
    else:
        m = re.search(r'<returnAuthMsg>(.*?)</returnAuthMsg>', raw) or re.search(r'<errMsg>(.*?)</errMsg>', raw) or re.search(r'<resultMsg>(.*?)</resultMsg>', raw)
        print(f'{name:20s} {path:42s} → XML {(m.group(1) if m else raw[:70]).strip()[:60]}')
