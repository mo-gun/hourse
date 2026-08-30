# -*- coding: utf-8 -*-
"""미신청 API 후보의 공개 참고문서를 수확 (인증 불필요). 기존 specs.json 은 건드리지 않는다."""
import sys, json, io, os
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')
from harvest import spec

IDS = """15105087 말훈련내역
15063955 경주마장제정보
15059488 씨수말정보
15058115 경주마상세정보
15057985 마필종합상세정보
15056828 기수상세정보
15056591 기수성적정보
15119558 당일확정배당율종합
15089503 경주마명단
15086329 현직기수정보
15057090 복승식확정배당율
15059267 승식별최고배당률
15063972 발매금액정보
15058677 출전표상세정보""".strip().split('\n')

out=[]
for line in IDS:
    did, label = line.split()
    try: s = spec(did)
    except Exception as e: s = {'id':did,'error':f'{type(e).__name__}: {e}'[:160]}
    s['label']=label; out.append(s)
    print(f"{did}  {label:14s} ep={str(s.get('endpoint'))[:62]:64s} resp={s.get('n_resp','-'):>3}  {s.get('error','')}")
io.open('kra_specs/specs_candidates.json','w',encoding='utf-8').write(json.dumps(out,ensure_ascii=False,indent=1))
print('\nsaved kra_specs/specs_candidates.json')
