# -*- coding: utf-8 -*-
"""승인된 KRA OpenAPI의 참고문서(docx)를 내려받아 엔드포인트/요청/응답 명세를 수확."""
import urllib.request, urllib.parse, re, ssl, json, io, os, html as H
import zipfile, xml.etree.ElementTree as ET
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
UA = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
NS = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
os.makedirs('kra_specs', exist_ok=True)

def fetch(url, referer=None):
    hd=dict(UA)
    if referer: hd['Referer']=referer
    return urllib.request.urlopen(urllib.request.Request(url, headers=hd), timeout=40, context=ctx).read()

def page_info(did):
    h = fetch(f'https://www.data.go.kr/data/{did}/openapi.do').decode('utf-8','replace')
    title = re.search(r'<title>(.*?)</title>', h, re.S)
    title = H.unescape(re.sub(r'\s+',' ',re.sub(r'<[^>]+>','',title.group(1)))).replace('| 공공데이터포털','').strip() if title else did
    files = re.findall(r'<div class="file-name">([^<]+)</div>.*?fn_fileDownload\(\'([^\']+)\',\'(\d+)\'\)', h, re.S)
    traffic = re.search(r'신청 가능 트래픽.*?<div class="value">(.*?)</div>', h, re.S)
    traffic = re.sub(r'\s+',' ',re.sub(r'<[^>]+>','',traffic.group(1))).strip() if traffic else ''
    lic = '제한 없음' if '이용허락범위 제한 없음' in h else ('' if '이용허락범위' not in h else '기타')
    upd = re.search(r'수정일\s*</strong>\s*<div class="value">\s*([0-9\-]+)', h) or re.search(r'([0-9]{4}-[0-9]{2}-[0-9]{2})\s*</div>\s*</li>', h)
    return {'id':did,'title':title,'files':files,'traffic':traffic,'license':lic,
            'updated': upd.group(1) if upd else ''}

def blocks(path):
    z=zipfile.ZipFile(path); root=ET.fromstring(z.read('word/document.xml')); body=root.find(NS+'body')
    def ptext(p): return ''.join(t.text or '' for t in p.iter(NS+'t')).strip()
    out=[]
    for el in body:
        tag=el.tag.replace(NS,'')
        if tag=='p':
            t=ptext(el)
            if t: out.append(('p',t))
        elif tag=='tbl':
            rows=[]
            for tr in el.findall(NS+'tr'):
                cells=[' '.join(ptext(p) for p in tc.findall(NS+'p')).strip() for tc in tr.findall(NS+'tc')]
                if any(cells): rows.append(cells)
            if rows: out.append(('tbl',rows))
    return out

def spec(did):
    info = page_info(did)
    doc = next((f for f in info['files'] if f[0].lower().endswith('.docx')), None)
    info['doc'] = doc[0] if doc else None
    if not doc:
        info['error']='no docx'; return info
    path = f'kra_specs/{did}.docx'
    if not os.path.exists(path):
        b = fetch(f'https://www.data.go.kr/cmm/cmm/fileDownload.do?atchFileId={doc[1]}&fileDetailSn={doc[2]}',
                  referer=f'https://www.data.go.kr/data/{did}/openapi.do')
        open(path,'wb').write(b)
    bl = blocks(path)
    flat = '\n'.join(b[1] if b[0]=='p' else '\n'.join(' | '.join(r) for r in b[1]) for b in bl)
    eps = sorted(set(re.findall(r'https?://apis\.data\.go\.kr/[A-Za-z0-9_/\.\-]+', flat)), key=len, reverse=True)
    info['endpoints'] = eps[:4]
    info['endpoint'] = eps[0] if eps else None
    # operation list table (상세기능 목록)
    ops=[]; req=[]; resp=[]
    for i,(k,v) in enumerate(bl):
        if k!='tbl': continue
        hdr = [c.strip() for c in v[0]]
        if '상세기능명(영문)' in hdr:
            ci = hdr.index('상세기능명(영문)')
            ops += [r[ci] for r in v[1:] if len(r)>ci and r[ci]]
        if hdr[:2]==['항목명(영문)','항목명(국문)']:
            # decide request vs response by preceding paragraph
            prev = next((bl[j][1] for j in range(i-1,-1,-1) if bl[j][0]=='p'), '')
            fields=[{'en':r[0],'ko':r[1] if len(r)>1 else '','size':r[2] if len(r)>2 else '',
                     'req':r[3] if len(r)>3 else '','sample':r[4] if len(r)>4 else '',
                     'desc':r[5] if len(r)>5 else ''} for r in v[1:] if r and r[0]]
            if '요청' in prev: req += fields
            else: resp += fields
    info['operations']=ops; info['request_fields']=req; info['response_fields']=resp
    info['n_resp']=len(resp)
    return info

if __name__=='__main__':
    import sys
    ids = sys.argv[1:]
    out=[]
    for d in ids:
        try: s=spec(d)
        except Exception as e: s={'id':d,'error':f'{type(e).__name__}: {e}'[:200]}
        out.append(s)
        print(f"{s.get('id')}  {s.get('title','')[:44]:46s} ep={str(s.get('endpoint'))[:58]:60s} resp={s.get('n_resp','-')}  {s.get('error','')}")
    io.open('kra_specs/specs.json','w',encoding='utf-8').write(json.dumps(out,ensure_ascii=False,indent=1))
    print('\nsaved kra_specs/specs.json')
