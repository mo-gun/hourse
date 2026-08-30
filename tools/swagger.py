# -*- coding: utf-8 -*-
"""data.go.kr openapi.do 페이지에 인라인된 Swagger JSON 추출 → 엔드포인트/파라미터/응답 스키마."""
import urllib.request, re, ssl, json, io, os, html as H
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
UA={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
os.makedirs('kra_specs', exist_ok=True)

def page(did):
    return urllib.request.urlopen(urllib.request.Request(
        f'https://www.data.go.kr/data/{did}/openapi.do', headers=UA), timeout=40, context=ctx).read().decode('utf-8','replace')

def extract(did):
    h = page(did)
    i = h.find('"swagger"')
    if i < 0:
        i = h.find('"host":"apis.data.go.kr')
        if i < 0: return None
        i = h.rfind('{', 0, i)
    # brace-match forward from the opening brace of the swagger object
    start = h.rfind('{', 0, i+1)
    depth = 0; instr = False; esc = False
    for j in range(start, len(h)):
        c = h[j]
        if instr:
            if esc: esc = False
            elif c == chr(92): esc = True
            elif c == '"': instr = False
            continue
        if c == '"': instr = True
        elif c == '{': depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                blob = h[start:j+1]
                try: return json.loads(H.unescape(blob))
                except Exception:
                    try: return json.loads(blob)
                    except Exception as e: return {'_parse_error': str(e)[:120], '_len': len(blob)}
    return None

def summarize(did, sw):
    if not sw or '_parse_error' in (sw or {}): return {'id':did, 'error': (sw or {}).get('_parse_error','no swagger')}
    host = sw.get('host',''); base = sw.get('basePath','')
    info = sw.get('info',{})
    out = {'id':did, 'title':info.get('title',''), 'host':host, 'basePath':base,
           'desc':(info.get('description','') or '')[:400], 'ops':[]}
    for path, methods in (sw.get('paths') or {}).items():
        for m, d in methods.items():
            params=[{'name':p.get('name'),'req':p.get('required'),'desc':(p.get('description') or '')[:60],
                     'type':p.get('type'),'default':p.get('default')} for p in (d.get('parameters') or [])]
            # response schema fields
            fields=[]
            for code,r in (d.get('responses') or {}).items():
                sch = r.get('schema') or {}
                def walk(s, prefix=''):
                    if not isinstance(s, dict): return
                    for k,v in (s.get('properties') or {}).items():
                        t=v.get('type'); desc=(v.get('description') or '')[:50]
                        if t in ('object','array') or 'properties' in v or 'items' in v:
                            fields.append({'name':prefix+k,'type':t,'desc':desc})
                            walk(v.get('items') or v, prefix+k+'.')
                        else:
                            fields.append({'name':prefix+k,'type':t,'desc':desc,'example':v.get('example')})
                walk(sch)
            out['ops'].append({'path':path,'method':m,'summary':d.get('summary',''),
                               'params':params,'fields':fields})
    return out

if __name__=='__main__':
    import sys
    res=[]
    for did in sys.argv[1:]:
        try: s=summarize(did, extract(did))
        except Exception as e: s={'id':did,'error':f'{type(e).__name__}: {e}'[:160]}
        res.append(s)
        if 'error' in s: print(f"{did}  ERROR {s['error']}"); continue
        for o in s['ops']:
            full=f"{s['host'].rstrip('/')}{s['basePath']}{o['path']}"
            print(f"{did}  {s['title'][:34]:36s} https://{full:52s} params={len(o['params'])} fields={len(o['fields'])}")
    io.open('kra_specs/swagger.json','w',encoding='utf-8').write(json.dumps(res,ensure_ascii=False,indent=1))
    print('\nsaved kra_specs/swagger.json')
