# -*- coding: utf-8 -*-
"""data.go.kr 참고문서(docx) → 엔드포인트 / 요청 파라미터 / 응답 필드 추출."""
import zipfile, re, sys, io, json
NS = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
import xml.etree.ElementTree as ET

def docx_blocks(path):
    """문서를 순서대로 [('p', text) | ('tbl', [[cell,...],...])] 로."""
    z = zipfile.ZipFile(path)
    root = ET.fromstring(z.read('word/document.xml'))
    body = root.find(NS+'body')
    out=[]
    def ptext(p):
        return ''.join(t.text or '' for t in p.iter(NS+'t')).strip()
    for el in body:
        tag = el.tag.replace(NS,'')
        if tag=='p':
            t=ptext(el)
            if t: out.append(('p',t))
        elif tag=='tbl':
            rows=[]
            for tr in el.findall(NS+'tr'):
                cells=[]
                for tc in tr.findall(NS+'tc'):
                    cells.append(' '.join(ptext(p) for p in tc.findall(NS+'p')).strip())
                if any(cells): rows.append(cells)
            if rows: out.append(('tbl',rows))
    return out

def summarize(path):
    blocks = docx_blocks(path)
    text = '\n'.join(b[1] if b[0]=='p' else '\n'.join(' | '.join(r) for r in b[1]) for b in blocks)
    eps = sorted(set(re.findall(r'https?://[A-Za-z0-9\.\-]*data\.go\.kr/[A-Za-z0-9_/\.\-]+', text)))
    return blocks, text, eps

if __name__=='__main__':
    p = sys.argv[1]
    blocks, text, eps = summarize(p)
    print('ENDPOINTS:', eps)
    print('BLOCKS:', len(blocks))
    for kind, val in blocks[:60]:
        if kind=='p':
            print('P  ', val[:150])
        else:
            print(f'TBL rows={len(val)} cols={max(len(r) for r in val)}')
            for r in val[:14]:
                print('    ', ' | '.join(c[:34] for c in r))
            if len(val)>14: print(f'     ... +{len(val)-14} rows')
