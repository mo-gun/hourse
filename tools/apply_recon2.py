# -*- coding: utf-8 -*-
"""2단계: 실제 활용신청 폼 열어서 구조 덤프."""
import sys, os, io, time
sys.stdout.reconfigure(encoding='utf-8')
from playwright.sync_api import sync_playwright

PROFILE = r"C:\Users\SSAFY\AppData\Local\Temp\claude\c--Users-SSAFY-Desktop-----\5200d35d-bce2-4650-a88b-15ceb3e9d1f1\scratchpad\chrome-profile"
OUT     = r"C:\Users\SSAFY\AppData\Local\Temp\claude\c--Users-SSAFY-Desktop-----\5200d35d-bce2-4650-a88b-15ceb3e9d1f1\scratchpad"
DID = "15105087"

with sync_playwright() as p:
    ctx = p.chromium.launch_persistent_context(PROFILE, channel="chrome", headless=False,
                                               args=["--start-maximized"], no_viewport=True)
    pg = ctx.pages[0] if ctx.pages else ctx.new_page()
    popups = []
    ctx.on("page", lambda pp: popups.append(pp))

    pg.goto(f"https://www.data.go.kr/data/{DID}/openapi.do", timeout=60000)
    pg.wait_for_timeout(2000)
    body = pg.inner_text("body")
    print("로그인 상태:", "OK" if ("로그아웃" in body or "마이페이지" in body) else "NO", flush=True)

    print("fn_goOpenAPIRequestForm 존재:",
          pg.evaluate("typeof fn_goOpenAPIRequestForm"), flush=True)
    try:
        pg.evaluate(f"fn_goOpenAPIRequestForm('{DID}')")
    except Exception as e:
        print("eval err:", e, flush=True)
    pg.wait_for_timeout(4000)
    try: pg.wait_for_load_state("networkidle", timeout=20000)
    except Exception: pass

    target = popups[-1] if popups else pg
    if popups: print(f"팝업 {len(popups)}개 발생", flush=True)
    print(">>> URL:", target.url, flush=True)
    io.open(os.path.join(OUT, "req_form.html"), "w", encoding="utf-8").write(target.content())

    print("\n--- 제목 ---", flush=True)
    for h in target.query_selector_all("h1,h2,h3,legend,caption"):
        t=(h.inner_text() or '').strip()
        if t: print("  ", t[:70], flush=True)

    print("\n--- 입력 요소 ---", flush=True)
    for el in target.query_selector_all("input, select, textarea"):
        tag = el.evaluate("e=>e.tagName")
        typ = el.get_attribute("type")
        nm, idd = el.get_attribute("name"), el.get_attribute("id")
        val = str(el.get_attribute("value"))[:30]
        lbl = ""
        try:
            lbl = el.evaluate("""e=>{
                let l = e.id ? document.querySelector(`label[for="${e.id}"]`) : null;
                if(l) return l.innerText.trim();
                let p = e.closest('label'); if(p) return p.innerText.trim();
                let td = e.closest('td,li,div'); return td ? td.innerText.trim().slice(0,50) : '';
            }""")
        except Exception: pass
        print(f"  {tag}/{typ:<9} name={str(nm):<24} id={str(idd):<24} val={val:<18} | {str(lbl)[:46]}", flush=True)

    print("\n--- SELECT 옵션 ---", flush=True)
    for s in target.query_selector_all("select"):
        opts = [o.inner_text().strip() for o in s.query_selector_all("option")]
        print(f"  id={s.get_attribute('id')} name={s.get_attribute('name')}: {opts[:12]}", flush=True)

    print("\n--- 버튼 ---", flush=True)
    for b in target.query_selector_all("a,button,input[type=button],input[type=submit]"):
        t=(b.inner_text() or b.get_attribute("value") or "").strip()
        if t and len(t)<20 and any(k in t for k in ["신청","동의","확인","전체","선택"]):
            print(f"  [{t}] onclick={str(b.get_attribute('onclick'))[:70]}", flush=True)
    ctx.close()
