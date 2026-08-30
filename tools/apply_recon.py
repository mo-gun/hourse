# -*- coding: utf-8 -*-
"""1단계: 브라우저 띄우고 로그인 대기 → 활용신청 폼 구조 덤프."""
import sys, os, time, io
sys.stdout.reconfigure(encoding='utf-8')
from playwright.sync_api import sync_playwright

PROFILE = r"C:\Users\SSAFY\AppData\Local\Temp\claude\c--Users-SSAFY-Desktop-----\5200d35d-bce2-4650-a88b-15ceb3e9d1f1\scratchpad\chrome-profile"
OUT     = r"C:\Users\SSAFY\AppData\Local\Temp\claude\c--Users-SSAFY-Desktop-----\5200d35d-bce2-4650-a88b-15ceb3e9d1f1\scratchpad"
os.makedirs(PROFILE, exist_ok=True)
FIRST_ID = "15105087"   # 말훈련내역

with sync_playwright() as p:
    ctx = p.chromium.launch_persistent_context(
        PROFILE, channel="chrome", headless=False,
        args=["--start-maximized"], no_viewport=True)
    pg = ctx.pages[0] if ctx.pages else ctx.new_page()
    pg.goto("https://www.data.go.kr/index.do", timeout=60000)
    print(">>> 브라우저 열림. 네이버로 로그인해 주세요. (최대 8분 대기)", flush=True)

    deadline = time.time() + 480
    logged = False
    while time.time() < deadline:
        try:
            body = pg.inner_text("body", timeout=5000)
            if "로그아웃" in body or "마이페이지" in body:
                logged = True; break
        except Exception:
            pass
        time.sleep(3)

    if not logged:
        print("!!! 로그인 감지 실패 — 창을 열어둔 채 종료합니다.", flush=True)
        ctx.close(); sys.exit(2)

    print(">>> 로그인 감지됨. 활용신청 폼 구조를 확인합니다.", flush=True)
    pg.goto(f"https://www.data.go.kr/data/{FIRST_ID}/openapi.do", timeout=60000)
    pg.wait_for_timeout(2500)

    # 활용신청 버튼 후보 전부 나열
    print("\n--- 버튼/링크 후보 ---", flush=True)
    for el in pg.query_selector_all("a, button, input[type=button], input[type=submit]"):
        try:
            t = (el.inner_text() or el.get_attribute("value") or "").strip()
        except Exception:
            continue
        if not t or len(t) > 24: continue
        if any(k in t for k in ["활용신청", "신청", "장바구니"]):
            print(f"  [{t}] tag={el.evaluate('e=>e.tagName')} href={el.get_attribute('href')} onclick={str(el.get_attribute('onclick'))[:90]}", flush=True)

    # 활용신청 클릭
    btn = pg.query_selector("a:has-text('활용신청'), button:has-text('활용신청')")
    if btn:
        btn.click()
        pg.wait_for_load_state("networkidle", timeout=60000)
        pg.wait_for_timeout(2000)
        print(f"\n>>> 이동한 URL: {pg.url}", flush=True)
        io.open(os.path.join(OUT, "apply_form.html"), "w", encoding="utf-8").write(pg.content())
        print(">>> apply_form.html 저장", flush=True)
        print("\n--- 폼 입력 요소 ---", flush=True)
        for el in pg.query_selector_all("input, select, textarea"):
            print(f"  {el.evaluate('e=>e.tagName')}/{el.get_attribute('type')} "
                  f"name={el.get_attribute('name')} id={el.get_attribute('id')} "
                  f"value={str(el.get_attribute('value'))[:40]}", flush=True)
    else:
        print("!!! 활용신청 버튼을 못 찾음", flush=True)
        io.open(os.path.join(OUT, "detail_page.html"), "w", encoding="utf-8").write(pg.content())

    print("\n>>> 브라우저는 열어둡니다.", flush=True)
    ctx.close()
