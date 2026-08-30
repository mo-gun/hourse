# -*- coding: utf-8 -*-
"""
data.go.kr 한국마사회 오픈API 활용신청 — 반자동.

역할 분담
  스크립트 : 브라우저 구동, 신청 폼 열기, 활용목적/상세내용/상세기능 체크 채우기
  사람     : 네이버 로그인, 자동등록방지(캡차) 입력, 최종 '신청' 버튼 클릭

캡차는 사람이 풀라고 있는 장치이므로 자동화하지 않는다.
세션 쿠키가 브라우저 종료 시 소멸하므로 로그인~신청 전체를 한 프로세스에서 끝낸다.

실행:  python tools/apply_kra.py
로그:  <scratchpad>/apply_log.txt  (실시간 append)
"""
import sys, os, io, time, json

sys.stdout.reconfigure(encoding='utf-8')
from playwright.sync_api import sync_playwright

SP = (r"C:\Users\SSAFY\AppData\Local\Temp\claude"
      r"\c--Users-SSAFY-Desktop-----\5200d35d-bce2-4650-a88b-15ceb3e9d1f1\scratchpad")
PROFILE = os.path.join(SP, "chrome-profile")
LOG = os.path.join(SP, "apply_log.txt")
DUMP = os.path.join(SP, "form_dump.txt")

# ── 신청 대상 8종 ────────────────────────────────────────────────────────
TARGETS = [
    ("15105087", "말훈련내역"),                    # A급 조교
    ("15063955", "경주마장제정보"),                # A급 편자
    ("15057985", "마필종합 상세정보"),             # A급 혈통
    ("15059488", "씨수말 정보"),                   # A급 혈통 보강
    ("15089492", "경주별상세성적표"),              # B급
    ("15119524", "경마시행당일_경주결과종합"),     # B급 백필
    ("15119558", "경마시행당일_확정배당율종합"),   # B급 백필
    ("15058115", "경주마 상세정보"),               # B급
]

PURPOSE = (
    "SSAFY(삼성 청년 SW 아카데미) 교육과정 팀 프로젝트입니다. "
    "한국마사회 경주 데이터로 경주 결과 예측 모델을 학습하고, 모델의 예측 정확도를 "
    "시장 배당률과 비교·시각화하는 웹 서비스를 개발합니다. "
    "교육 목적의 비영리 프로젝트이며 실제 베팅·환전 기능은 포함하지 않습니다."
)
# 활용목적 라디오에서 우선 선택할 키워드 (앞에서부터 매칭)
PURPOSE_PICK = ["웹 사이트 개발", "웹사이트 개발", "앱개발", "참고자료", "기타"]


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    io.open(LOG, "a", encoding="utf-8").write(msg + "\n")


def dump(pg, tag):
    """폼 구조를 파일로 기록 — 셀렉터가 틀렸을 때 원인 추적용."""
    L = [f"===== {tag} =====", f"URL: {pg.url}", "--- 입력 요소 ---"]
    for el in pg.query_selector_all("input, select, textarea"):
        try:
            if not el.is_visible():
                continue
            lbl = el.evaluate(
                "e=>{let l=e.id?document.querySelector('label[for=\"'+e.id+'\"]'):null;"
                "if(l)return l.innerText.trim();let p=e.closest('label');"
                "if(p)return p.innerText.trim();let td=e.closest('td,li');"
                "return td?td.innerText.trim().slice(0,60):''}") or ""
            L.append(f"  {el.evaluate('e=>e.tagName')}/{el.get_attribute('type') or ''} "
                     f"name={el.get_attribute('name')} id={el.get_attribute('id')} "
                     f"| {lbl[:60]}")
        except Exception:
            pass
    L.append("--- 버튼 ---")
    for b in pg.query_selector_all("a,button,input[type=button],input[type=submit]"):
        try:
            t = (b.inner_text() or b.get_attribute("value") or "").strip()
            if t and len(t) < 22 and b.is_visible():
                L.append(f"  [{t}] onclick={str(b.get_attribute('onclick'))[:70]}")
        except Exception:
            pass
    io.open(DUMP, "a", encoding="utf-8").write("\n".join(L) + "\n\n")


def open_request_form(pg, did):
    pg.goto(f"https://www.data.go.kr/data/{did}/openapi.do", timeout=60000)
    pg.wait_for_timeout(1800)
    pg.evaluate(f"fn_goOpenAPIRequestForm('{did}')")
    pg.wait_for_timeout(3500)
    try:
        pg.wait_for_load_state("networkidle", timeout=20000)
    except Exception:
        pass


def fill_form(pg):
    """활용목적 라디오 · 상세내용 textarea · 상세기능/동의 체크박스를 채운다."""
    done = []

    # 1) 활용목적 라디오
    for kw in PURPOSE_PICK:
        hit = False
        for r in pg.query_selector_all("input[type=radio]"):
            try:
                if not r.is_visible():
                    continue
                lbl = r.evaluate(
                    "e=>{let l=e.id?document.querySelector('label[for=\"'+e.id+'\"]'):null;"
                    "if(l)return l.innerText;let p=e.closest('label');"
                    "if(p)return p.innerText;return ''}") or ""
                if kw in lbl:
                    r.check()
                    done.append(f"활용목적={kw}")
                    hit = True
                    break
            except Exception:
                pass
        if hit:
            break

    # 2) 상세내용 textarea (가장 큰 것)
    tas = [t for t in pg.query_selector_all("textarea") if t.is_visible()]
    if tas:
        tas[0].fill(PURPOSE)
        done.append("상세내용 입력")

    # 3) 체크박스 — 상세기능정보 전체 + 라이선스 동의. 캡차와 무관.
    n = 0
    for c in pg.query_selector_all("input[type=checkbox]"):
        try:
            if c.is_visible() and not c.is_checked():
                c.check()
                n += 1
        except Exception:
            pass
    if n:
        done.append(f"체크박스 {n}개")

    return done


def submitted(pg):
    """사람이 제출을 끝냈는지 판정."""
    try:
        b = pg.inner_text("body", timeout=4000)
    except Exception:
        return False
    return any(k in b for k in ("신청이 완료", "승인되었습니다", "개발계정",
                                "신청이 정상적으로", "활용신청 상세"))


with sync_playwright() as p:
    ctx = p.chromium.launch_persistent_context(
        PROFILE, channel="chrome", headless=False,
        args=["--start-maximized"], no_viewport=True)
    pg = ctx.pages[0] if ctx.pages else ctx.new_page()

    pg.goto("https://www.data.go.kr/index.do", timeout=60000)
    log(">>> 네이버로 로그인해 주세요. (최대 10분 대기)")
    end = time.time() + 600
    ok = False
    while time.time() < end:
        try:
            if "로그아웃" in pg.inner_text("body", timeout=5000):
                ok = True
                break
        except Exception:
            pass
        time.sleep(3)
    if not ok:
        log("!!! 로그인 타임아웃 — 종료")
        ctx.close()
        sys.exit(2)
    log(">>> 로그인 확인.\n")

    results = []
    for i, (did, name) in enumerate(TARGETS, 1):
        log(f"[{i}/{len(TARGETS)}] {name} ({did})")
        try:
            open_request_form(pg, did)
            dump(pg, f"{did} {name}")
            if "openapi.do" in pg.url and "Request" not in pg.url:
                log("    ! 신청 폼으로 이동하지 못함 — 이미 신청했거나 로그인 만료일 수 있음")
                results.append({"id": did, "name": name, "status": "form_not_opened",
                                "url": pg.url})
                continue

            done = fill_form(pg)
            log("    채움:", ", ".join(done) if done else "(입력 요소 못 찾음)")
            log("    >>> 캡차 입력 + '신청' 버튼을 직접 눌러주세요. (최대 5분 대기)")

            end = time.time() + 300
            got = False
            while time.time() < end:
                if submitted(pg):
                    got = True
                    break
                time.sleep(3)
            status = "submitted" if got else "timeout_or_skipped"
            log(f"    → {status}\n")
            results.append({"id": did, "name": name, "status": status, "url": pg.url})
        except Exception as e:
            log(f"    오류: {type(e).__name__}: {e}"[:200], "\n")
            results.append({"id": did, "name": name, "status": "error",
                            "error": f"{type(e).__name__}: {e}"[:200]})

    io.open(os.path.join(SP, "apply_result.json"), "w", encoding="utf-8").write(
        json.dumps(results, ensure_ascii=False, indent=1))
    log("=== 완료 ===")
    for r in results:
        log(f"  {r['status']:22s} {r['name']}")
    log("\n브라우저 60초 후 종료")
    pg.wait_for_timeout(60000)
    ctx.close()
