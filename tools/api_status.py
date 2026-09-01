# -*- coding: utf-8 -*-
"""
접근 가능한 KRA OpenAPI 전수 점검.

resultCode=00 → 승인됨, 즉시 사용 가능
HTTP 403      → 엔드포인트는 살아있으나 활용신청 필요
HTTP 400/404  → 경로가 틀렸거나 폐기됨

실행: python tools/api_status.py
"""
import urllib.request, urllib.parse, ssl, json, sys, re, time

sys.stdout.reconfigure(encoding="utf-8")
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
KEY = [l.split("=", 1)[1].strip() for l in open(".env", encoding="utf-8")
       if l.startswith("KRA_API_KEY_ENCODED")][0]
BASE = "https://apis.data.go.kr/B551015"

# (라벨, 경로, 시험 파라미터, 용도)
APIS = [
    # ── 승인 완료 (원장·핵심) ─────────────────────────────────────────
    ("경주기록(원장)",      "API4_3/raceResult_3",        {"meet": 1, "rc_month": "202605"}, "학습 원장 89필드"),
    ("출전표",              "API26_2/entrySheet_2",       {"meet": 1, "rc_date": "20260830"}, "발주시각·두수"),
    ("경주계획표",          "API72_2/racePlan_2",         {"meet": 1, "rc_date": "20260830"}, "편성"),
    ("구간기록",            "API37_1/sectionRecord_1",    {"meet": 1, "rc_month": "202605"}, "구간 최고/평균"),
    ("확정배당율 통합",     "API160_1/integratedInfo_1",  {"meet": 1, "rc_date": "20260830"}, "★ 승식 7종 배당"),
    ("AI학습용 경주결과",   "API155/raceResult",          {"race_dt": "20260830"}, "보조"),
    ("경주마 성적",         "API15_2/raceHorseResult_2",  {"meet": 1}, "⚠ 누수"),
    ("경주마 레이팅",       "API77/raceHorseRating",      {}, "⚠ 누수"),
    ("조교사 통산전적",     "trtresult/gettrtresult",     {"meet": 1}, "⚠ 누수"),
    # ── 2026-08-31 추가 승인 ──────────────────────────────────────────
    ("말훈련내역(조교)",    "trcontihi/gettrcontihi",     {"tr_date_fr": "20260801", "tr_date_to": "20260810"}, "★ 조교"),
    ("경주마장제(편자)",    "API191_1/HorseShoe_1",       {"meet": 1, "shoe_date_fr": "20260701", "shoe_date_to": "20260731"}, "★ 편자"),
    ("마필종합(혈통)",      "API42_1/totalHorseInfo_1",   {"hr_no": "0047543"}, "★ 혈통 3대. 말당 1콜"),
    ("씨수말",              "API79_1/stallionInfo_1",     {}, "부마 마스터"),
    ("경주별상세성적표",    "racedetailresult/getracedetailresult", {"meet": 1}, "상세 성적"),
    ("경주마 상세정보",     "API8_2/raceHorseInfo_2",     {"meet": 1}, "★ 혈통(부·모) 벌크"),
    # ── 미신청 후보 — 문헌이 지목한 피처 ──────────────────────────────
    ("출발조교현황",        "API190/startTraining",       {}, "☆ K2 Shapley 상위"),
    ("출발심사결과",        "API189/startExam",           {}, "☆ K2 Shapley 상위"),
    ("말진료현황",          "API192/horseTreatment",      {}, "☆ 질병진단 K2 상위"),
    ("진료소방역정보",      "API63/clinicInfo",           {}, "☆ 질병"),
    ("주행심사결과",        "API188/runExam",             {}, "☆ 주행심사"),
    # ── 기타 미신청 ───────────────────────────────────────────────────
    ("경주마명단",          "racehorselist/getracehorselist", {}, "마스터"),
    ("현직기수정보",        "currentjockeyInfo/getcurrentjockeyinfo", {}, "마스터"),
    ("기수 상세정보",       "API12_1/jockeyInfo_1",       {}, "마스터"),
    ("복승식 확정배당율",   "API5/quinellaOddsInfo",      {"meet": 1}, "복승 배당"),
    ("승식별 최고배당률",   "API35/highestDividendRateInfo", {"meet": 1}, "참고"),
    ("당일 경주결과종합",   "API214_1/RaceDetailResult_1", {}, "백필 효율"),
    ("경마경주정보",        "API187/HorseRaceInfo",       {"ym_fr": "202601", "ym_to": "202608"}, "연간계획"),
]


def probe(path, params):
    q = {"serviceKey": KEY, "_type": "json", "numOfRows": "3", "pageNo": "1"}
    q.update({k: str(v) for k, v in params.items()})
    url = f"{BASE}/{path}?" + "&".join(
        f"{k}={v if k == 'serviceKey' else urllib.parse.quote(str(v))}" for k, v in q.items())
    try:
        raw = urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
            timeout=30, context=CTX).read().decode("utf-8", "replace")
    except Exception as e:
        code = getattr(e, "code", None)
        return ("신청필요" if code == 403 else f"HTTP {code or type(e).__name__}"), None, 0
    if raw.lstrip().startswith("{"):
        d = json.loads(raw)
        h = d.get("response", {}).get("header", {})
        b = d.get("response", {}).get("body", {}) or {}
        it = (b.get("items") or {})
        it = it.get("item", []) if isinstance(it, dict) else []
        it = [it] if isinstance(it, dict) else it
        rc = h.get("resultCode")
        return ("사용가능" if rc == "00" else f"code {rc}"), b.get("totalCount"), len(it[0]) if it else 0
    m = re.search(r"<returnAuthMsg>(.*?)</returnAuthMsg>|<errMsg>(.*?)</errMsg>", raw)
    msg = (m.group(1) or m.group(2)) if m else raw[:40]
    return ("신청필요" if "SERVICE_KEY" in str(msg) else f"XML {msg}"[:24]), None, 0


if __name__ == "__main__":
    ok, need, dead = [], [], []
    print(f"{'상태':<9} {'API':<22} {'total':>9} {'필드':>4}  용도")
    print("-" * 88)
    for label, path, params, note in APIS:
        st, tc, nf = probe(path, params)
        tcs = f"{tc:,}" if isinstance(tc, int) else "-"
        print(f"{st:<9} {label:<22} {tcs:>9} {nf:>4}  {note}")
        (ok if st == "사용가능" else need if st == "신청필요" else dead).append((label, path))
        time.sleep(0.15)
    print("-" * 88)
    print(f"사용가능 {len(ok)} / 신청필요 {len(need)} / 경로오류·폐기 {len(dead)}")
    if need:
        print("\n신청하면 바로 쓸 수 있는 것:")
        for l, p in need:
            print(f"  - {l}  ({p})")
    if dead:
        print("\n경로 확인 필요 (data.go.kr Swagger 로 재확인):")
        for l, p in dead:
            print(f"  - {l}  ({p})")
