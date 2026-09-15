# -*- coding: utf-8 -*-
"""발매 중에 예상배당이 어디든 올라오는지 실측한다 — 논쟁 대신 계측.

## 왜 필요한가

2026-09-10(발매일 아님) 조사로는 "공개 웹에 예상배당이 없다"까지 나왔다:
  · 오픈 API 배당 5종 전부 '시행된 경주의 확정배당'. 내일 조회 0행
  · race.kra.co.kr .do 경로 73개 / m.kra.co.kr 65개에 배당 페이지 없음
  · m.kra.co.kr JS 번들 9개에 배당 관련 AJAX 엔드포인트 0개
  · scoretableDailyList.do 는 일자별 요약성적표 — 지난 개최일(9/6)은 승식 배당이
    채워지고 내일(9/11)은 비었다. 세 조회 모두 '예상' 0회

**하지만 이건 발매 시간이 아닌 때의 관측이다.** "발매가 열리면 채워진다"면
위 결과로는 구별이 안 된다. 그래서 경마 시행일 발매 중에 같은 곳들을 반복 조회해
**채워지는 순간이 있는지** 기록한다. 마체중 tier 를 T-61~65분으로 좁힌 것과 같은 방법이다.

## 무엇을 보나

  ① 오픈 API — 원장 winOdds/plcOdds, 확정배당 API(API160_1)
  ② m.kra.co.kr 일자별 요약성적표 (scoretableDailyList.do)
  ③ m.kra.co.kr 경주별 상세성적표 (scoretableAllScoreList.do)
각 관측에 그 경주의 T-minus 를 같이 남긴다. 발주 **전에** 숫자가 보이면 예상배당이고,
발주 후에만 보이면 확정배당이다 — T-minus 부호로 갈린다.

    python probe_odds_live.py                 # 오늘 편성 전 경주를 한 번 훑는다
    python probe_odds_live.py --watch         # 5분마다 반복 (발매 시간대에 켜 둘 것)
    python probe_odds_live.py --day 20260911

기록: snapshots/odds_live.jsonl (한 줄 = 한 관측)
"""
import argparse
import gzip
import io
import json
import os
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from kra_client import KRA, KRAError   # noqa: E402

LOG = "snapshots/odds_live.jsonl"
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36",
      "Accept-Language": "ko-KR", "Accept-Encoding": "gzip",
      "Referer": "https://m.kra.co.kr/race/seoul/main.do"}
MEETPATH = {1: "seoul", 2: "jeju", 3: "bukyeong"}
MEETNM = {1: "서울", 2: "제주", 3: "부산경남"}
PLACEHOLDER = {"", "-", "0", "0()", "0.0"}


def fetch(url, tries=2):
    for i in range(tries):
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                       timeout=25, context=CTX)
            raw = r.read()
            if r.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
            return raw.decode("utf-8", "replace")
        except Exception:
            time.sleep(1.0 * (i + 1))
    return ""


def odds_numbers(html):
    """승식 배당 격자에서 '②1.6' 같은 (마번, 배당) 쌍을 뽑는다.

    동그라미 숫자(①~⑯)가 마번이고 바로 뒤 소수가 배당이다. 격자가 비어 있으면 0쌍.
    """
    txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
    circ = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯"
    return re.findall(r"([" + circ + r"]+)\s*([0-9]+\.[0-9])", txt)


def probe_once(kra, day, plans, now=None):
    now = now or datetime.now()
    recs = []
    for meet, sc in plans.items():
        path = MEETPATH[meet]
        # 페이지는 경마장·날짜 단위라 경주마다 다시 받지 않는다
        daily = fetch("https://m.kra.co.kr/race/%s/scoretableDailyList.do?%s"
                      % (path, urllib.parse.urlencode(dict(meet=meet, rcDate=day))))
        d_pairs = odds_numbers(daily)
        for rn, hhmm in sorted(sc.items()):
            tm = None
            if len(hhmm) == 4 and hhmm.isdigit():
                post = datetime(day // 10000, day // 100 % 100, day % 100,
                                int(hhmm[:2]), int(hhmm[2:]))
                tm = int(round((post - now).total_seconds() / 60))
            # ① 오픈 API
            try:
                led = kra.fetch("race_result", rc_date=str(day), meet=meet, rc_no=rn)
            except KRAError:
                led = []
            n_win = sum(1 for r in led
                        if str(r.get("winOdds", "")).strip() not in PLACEHOLDER)
            try:
                api_odds = len(kra.fetch("odds", rc_date=str(day), meet=meet,
                                         rc_no=rn, pool="WIN"))
            except KRAError:
                api_odds = 0
            # ③ 경주별 상세성적표
            det = fetch("https://m.kra.co.kr/race/%s/scoretableAllScoreList.do?%s"
                        % (path, urllib.parse.urlencode(
                            dict(meet=meet, rcDate=day, rcNo=rn))))
            rec = dict(obs=now.strftime("%Y%m%d_%H%M%S"), day=day, meet=meet, rcNo=rn,
                       sch=hhmm, tminus=tm, n_horses=len(led),
                       api_ledger_winOdds=n_win, api_odds_rows=api_odds,
                       web_daily_pairs=len(d_pairs), web_detail_pairs=len(odds_numbers(det)),
                       web_detail_has_yesang=det.count("예상"),
                       web_daily_has_yesang=daily.count("예상"))
            recs.append(rec)
            # 판정은 **경주 단위로 확실한 신호**만 쓴다.
            # 웹 격자(83쌍 등)는 날짜 단위라, 1R 이 끝나면 아직 안 뛴 2R 조회에도
            # 숫자가 잡혀 오탐이 난다. API 원장/확정배당은 rc_no 단위라 안전하다.
            flag = ""
            if tm is not None and tm > 0 and (n_win or api_odds):
                flag = "  ★★ 발주 전에 배당이 보인다 — 예상배당 확보 가능!"
            elif tm is not None and tm > 0 and rec["web_detail_pairs"]:
                flag = "  (웹에 숫자 있으나 날짜 단위라 이 경주 것인지 불명 — jsonl 확인)"
            print("  %s %2dR 발주 %s T%+5s분 | API원장 %2d/%-2d · API배당 %3d행 "
                  "| 웹(날짜단위) 일자별 %3d쌍 · 경주별 %3d쌍%s"
                  % (MEETNM[meet], rn, hhmm, tm if tm is not None else "?",
                     n_win, len(led), api_odds, rec["web_daily_pairs"],
                     rec["web_detail_pairs"], flag))
    os.makedirs("snapshots", exist_ok=True)
    with io.open(LOG, "a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, default=int(datetime.now().strftime("%Y%m%d")))
    ap.add_argument("--watch", action="store_true",
                    help="5분마다 반복 (마지막 발주 +N분에 자동 종료)")
    ap.add_argument("--every", type=int, default=300)
    ap.add_argument("--stop-after", type=int, default=20,
                    help="마지막 경주 발주 후 몇 분까지 볼지. 기본 20분")
    ap.add_argument("--max-calls", type=int, default=2000,
                    help="API 호출 상한. 넘으면 멈춘다")
    a = ap.parse_args()

    kra = KRA()
    plans = {}
    for meet in (1, 2, 3):
        try:
            pl = kra.fetch("race_plan", meet=meet, rc_date=str(a.day))
        except KRAError:
            pl = []
        if pl:
            plans[meet] = {int(p["rcNo"]): str(p.get("schStTime") or "") for p in pl}
    if not plans:
        print("%d 에 편성된 경주가 없다. 경마 시행일에 돌려라." % a.day)
        return
    print("%d · %d경마장 %d경주" % (a.day, len(plans), sum(len(v) for v in plans.values())))
    print("판독 — 발주 **전**(T-minus 양수)에 배당 숫자가 보이면 그게 예상배당이다.")
    print("       발주 후에만 보이면 확정배당이고, 실시간 예측에는 쓸 수 없다.\n")

    # ── 종료 조건 ────────────────────────────────────────────────────
    # 2026-09-11 에 --watch 를 켜 두고 회수하지 않아 **사흘간** 돌면서 API 할당량을
    # 태웠다 (T-4,429분짜리 관측이 기록에 남았다). 그래서 --watch 에 끝을 박는다.
    #   ① 마지막 경주 발주 + stop-after 분이 지나면 멈춘다 — 그 뒤엔 볼 것이 없다
    #   ② 호출이 max-calls 를 넘으면 멈춘다 — 편성을 잘못 읽어도 폭주하지 않는다
    last = None
    for sc in plans.values():
        for hhmm in sc.values():
            if len(hhmm) == 4 and hhmm.isdigit():
                t = datetime(a.day // 10000, a.day // 100 % 100, a.day % 100,
                             int(hhmm[:2]), int(hhmm[2:]))
                last = t if last is None or t > last else last
    deadline = (last + timedelta(minutes=a.stop_after)) if last else None
    if a.watch and deadline is None:
        print("발주 시각을 못 읽었다 — 반복하지 않고 1회만 관측한다.")
        a.watch = False
    if a.watch:
        print("종료 — %s 까지 또는 %d콜"
              % (deadline.strftime("%m/%d %H:%M"), a.max_calls))

    while True:
        print("=== 관측 %s ===" % datetime.now().strftime("%H:%M:%S"), flush=True)
        probe_once(kra, a.day, plans)
        if not a.watch:
            break
        now = datetime.now()
        if now >= deadline:
            print("  마지막 발주 +%d분 경과 — 종료" % a.stop_after, flush=True)
            break
        if kra.calls >= a.max_calls:
            print("  호출 상한 %d 도달 — 종료" % a.max_calls, flush=True)
            break
        print("  (%d초 대기 · %d콜 · 종료까지 %d분)"
              % (a.every, kra.calls, (deadline - now).total_seconds() // 60), flush=True)
        time.sleep(a.every)
    print("")
    print("기록 -> %s  (총 %d콜)" % (LOG, kra.calls))


if __name__ == "__main__":
    main()
