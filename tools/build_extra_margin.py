# -*- coding: utf-8 -*-
"""H1 검정용 피처 구성 — 착차(diffUnit)를 마신으로 수치화해 as-of 집계한다.

가설(먼저 적고 시작한다):
  H1. 착순 백분위는 마신차를 버린다. 머리 차 2착과 10마신 차 2착이 지금 같은 값이다.
      착차를 수치화해 과거 성적에 붙이면 같은 착순 안에서 실력이 구분된다.
      → 지금 모델 gain 의 30.3% 를 차지하는 F1_ordpct_avg5_z 의 정보량을 늘리는 것.

★ 실측으로 확인한 의미 (2026-09-10):
  diffUnit 은 **우승마 누적이 아니라 앞말과의 간격**이고 단위는 마신이다.
  근거 — 부경 6R: 2착 "6"(우승마 대비 +1.0초) / 3착 "1¾"(+1.3초, 앞말과 0.3초)
  / 4착 "목"(+1.4초, 0.1초). 초당 5~6마신으로 일관. 누적으로 가정하면 조용히 오염된다.
  → 우승마 대비 마신차는 경주 내에서 **착순으로 정렬해 cumsum** 해야 한다.

규약 (레포 규칙 그대로):
  · 집계는 대상 경주보다 **엄격히 이전** 행만 쓴다 (emit → update 순서)
  · train 에서만 구성한다는 뜻은 "통계량을 valid 로 적합하지 않는다"는 것이다.
    as-of 누적은 시점 기반이라 분할과 무관하게 정의된다 — 미래를 안 보는 것이 핵심이다
  · 산출물은 row_id 키의 얇은 파케이. 기존 파케이를 덮어쓰지 않는다
  · 새 피처가 y_ord 와 |r|>0.55 면 누수 의심 (validate_v2 §6 과 같은 기준)

    python tools/build_extra_margin.py
"""
import io
import os
import re
import sys
from collections import defaultdict, deque

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

LEDGER = "data/raw/ledger_2010_2026.csv"
OUT = "dataset/v2/extra/margin.parquet"
ORD_MAX = 16
W365 = 365

# 전통 표기 → 마신. 코<머리<목 순서는 경마 관례.
WORD = {"-": 0.0, "동순위": 0.0, "코": 0.05, "머리": 0.10, "목": 0.25}
FRAC = {"½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 1 / 3, "⅔": 2 / 3,
        "⅛": 0.125, "⅜": 0.375, "⅝": 0.625, "⅞": 0.875}
UNPARSED = defaultdict(int)


def parse_margin(v):
    """'1¾' → 1.75 · '목' → 0.25 · '-' → 0.0 · 못 읽으면 NaN."""
    s = str(v).strip()
    if not s or s in ("nan", "None"):
        return np.nan
    if s in WORD:
        return WORD[s]
    # 숫자 + 분수 조합 (예: '1', '1¾', '¾')
    m = re.fullmatch(r"(\d*)\s*([½¼¾⅓⅔⅛⅜⅝⅞]?)", s)
    if m and (m.group(1) or m.group(2)):
        whole = float(m.group(1)) if m.group(1) else 0.0
        return whole + FRAC.get(m.group(2), 0.0)
    UNPARSED[s] += 1
    return np.nan


def main():
    print("원장 로드...", flush=True)
    cols = ["rcDate", "meet", "rcNo", "chulNo", "hrNo", "ord", "diffUnit", "rcTime", "dusu"]
    d = pd.read_csv(LEDGER, dtype=str, encoding="utf-8-sig",
                    usecols=lambda c: c in cols, low_memory=False)
    for c in ("rcDate", "rcNo", "chulNo", "ord", "rcTime"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d["ord"].between(1, ORD_MAX) & d["rcDate"].notna() & d["hrNo"].notna()].copy()
    # ★ race_id / row_id 는 **빌더와 글자 하나까지 같아야** 조인이 된다.
    #   build_v2 는 meet 를 MEET_CODE 로 숫자화한다("부산경남" -> 3). 원문 문자열을
    #   그대로 쓰면 교집합이 0 이 되어 조용히 전부 결측이 된다(실제로 한 번 그랬다).
    MEET_CODE = {"서울": 1, "제주": 2, "부산경남": 3, "1": 1, "2": 2, "3": 3}
    d["meet_s"] = d["meet"].astype(str).str.strip().map(MEET_CODE)
    d = d[d["meet_s"].notna()].copy()
    d["meet_s"] = d["meet_s"].astype(int).astype(str)
    d["race_id_src"] = (d["rcDate"].astype(int).astype(str) + "_" + d["meet_s"]
                        + "_" + d["rcNo"].astype("Int64").astype(str))
    print("  유효 완주 %d행" % len(d))

    # ── 착차 파싱 → 우승마 대비 누적 마신 ────────────────────────────
    d["gap"] = d["diffUnit"].map(parse_margin)
    if UNPARSED:
        print("  ⚠ 못 읽은 표기 %d종: %s"
              % (len(UNPARSED), sorted(UNPARSED.items(), key=lambda x: -x[1])[:10]))
    d = d.sort_values(["rcDate", "meet_s", "rcNo", "ord"], kind="stable")
    # 1착의 gap 은 0. 그 뒤는 앞말 대비이므로 경주 내 누적합이 우승마 대비 마신차다.
    d["behind"] = d.groupby("race_id_src", sort=False)["gap"].cumsum()
    print("  착차 파싱 충전 %.1f%% · 누적 마신 충전 %.1f%%"
          % (d["gap"].notna().mean() * 100, d["behind"].notna().mean() * 100))

    # ── 센티널·극단 처리 ────────────────────────────────────────────
    # diffUnit 의 큰 정수는 코드값이 섞여 있다. 실측: `500`(34행)의 실제 기록차는
    # 우승마 대비 **32.6초 ≈ 185마신** 이라 500 이라는 숫자 자체가 코드다.
    # 분포도 99까지 연속이고 그 위는 101/105/120/150/200/300/500 으로 띄엄띄엄하다.
    # (배당 9999.9 와 같은 종류의 함정이다 — schema_v2.ODDS_NONE 참조)
    #
    # 값을 만들어 넣지 않고 **99.5분위에서 윈저화**한다. 평균을 쓰는 피처라 이상치
    # 하나가 전체를 끌고 가는 것을 막는 것이 목적이고, 상한은 하나의 숫자로 문서화된다.
    CAP = float(d["behind"].quantile(0.995))
    n_clip = int((d["behind"] > CAP).sum())
    d["behind"] = d["behind"].clip(upper=CAP)
    print("  윈저화 상한 %.2f마신 (99.5분위) · 잘린 행 %d (%.3f%%)"
          % (CAP, n_clip, n_clip / len(d) * 100))

    # ── 교차검증: 누적 마신 vs rcTime 차이 ───────────────────────────
    w = d[d["ord"] == 1][["race_id_src", "rcTime"]].rename(columns={"rcTime": "t_win"})
    chk = d.merge(w, on="race_id_src", how="left")
    chk = chk[chk["rcTime"].notna() & chk["t_win"].notna() & (chk["rcTime"] > 0)
              & chk["behind"].notna() & (chk["ord"] > 1)]
    dt = chk["rcTime"] - chk["t_win"]
    ok = dt.between(0.05, 10)
    r = float(np.corrcoef(chk.loc[ok, "behind"], dt[ok])[0, 1])
    lps = float((chk.loc[ok, "behind"] / dt[ok]).median())
    print("  교차검증 — 누적마신 vs 기록차 상관 r=%.4f · 중앙값 %.2f마신/초 (n=%d)"
          % (r, lps, int(ok.sum())))
    if r < 0.90:
        raise SystemExit("상관이 낮다 — 착차 해석이 틀렸다. 중단.")

    # ── as-of 집계 (emit → update) ───────────────────────────────────
    print("as-of 집계...", flush=True)
    d = d.sort_values(["rcDate", "meet_s", "rcNo", "chulNo"], kind="stable").reset_index(drop=True)
    ordi = [pd.Timestamp(str(int(x))).toordinal() for x in d["rcDate"]]
    n = len(d)
    hr = d["hrNo"].to_numpy()
    beh = d["behind"].to_numpy(float)
    dus = pd.to_numeric(d["dusu"], errors="coerce").to_numpy(float) if "dusu" in d else np.full(n, np.nan)
    orr = d["ord"].to_numpy(float)

    # H1 을 **직접** 검정하는 최소 2개만 만든다. 변형을 여러 개 만들어 제일 좋은 걸
    # 고르면 그게 선택 누수다 — 규약대로 한 번에 하나씩 간다.
    #   F1_behind_avg5   최근 5출전 평균 마신차 (우승마 대비)
    #   F1_behind_best5  최근 5출전 중 최소 = 최고 성적의 마신차
    hist = defaultdict(lambda: deque(maxlen=5))
    out = {c: np.full(n, np.nan) for c in ("F1_behind_avg5", "F1_behind_best5")}
    for i in range(n):
        h = [x for x in hist[hr[i]] if x == x]
        if h:
            out["F1_behind_avg5"][i] = float(np.mean(h))
            out["F1_behind_best5"][i] = float(np.min(h))
        # update — 이 경주 결과는 여기서 처음 들어간다 (미래 차단)
        hist[hr[i]].append(beh[i])
        if i and i % 100000 == 0:
            print("    %d/%d" % (i, n), flush=True)

    res = pd.DataFrame({"row_id": d["race_id_src"] + "_" + d["chulNo"].astype("Int64").astype(str)})
    for c, v in out.items():
        res[c] = v.astype(np.float32)

    print("")
    print("충전율 / 분포:")
    for c in [c for c in res.columns if c != "row_id"]:
        s = res[c]
        print("  %-20s 충전 %5.1f%%  중앙 %7.2f  최소 %6.2f  최대 %7.2f"
              % (c, s.notna().mean() * 100, s.median(), s.min(), s.max()))

    # ── 누수 점검 — 이번 경주 착순과의 상관 (validate_v2 §6 과 같은 기준) ──
    j = res.copy()
    j["_ord"] = d["ord"].to_numpy()
    print("")
    print("누수 점검 — 이번 경주 y_ord 와의 상관 (|r|>0.55 면 의심):")
    for c in [c for c in res.columns if c != "row_id"]:
        m = j[c].notna()
        rr = float(np.corrcoef(j.loc[m, c], j.loc[m, "_ord"])[0, 1]) if m.sum() > 1000 else np.nan
        flag = "  ★의심" if abs(rr) > 0.55 else ""
        print("  %-20s r=%+.4f%s" % (c, rr, flag))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    res.to_parquet(OUT, index=False)
    print("")
    print("저장 %s — %d행 × %d열" % (OUT, len(res), len(res.columns)))


if __name__ == "__main__":
    main()
