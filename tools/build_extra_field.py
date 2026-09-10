# -*- coding: utf-8 -*-
"""H2 검정용 피처 구성 — "누구를 상대로 낸 성적인가" (경쟁 강도 보정).

가설(먼저 적고 시작한다):
  H2. 같은 착순이라도 상대 수준이 다르다. `5,000점초과` 경주의 2착과 `350점이하`
      경주의 2착이 지금 같은 값이다. 과거 경주에서 **만난 상대의 수준**과
      **이긴 상대의 수준**을 붙이면 같은 착순이 구분된다.
      → Benter(1994) 5개 범주 중 3번 "과거 성적의 보정"의 경쟁강도 축.
        H1(착차)이 같은 범주의 "마신차" 축을 이미 건드렸다.

★ `prizeCond` 를 쓰지 않은 이유 (실측 2026-09-10):
  246개 이질 표기다 — `R0~0`·`R1~35`(레이팅 구간) / `350점이하`(점수) /
  `1,200만원이하`(금액) / `(1년)1000점이하`(기간 한정). 최소 4개 체계가 섞여 있어
  하나의 수치로 파싱하면 잡음이 크고 행정적 변동이 섞인다.
  단 전제 자체는 살아 있다 — 등급(X_grade)을 알고도 남는 prizeCond 엔트로피가
  1.568/3.889 = 40% 다. 등급이 상대 수준을 다 설명하지 못한다.
  그래서 텍스트 대신 **원장에서 상대의 실제 실력을 직접 집계**한다. X_rating 은
  충전 46%(2014년경 시작)라 전 구간을 못 덮어서 쓰지 않고, as-of 통산승률을 쓴다.

규약:
  · 2패스. 1패스에서 행별 as-of 통산승률(자기 과거만)을 만들고, 2패스에서 그것을
    경주 단위로 모아 "만난 상대 / 이긴 상대"를 계산한다. 두 패스 모두 emit → update 라
    대상 경주보다 **엄격히 이전** 정보만 들어간다
  · 산출물은 row_id 키의 얇은 파케이. 기존 파케이를 덮어쓰지 않는다
  · y_ord 와 |r|>0.55 면 누수 의심 (validate_v2 §6 기준)

    python tools/build_extra_field.py
"""
import os
import sys
from collections import defaultdict, deque

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# 원장 경로와 산출물 이름을 인자로 받는다 — 9/11 사전 예측용으로 "원장 + 출전표"
# 결합본에 대해서도 같은 코드로 만들어야 정의가 갈리지 않는다.
LEDGER = sys.argv[1] if len(sys.argv) > 1 else "data/raw/ledger_2010_2026.csv"
OUT = "dataset/v2/extra/field%s.parquet" % ("_" + sys.argv[2] if len(sys.argv) > 2 else "")
ORD_MAX = 16
MEET_CODE = {"서울": 1, "제주": 2, "부산경남": 3, "1": 1, "2": 2, "3": 3}
MIN_STARTS = 3       # 출주 3회 미만은 승률이 잡음이라 상대 수준 집계에서 뺀다


def main():
    print("원장 로드...", flush=True)
    use = ["rcDate", "meet", "rcNo", "chulNo", "hrNo", "ord"]
    d = pd.read_csv(LEDGER, dtype=str, encoding="utf-8-sig",
                    usecols=lambda c: c in use, low_memory=False)
    for c in ("rcDate", "rcNo", "chulNo", "ord"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d["ord"].between(1, ORD_MAX) & d["rcDate"].notna() & d["hrNo"].notna()].copy()
    d["meet_i"] = d["meet"].astype(str).str.strip().map(MEET_CODE)
    d = d[d["meet_i"].notna()].copy()
    d["race_id"] = (d["rcDate"].astype(int).astype(str) + "_"
                    + d["meet_i"].astype(int).astype(str) + "_"
                    + d["rcNo"].astype("Int64").astype(str))
    d = d.sort_values(["rcDate", "meet_i", "rcNo", "chulNo"], kind="stable").reset_index(drop=True)
    n = len(d)
    print("  유효 완주 %d행" % n)

    hr = d["hrNo"].to_numpy()
    orr = d["ord"].to_numpy(float)
    rid = d["race_id"].to_numpy()

    # ── 1패스: 행별 as-of 통산 (출주수, 승률) ────────────────────────
    print("1패스 — as-of 통산승률...", flush=True)
    life = defaultdict(lambda: [0, 0])
    starts = np.zeros(n)
    wr = np.full(n, np.nan)
    for i in range(n):
        h = life[hr[i]]
        starts[i] = h[0]
        if h[0] >= MIN_STARTS:
            wr[i] = h[1] / h[0]
        h[0] += 1
        h[1] += 1 if orr[i] == 1 else 0
    print("  as-of 승률 충전 %.1f%% (출주 %d회 이상)" % (np.isfinite(wr).mean() * 100, MIN_STARTS))

    # ── 2패스: 경주 단위로 "만난 상대 / 이긴 상대" 수준 ───────────────
    print("2패스 — 경주별 상대 수준...", flush=True)
    # 경주별 합/개수로 자기 제외 평균을 O(1) 로 만든다
    df = pd.DataFrame({"rid": rid, "wr": wr, "ord": orr})
    g = df.groupby("rid", sort=False)["wr"]
    s_all = g.transform("sum").to_numpy()
    c_all = g.transform("count").to_numpy()
    faced = np.where(c_all > 1, (s_all - np.nan_to_num(wr)) / np.maximum(c_all - np.isfinite(wr), 1), np.nan)
    faced = np.where((c_all - np.isfinite(wr)) > 0, faced, np.nan)

    # 이긴 상대(= 자기보다 착순이 뒤인 말)의 평균 수준. 경주별로 정렬해 누적으로 구한다
    beaten = np.full(n, np.nan)
    order = np.lexsort((orr, rid))
    i = 0
    while i < n:
        j = i
        while j < n and rid[order[j]] == rid[order[i]]:
            j += 1
        idx = order[i:j]                      # 이 경주, 착순 오름차순
        w = wr[idx]
        fin = np.isfinite(w)
        # 뒤쪽(자기보다 착순이 큰) 구간의 합/개수 = 전체 - 앞쪽 누적 - 자기
        tot_s, tot_c = np.nansum(w), int(fin.sum())
        cum_s = np.concatenate([[0.0], np.nancumsum(w)])
        cum_c = np.concatenate([[0], np.cumsum(fin)])
        for k in range(len(idx)):
            s = tot_s - cum_s[k + 1]
            c = tot_c - cum_c[k + 1]
            if c > 0:
                beaten[idx[k]] = s / c
        i = j
    print("  만난 상대 충전 %.1f%% · 이긴 상대 충전 %.1f%%"
          % (np.isfinite(faced).mean() * 100, np.isfinite(beaten).mean() * 100))

    # ── 3패스: 말별 최근 5출전 평균 (as-of) ──────────────────────────
    print("3패스 — 말별 최근 5출전 집계...", flush=True)
    hf = defaultdict(lambda: deque(maxlen=5))
    hb = defaultdict(lambda: deque(maxlen=5))
    out = {c: np.full(n, np.nan) for c in ("F1_faced_wr_avg5", "F1_beaten_wr_avg5")}
    for i in range(n):
        a = [x for x in hf[hr[i]] if x == x]
        b = [x for x in hb[hr[i]] if x == x]
        if a:
            out["F1_faced_wr_avg5"][i] = float(np.mean(a))
        if b:
            out["F1_beaten_wr_avg5"][i] = float(np.mean(b))
        hf[hr[i]].append(faced[i])            # update — 이 경주는 여기서 처음 들어간다
        hb[hr[i]].append(beaten[i])
        if i and i % 150000 == 0:
            print("    %d/%d" % (i, n), flush=True)

    res = pd.DataFrame({"row_id": d["race_id"] + "_" + d["chulNo"].astype("Int64").astype(str)})
    for c, v in out.items():
        res[c] = v.astype(np.float32)

    print("")
    print("충전율 / 분포:")
    for c in [c for c in res.columns if c != "row_id"]:
        s = res[c]
        print("  %-22s 충전 %5.1f%%  중앙 %.4f  최소 %.4f  최대 %.4f"
              % (c, s.notna().mean() * 100, s.median(), s.min(), s.max()))

    print("")
    print("누수 점검 — 이번 경주 y_ord 와의 상관 (|r|>0.55 면 의심):")
    for c in [c for c in res.columns if c != "row_id"]:
        m = res[c].notna()
        rr = float(np.corrcoef(res.loc[m, c], orr[m.to_numpy()])[0, 1])
        print("  %-22s r=%+.4f%s" % (c, rr, "  ★의심" if abs(rr) > 0.55 else ""))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    res.to_parquet(OUT, index=False)
    print("")
    print("저장 %s — %d행 × %d열" % (OUT, len(res), len(res.columns)))


if __name__ == "__main__":
    main()
