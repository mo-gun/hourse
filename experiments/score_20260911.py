# -*- coding: utf-8 -*-
"""2026-09-11 예측 채점 — 봉인한 예측을 실제 결과와 대조한다.

    python experiments/score_20260911.py                    # 최신 봉인 파일 자동 선택
    python experiments/score_20260911.py pred_20260911_...csv

봉인 파일은 발주 **전**에 만들어졌다(파일명에 시각이 박혀 있다). 여기서는 결과만 받아와
붙이고, 예측은 한 글자도 고치지 않는다.

시장 기준선은 **경주 후에야** 만들 수 있다 — 발주 전 배당이 공공 API 로 제공되지 않기
때문이다(레포 실측). 그래서 "시장 대비"는 사후 비교이고, 모델은 그 정보 없이 예측했다.

★ 표본 경고 — 16경주다. top-1 적중률의 표준오차가 약 ±12%p 다. 우연히 6/16(37.5%)이
  나올 수도 3/16(18.8%)이 나올 수도 있다. **이 숫자로 모델 우열을 판정하면 안 된다.**
  valid 1,254경주(±0.85%p)가 판정용이고, 이건 파이프라인이 실제로 도는지 보는 것이다.
"""
import glob
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(r"c:/Users/SSAFY/Desktop/주제선정")
sys.path.insert(0, str(HERE))
from kra_client import KRA   # noqa: E402

TARGET = 20260911
OUT = HERE / "experiments" / "out"
MODELS = [("rank_A71_train", "A) tier A 71피처 · train"),
          ("rank_A71_trval", "B) tier A 71피처 · train+valid"),
          ("rank_F73_train", "C) 73피처(날씨·주로 NaN) · train")]
ODDS_NONE = 900.0        # 9999.9 = "배당 없음" 특수값. 안 거르면 인기순위가 뒤집힌다


def load_pred(arg):
    if arg:
        p = Path(arg) if Path(arg).exists() else OUT / arg
    else:
        cands = sorted(glob.glob(str(OUT / ("pred_%d_*.csv" % TARGET))))
        if not cands:
            raise SystemExit("봉인 파일이 없다. predict_20260911.py 를 먼저 돌려라.")
        p = Path(cands[-1])
    print("봉인 파일: %s" % p.name)
    return pd.read_csv(p, encoding="utf-8-sig"), p


def fetch_results():
    kra = KRA()
    rows = []
    for meet in (1, 2, 3):
        rows += kra.fetch("race_result", rc_date=str(TARGET), meet=meet)
    if not rows:
        raise SystemExit("9/11 결과가 아직 API 에 없다. 발주 후 15분쯤 지나 다시 돌려라.")
    d = pd.DataFrame(rows)
    for c in ("rcNo", "chulNo", "ord", "winOdds", "plcOdds"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["winOdds"] = d["winOdds"].replace(0, np.nan)
    d.loc[d["winOdds"] >= ODDS_NONE, "winOdds"] = np.nan
    print("결과 %d행 · 완주 %d행" % (len(d), int(d["ord"].between(1, 16).sum())))
    return d


def main():
    P, path = load_pred(sys.argv[1] if len(sys.argv) > 1 else None)
    R = fetch_results()

    MEETCODE = {"서울": 1, "제주": 2, "부산경남": 3}
    R["meet_i"] = R["meet"].astype(str).str.strip().map(MEETCODE)
    key = ["meet_i", "rcNo", "chulNo"]
    R2 = R[key + ["ord", "winOdds", "hrName"]].rename(columns={"hrName": "hrName_r"})
    P["meet_i"] = pd.to_numeric(P["meet"])
    P = P.rename(columns={"X_chulNo": "chulNo"})
    M = P.merge(R2, on=key, how="left", validate="one_to_one")

    miss = int(M["ord"].isna().sum())
    print("예측 %d두 중 결과 붙은 행 %d · 미결(취소·미출주) %d" % (len(M), len(M) - miss, miss))
    # 마명 대조 — 키가 어긋나면 조용히 틀린 채점이 된다
    both = M[M["hrName_r"].notna()]
    bad = int((both["hrName"].astype(str).str.strip()
               != both["hrName_r"].astype(str).str.strip()).sum())
    print("마명 불일치 %d건 %s" % (bad, "★키 어긋남 — 채점 신뢰 불가" if bad else "(키 정상)"))

    M["win"] = (M["ord"] == 1).astype(float)
    M["plc"] = M["ord"].le(3).astype(float)
    M["mkt_rank"] = M.groupby("race_id")["winOdds"].rank(method="min")

    # 결과가 있는 경주만 채점한다 (경주 단위 — 일부 두수만 취소된 경주는 살린다)
    ok = M.groupby("race_id")["ord"].transform(lambda s: s.notna().any())
    M = M[ok].copy()
    n_race = M["race_id"].nunique()

    print("")
    print("=" * 74)
    print("채점 — %d경주" % n_race)
    print("=" * 74)
    print("%-34s%9s%9s%10s" % ("모델", "1착", "연승권", "3픽중1착"))
    print("-" * 62)

    def score(rank_col):
        g = M.sort_values(rank_col).groupby("race_id", sort=False)
        p1 = g.head(1)
        p3 = M[M[rank_col] <= 3]
        return (p1["win"].mean() * 100, p1["plc"].mean() * 100,
                p3.groupby("race_id")["win"].max().mean() * 100)

    rows = []
    for col, nm in MODELS:
        if col not in M.columns:
            continue
        w, p, t3 = score(col)
        rows.append((nm, w, p, t3))
        print("%-34s%8.1f%%%8.1f%%%9.1f%%" % (nm, w, p, t3))
    if M["mkt_rank"].notna().any():
        w, p, t3 = score("mkt_rank")
        print("%-34s%8.1f%%%8.1f%%%9.1f%%" % ("시장 (인기 1위마, 사후)", w, p, t3))
    se = np.sqrt(0.33 * 0.67 / max(n_race, 1)) * 100
    print("-" * 62)
    print("표준오차 약 ±%.1f%%p — 이 표로 우열을 판정하지 말 것 (판정은 valid 1,254경주)" % se)

    print("")
    print("=" * 74)
    print("경주별 — 주 모델(A) 1픽")
    print("=" * 74)
    for rid, g in M.groupby("race_id", sort=True):
        g = g.sort_values("rank_A71_train")
        t = g.iloc[0]
        winner = g[g["ord"] == 1]
        wn = winner["hrName"].iloc[0] if len(winner) else "?"
        mk = g[g["mkt_rank"] == 1]
        mkn = mk["hrName"].iloc[0] if len(mk) else "?"
        mark = "O" if t["win"] == 1 else ("△" if t["plc"] == 1 else "X")
        od = t["winOdds"]
        print("  %s %2dR %-9s 1픽 %-9s(%s배당 %s) | 1착 %-9s | 시장1픽 %-9s"
              % (t["meet_nm"], int(t["rcNo"]), mark, t["hrName"],
                 "" if pd.isna(od) else "", "-" if pd.isna(od) else ("%.1f" % od),
                 wn, mkn))

    stamp = path.stem.replace("pred_", "scored_")
    outp = OUT / (stamp + ".csv")
    M.to_csv(outp, index=False, encoding="utf-8-sig")
    print("")
    print("채점 결과 → %s" % outp.name)


if __name__ == "__main__":
    main()
