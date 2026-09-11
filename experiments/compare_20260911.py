# -*- coding: utf-8 -*-
"""오늘 경주 3갈래 비교 — 실시간 / 배당을 알았다면 / 시장.

    A) 75피처 실시간   T-12분에 실제로 봉인한 예측 (live_pred_20260911.csv)
    B) 79피처 사후     확정배당으로 F6 를 만들어 다시 예측 — **실전이 아니다.**
                       "배당을 알았다면 어떻게 예측했을까" 를 재는 것이다
    C) 시장            인기 1위마 (확정배당 최저)

B 를 실전 성적으로 읽으면 안 된다. 확정배당은 발주 -5~+1분에 들어오므로(오늘 완주
경주 전부 실측) T-12분에는 존재하지 않는다. B 의 값은 **배당 접근의 가치가 오늘
표본에서 얼마였는지**를 재는 데 있다.

세 지표를 같이 본다 — 뜻이 다르다:
    1착 적중    1픽이 그대로 우승
    연승권      1픽이 3착 이내
    3픽중 1착   우리 1·2·3픽 중에 우승마가 있음

    python experiments/compare_20260911.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(r"c:/Users/SSAFY/Desktop/주제선정")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "experiments"))
from kra_client import KRA                                       # noqa: E402
from live_predict_20260911 import (TARGET, LOG, prepared_frame, feature_sets,  # noqa: E402
                                   load_models, fetch_live, apply_market,
                                   LIVE_COLS, sched)

MEETNM = {1: "서울", 2: "제주", 3: "부산경남"}
ODDS_NONE = 900.0


def main():
    kra = KRA()
    live = pd.read_csv(LOG, encoding="utf-8-sig")
    _, frame = prepared_frame()
    meta79, models79 = load_models("79")

    plans = {}
    for meet in (1, 2, 3):
        try:
            plans[meet] = sched(kra, meet)
        except Exception:
            pass

    rows = []
    for meet in sorted(plans):
        res = pd.DataFrame(kra.fetch("race_result", rc_date=str(TARGET), meet=meet))
        if res.empty:
            continue
        res["r"] = pd.to_numeric(res["rcNo"])
        res["c"] = pd.to_numeric(res["chulNo"])
        res["o"] = pd.to_numeric(res["ord"], errors="coerce")
        res["od"] = pd.to_numeric(res["winOdds"], errors="coerce")
        res.loc[(res["od"] <= 0) | (res["od"] >= ODDS_NONE), "od"] = np.nan

        for rn, g in res.groupby("r"):
            fin = g[g["o"].between(1, 16)]
            if len(fin) < 1 or not g["od"].notna().any():
                continue                                  # 아직 안 끝났다
            win = fin[fin["o"] == 1]
            if not len(win):
                continue
            wn = str(win["hrName"].iloc[0]).strip()
            plc = set(fin[fin["o"] <= 3]["hrName"].astype(str).str.strip())

            # A) 실시간 75피처 — 실제 봉인본
            P = live[(live["meet"] == meet) & (live["rcNo"] == rn)].sort_values("rank")
            if not len(P):
                continue
            a1 = str(P.iloc[0]["hrName"]).strip()
            a3 = set(P.head(3)["hrName"].astype(str).str.strip())

            # B) 79피처 사후 — 확정배당으로 F6 를 만들어 다시 예측
            fr = frame[(pd.to_numeric(frame["meet"]) == meet)
                       & (pd.to_numeric(frame["rcNo"]) == rn)].copy()
            b1, b3 = "", set()
            if len(fr):
                lv = fetch_live(kra, meet, rn)
                for i, row in fr.iterrows():
                    v = lv.get(int(row["X_chulNo"]))
                    if v:
                        for c in LIVE_COLS:
                            if v[c] == v[c]:
                                fr.at[i, c] = v[c]
                odds = dict(zip(g["c"].astype(int), g["od"]))
                odds = {k: v for k, v in odds.items() if v == v}
                if odds:
                    fr = apply_market(fr, odds)
                    x = fr.copy()
                    for c, levels in meta79["levels"].items():
                        x[c] = pd.Categorical(x[c].astype(str), categories=levels).codes
                    sc = np.mean([m.predict(x[meta79["feats"]].to_numpy(float))
                                  for m in models79], axis=0)
                    fr = fr.assign(_s=sc).sort_values("_s", ascending=False)
                    b1 = str(fr.iloc[0]["hrName"]).strip()
                    b3 = set(fr.head(3)["hrName"].astype(str).str.strip())

            # C) 시장
            mk = g[g["od"].notna()].nsmallest(3, "od")
            c1 = str(mk["hrName"].iloc[0]).strip()
            c3 = set(mk["hrName"].astype(str).str.strip())

            hh = plans.get(meet, {}).get(rn, "")
            rows.append(dict(
                장=MEETNM[meet], R=int(rn), 발주=(hh[:2] + ":" + hh[2:]) if hh else "",
                두수=len(fin), 일착=wn,
                배당=(f"{win['od'].iloc[0]:.1f}" if pd.notna(win["od"].iloc[0]) else "-"),
                A픽=a1, A_1착=int(a1 == wn), A_연승=int(a1 in plc), A_3픽=int(wn in a3),
                B픽=b1, B_1착=int(b1 == wn), B_3픽=int(wn in b3),
                C픽=c1, C_1착=int(c1 == wn), C_3픽=int(wn in c3),
                AC=("같음" if a1 == c1 else "★갈림"),
                AB=("같음" if (b1 and a1 == b1) else ("★갈림" if b1 else "-"))))

    if not rows:
        print("아직 채점할 경주가 없다.")
        return
    T = pd.DataFrame(rows).sort_values("발주")
    show = T[["장", "R", "발주", "두수", "일착", "배당", "A픽", "A_1착",
              "B픽", "B_1착", "C픽", "C_1착", "AC", "AB"]]
    print(show.to_string(index=False))

    n = len(T)
    print("")
    print("=" * 70)
    print("채점 %d경주 (평균 %.1f두)" % (n, T["두수"].mean()))
    print("=" * 70)
    print("%-30s%10s%10s%10s" % ("", "1착 적중", "연승권", "3픽중1착"))
    print("-" * 62)
    print("%-30s%9.1f%%%9.1f%%%9.1f%%"
          % ("A) 75피처 실시간 (T-12분)", T["A_1착"].mean() * 100,
             T["A_연승"].mean() * 100, T["A_3픽"].mean() * 100))
    if T["B픽"].astype(bool).any():
        m = T["B픽"].astype(bool)
        print("%-30s%9.1f%%%9s%9.1f%%"
              % ("B) 79피처 사후 (배당 포함)", T.loc[m, "B_1착"].mean() * 100, "—",
                 T.loc[m, "B_3픽"].mean() * 100))
    print("%-30s%9.1f%%%9s%9.1f%%"
          % ("C) 시장 (인기 1위마)", T["C_1착"].mean() * 100, "—", T["C_3픽"].mean() * 100))
    se = np.sqrt(0.34 * 0.66 / n) * 100
    print("-" * 62)
    print("표준오차 약 ±%.1f%%p — 이 표본으로 우열을 판정하지 말 것" % se)
    print("")
    print("A vs C 갈린 경주 %d/%d" % ((T["AC"] == "★갈림").sum(), n))
    for r in T[T["AC"] == "★갈림"].itertuples():
        who = "우리 적중" if r.A_1착 else ("시장 적중" if r.C_1착 else "둘 다 틀림")
        print("   %s %dR  우리 %s / 시장 %s → 1착 %s · %s"
              % (r.장, r.R, r.A픽, r.C픽, r.일착, who))
    print("A vs B 갈린 경주 %d/%d  (배당을 알았다면 픽이 바뀌는 경주)"
          % ((T["AB"] == "★갈림").sum(), n))
    for r in T[T["AB"] == "★갈림"].itertuples():
        print("   %s %dR  실시간 %s / 배당포함 %s → 1착 %s"
              % (r.장, r.R, r.A픽, r.B픽, r.일착))
    out = HERE / "experiments" / "out" / ("compare_%d.csv" % TARGET)
    T.to_csv(out, index=False, encoding="utf-8-sig")
    print("")
    print("→ %s" % out.name)


if __name__ == "__main__":
    main()
