# -*- coding: utf-8 -*-
"""역배형이 이 주말 이변을 잡았을까 — 사후 재현. **실전 성적이 아니다.**

## 먼저 — 실시간에서는 애초에 못 돌린다

basemodel/config.py 의 `UPSET` 은 `"inherit": "BASIC"` 이다. 축 가중치를 기본형에서
그대로 물려받고 다른 것은 **시장 계수 m 하나뿐**이다("market": "fit-", 적합값 −0.27).
설명문도 그렇게 적혀 있다 — *"기본형이 좋게 보는데 시장은 아닌 말"*.

시장 계수는 확정배당을 먹는데, 배당은 발주 T−5 ~ +1분에만 존재한다.
**그러므로 주말 실시간에서 역배형 = 기본형이고, 이변을 잡을 장치가 없다.**

## 그래도 재는 것 — "배당을 알았다면"

경주가 끝난 지금은 확정배당이 있다. 그걸로 역배형의 결합식을 흉내 내 본다:

    결합 = z(우리 점수) + m × z(시장확률),   m = −0.27,  경주 안에서 표준화

시장확률이 높은 말(인기마)의 점수를 깎는다 — 그게 역배형이 하는 일이다.
**이 값을 실전 성적으로 읽으면 안 된다.** 발주 전에는 없는 정보다.

그리고 이변 4경주만 보면 안 된다. 인기마를 깎으면 **평범한 경주에서 잃는다.**
28경주 전체에서 같이 재야 정직하다.

    python tools/upset_posthoc.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from kra_client import KRA, KRAError        # noqa: E402

M = -0.27                                   # basemodel 역배형 적합값 (valid 에서 고름)
ODDS_NONE = 900.0
MEETNM = {1: "서울", 2: "제주", 3: "부산경남"}


def zin(v, g):
    s = pd.Series(v).groupby(g)
    return ((v - s.transform("mean")) / (s.transform("std") + 1e-9)).to_numpy(float)


def main():
    kra = KRA()
    rows = []
    for day in (20260912, 20260913):
        P = pd.read_csv(HERE / "out" / ("live_pred_%d.csv" % day), encoding="utf-8-sig")
        for meet in (1, 2, 3):
            try:
                R = pd.DataFrame(kra.fetch("race_result", rc_date=str(day), meet=meet))
            except KRAError:
                continue
            if R.empty:
                continue
            R["r"] = pd.to_numeric(R["rcNo"])
            R["o"] = pd.to_numeric(R["ord"], errors="coerce")
            R["od"] = pd.to_numeric(R["winOdds"], errors="coerce")
            R.loc[(R["od"] <= 0) | (R["od"] >= ODDS_NONE), "od"] = np.nan
            for rn, g in R.groupby("r"):
                if not (g["o"] == 1).any() or not g["od"].notna().any():
                    continue
                Q = P[(P["meet"] == meet) & (P["rcNo"] == rn)]
                if not len(Q):
                    continue
                tm = Q["tminus"].max()
                Q = Q[Q["tminus"] == tm]
                odds = dict(zip(g["hrName"].astype(str).str.strip(), g["od"]))
                Q = Q.assign(nm=Q["hrName"].astype(str).str.strip())
                Q = Q.assign(od=Q["nm"].map(odds))
                if Q["od"].isna().any():
                    continue
                inv = 1.0 / Q["od"].to_numpy(float)
                mkt = inv / inv.sum()                    # 시장확률
                key = np.full(len(Q), 1)
                base = zin(Q["score"].to_numpy(float), key)
                comb = base + M * zin(mkt, key)
                win = str(g[g["o"] == 1]["hrName"].iloc[0]).strip()
                nm = Q["nm"].to_numpy()
                rows.append(dict(
                    날짜="%02d/%02d" % (day // 100 % 100, day % 100), 장=MEETNM[meet], R=int(rn),
                    두수=len(Q), 일착=win, 배당=float(g[g["o"] == 1]["od"].iloc[0]),
                    기본픽=nm[np.argmax(base)], 역배픽=nm[np.argmax(comb)],
                    기본순위=int((-base).argsort().argsort()[list(nm).index(win)]) + 1,
                    역배순위=int((-comb).argsort().argsort()[list(nm).index(win)]) + 1))

    T = pd.DataFrame(rows)
    T["기본적중"] = (T["기본픽"] == T["일착"]).astype(int)
    T["역배적중"] = (T["역배픽"] == T["일착"]).astype(int)
    T["이변"] = T["배당"] >= 15

    print("=" * 84)
    print("역배형 사후 재현 — %d경주 · **실전 성적이 아니다** (배당은 발주 전에 없다)" % len(T))
    print("=" * 84)
    print("%-22s%10s%12s%12s" % ("", "1착 적중", "픽 평균배당", "이변 적중"))
    print("-" * 60)
    for c, lab in (("기본적중", "기본형 (= 실시간 역배형)"), ("역배적중", "역배형 m=-0.27 (사후)")):
        pick = "기본픽" if c == "기본적중" else "역배픽"
        od = T.apply(lambda r: r["배당"] if r[pick] == r["일착"] else np.nan, axis=1)
        print("%-22s%7d/%-3d%12s%9d/%d"
              % (lab, T[c].sum(), len(T),
                 "%.1f" % od.mean() if od.notna().any() else "-",
                 int(T[T["이변"]][c].sum()), int(T["이변"].sum())))
    print("-" * 60)
    ch = T[T["기본픽"] != T["역배픽"]]
    print("픽이 바뀐 경주 %d/%d" % (len(ch), len(T)))
    print("  바뀌어서 맞은 경주 %d · 바뀌어서 틀린 경주 %d"
          % (int(((ch["역배적중"] == 1) & (ch["기본적중"] == 0)).sum()),
             int(((ch["역배적중"] == 0) & (ch["기본적중"] == 1)).sum())))
    print("")
    print("이변 경주 %d개 — 역배형이 우승마를 위로 올렸나" % int(T["이변"].sum()))
    print("%-7s%-5s%-4s%-11s%8s%10s%10s" % ("날짜", "장", "R", "1착", "배당", "기본순위", "역배순위"))
    print("-" * 60)
    for r in T[T["이변"]].sort_values("배당", ascending=False).itertuples():
        arrow = "↑" if r.역배순위 < r.기본순위 else ("↓" if r.역배순위 > r.기본순위 else "=")
        print("%-7s%-5s%-4d%-11s%8.1f%10d%9d %s"
              % (r.날짜, r.장, r.R, r.일착, r.배당, r.기본순위, r.역배순위, arrow))
    d = (T["기본순위"] - T["역배순위"])
    print("")
    print("우승마 순위 변화 (양수 = 역배형이 위로 올림)")
    print("  전체 28경주  평균 %+.2f위 · 올림 %d · 내림 %d · 그대로 %d"
          % (d.mean(), int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())))
    print("  이변 4경주   평균 %+.2f위" % d[T["이변"]].mean())
    print("  평범 24경주  평균 %+.2f위" % d[~T["이변"]].mean())
    T.to_csv(HERE / "out" / "upset_posthoc.csv", index=False, encoding="utf-8-sig")
    print("")
    print("→ out/upset_posthoc.csv")


if __name__ == "__main__":
    main()
