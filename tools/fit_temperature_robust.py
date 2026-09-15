# -*- coding: utf-8 -*-
"""환산 ②(축 점수를 경주 내 순위로) 를 쓰면 T 가 모델에 안 휘둘리는가.

앞선 측정에서 ② 일 때 T ≈ 0.041 이고 명세값 0.0445 의 손해가 +0.0032 로 거의 없었다.
그런데 그건 '내 방식으로 만든 축 점수' 하나에서 나온 값이다. 축 점수를 다른 방식으로
만들면 T 가 또 달라진다면, ② 를 못박아도 소용이 없다.

그래서 축 점수 생성기를 **일부러 다르게** 넷 만들어 T 를 각각 잰다:
    a) 지표 순위 평균          (원래 방식)
    b) 축마다 대표 지표 1개     — 정보량이 훨씬 적다
    c) 지표 순위의 중앙값       — 극값에 둔감
    d) 지표 z-score 평균        — 순위가 아니라 값 기반

넷 다 ② 를 거치면 T 가 붙어야 한다. 붙으면 "0.0445 를 그냥 두면 된다" 가 되고,
흩어지면 모델을 확정한 뒤에 다시 재야 한다.

    python tools/fit_temperature_robust.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_temperature import (AXES, AXIS_FEATURES, SPEC_T, IMPL_T,        # noqa: E402
                             pct_within_race, race_index, nll, C)

MAXW, TOTAL, N = 40.0, 100.0, 150


def rank100(v, rid):
    return (pd.Series(v).groupby(rid.values).rank(pct=True) * 100).to_numpy(float)


def main():
    tr, va = C.load("train"), C.load("valid")
    q = race_index(va); nq = q.max() + 1
    win = va["y_win"].to_numpy(bool); tr_win = tr["y_win"].to_numpy(bool)
    rid = va["race_id"]

    direction, power = {}, {}
    for ax in AXES:
        for f in AXIS_FEATURES[ax]:
            if f in tr.columns and pd.api.types.is_numeric_dtype(tr[f]):
                p = pct_within_race(tr, f)
                if np.nanstd(p) > 1e-6:
                    direction[f] = 1.0 if p[tr_win].mean() >= 50.0 else -1.0
                    power[f] = abs(p[tr_win].mean() - 50.0)      # train 에서만 본다

    def cols(ax):
        return [f for f in AXIS_FEATURES[ax] if f in direction]

    def signed_pct(f):
        p = pct_within_race(va, f)
        return p if direction[f] > 0 else 100.0 - p

    def zcol(f):
        v = pd.to_numeric(va[f], errors="coerce")
        g = v.groupby(rid)
        z = ((v - g.transform("mean")) / (g.transform("std") + 1e-6)).fillna(0).to_numpy(float)
        return z * direction[f]

    gens = {
        "a) 지표 순위 평균": lambda ax: np.stack([signed_pct(f) for f in cols(ax)], 1).mean(1),
        "b) 대표 지표 1개": lambda ax: signed_pct(max(cols(ax), key=lambda f: power[f])),
        "c) 지표 순위 중앙값": lambda ax: np.median(np.stack([signed_pct(f) for f in cols(ax)], 1), 1),
        "d) 지표 z-score 평균": lambda ax: np.stack([zcol(f) for f in cols(ax)], 1).mean(1),
    }

    rng = np.random.default_rng(20260914)
    W = []
    while len(W) < N:
        w = rng.dirichlet(np.ones(6)) * TOTAL
        if w.max() <= MAXW:
            W.append(w)
    W = np.array(W)

    print("=" * 90)
    print("축 점수 생성기를 바꿔도 ② 환산 뒤 T 가 버티나 — 가중치 %d개 · valid 1,254경주" % N)
    print("=" * 90)
    print("%-22s%10s%12s%14s%12s%12s"
          % ("축 점수 생성기", "top-1", "공통 T", "손해(T*)", "0.0445", "0.1"))
    print("-" * 90)

    for name, gen in gens.items():
        AX = np.stack([rank100(gen(ax), rid) for ax in AXES], 1)     # ② 를 거친다
        raws = [AX @ w / 100.0 for w in W]
        best = np.array([optimize.minimize_scalar(
            nll, bounds=(1e-4, 50.0), method="bounded", args=(r, q, win, nq),
            options=dict(xatol=1e-7)).fun for r in raws])

        def total(t):
            return float(np.mean([nll(t, r, q, win, nq) for r in raws]))

        Tg = float(optimize.minimize_scalar(total, bounds=(1e-4, 50.0), method="bounded",
                                            options=dict(xatol=1e-7)).x)
        d = lambda t: float(np.mean([nll(t, r, q, win, nq) - best[i]
                                     for i, r in enumerate(raws)]))
        eq = AX @ (np.ones(6) * 100 / 6) / 100.0
        print("%-22s%9.1f%%%12.4f%+14.4f%+12.4f%+12.4f"
              % (name, C.evaluate(va, eq)["top1"], Tg, d(Tg), d(SPEC_T), d(IMPL_T)))

    print("-" * 90)
    print("")
    print("축 점수 표준편차 — ② 를 거치면 여섯 축이 전부 같아진다 (슬라이더 1점의 무게가 같다)")
    AX = np.stack([rank100(gens["a) 지표 순위 평균"](ax), rid) for ax in AXES], 1)
    raw_ax = np.stack([gens["a) 지표 순위 평균"](ax) for ax in AXES], 1)
    print("    ① 환산 전  " + "  ".join("%s %.1f" % (a[:4], s)
                                      for a, s in zip(AXES, raw_ax.std(0))))
    print("    ② 환산 후  " + "  ".join("%s %.1f" % (a[:4], s)
                                      for a, s in zip(AXES, AX.std(0))))


if __name__ == "__main__":
    main()
