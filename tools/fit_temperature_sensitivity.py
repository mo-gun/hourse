# -*- coding: utf-8 -*-
"""T 는 점수를 **어떻게 만드느냐**에 딸린 값이다 — 환산 방식 3종의 T 를 나란히 잰다.

schema.sql:307 은 entry_feature_score.score 를 "0~100. 경주 내 순위 기반 환산" 이라고만
적어 놓았다. 이 한 줄이 최소 세 가지로 읽힌다:

    ① 지표별 순위 환산 후 그룹 평균   — 평균이 극값을 깎아 축 점수 폭이 좁다 (std 13~23)
    ② 그룹 평균 후 다시 순위 환산     — 축 점수가 경주 내 0~100 균등 (std ≈ 29)
    ③ 지표별 순위 환산 후 그룹 평균을 다시 0~100 선형 재척도

폭이 좁으면 T 는 커야 하고, 넓으면 작아야 한다. **T 를 정하려면 ①②③ 중 하나를
먼저 못박아야 한다.** 숫자만 합의하면 셋이 또 어긋난다.

    python tools/fit_temperature_sensitivity.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_temperature import (AXES, PRESETS, SPEC_T, IMPL_T, AXIS_FEATURES,   # noqa: E402
                             pct_within_race, race_index, nll, C)

MAXW, TOTAL, N = 40.0, 100.0, 200


def rank100(v, rid):
    return (pd.Series(v).groupby(rid.values).rank(pct=True) * 100).to_numpy(float)


def main():
    tr, va = C.load("train"), C.load("valid")
    q = race_index(va); nq = q.max() + 1
    win = va["y_win"].to_numpy(bool)
    tr_win = tr["y_win"].to_numpy(bool)
    rid = va["race_id"]

    direction = {}
    for ax in AXES:
        for f in AXIS_FEATURES[ax]:
            if f in tr.columns and pd.api.types.is_numeric_dtype(tr[f]):
                p = pct_within_race(tr, f)
                if np.nanstd(p) > 1e-6:
                    direction[f] = 1.0 if p[tr_win].mean() >= 50.0 else -1.0

    mean_ax = np.stack([np.stack([pct_within_race(va, f) if direction[f] > 0
                                  else 100.0 - pct_within_race(va, f)
                                  for f in AXIS_FEATURES[ax] if f in direction], 1).mean(1)
                        for ax in AXES], 1)

    variants = {
        "① 지표순위 → 그룹평균": mean_ax,
        "② 그룹평균 → 다시 순위": np.stack([rank100(mean_ax[:, i], rid) for i in range(6)], 1),
        "③ 그룹평균 → 0~100 재척도": np.stack(
            [(lambda s: (s - s.groupby(rid.values).transform("min"))
              / (s.groupby(rid.values).transform("max")
                 - s.groupby(rid.values).transform("min") + 1e-9) * 100)(
                pd.Series(mean_ax[:, i])).to_numpy(float) for i in range(6)], 1),
    }

    rng = np.random.default_rng(20260914)
    W = []
    while len(W) < N:
        w = rng.dirichlet(np.ones(6)) * TOTAL
        if w.max() <= MAXW:
            W.append(w)
    W = np.array(W)

    print("=" * 88)
    print("점수 환산 방식이 T 를 얼마나 바꾸나 — 도달 가능한 가중치 %d개 · valid 1,254경주" % N)
    print("=" * 88)
    print("%-24s%12s%10s%12s%12s%12s"
          % ("환산 방식", "축점수 표준편차", "공통 T", "손해(평균)", "0.0445 손해", "0.1 손해"))
    print("-" * 88)

    for name, AX in variants.items():
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
        print("%-24s%12.1f%10.4f%+12.4f%+12.4f%+12.4f"
              % (name, AX.std(0).mean(), Tg, d(Tg), d(SPEC_T), d(IMPL_T)))

    print("-" * 88)
    print("")
    print("읽는 법 — '손해' 는 그 가중치의 최적 T 대비 logloss 초과분이다. 0 에 가까울수록 맞다.")
    print("축 점수 폭(표준편차)이 넓어질수록 T 는 반비례로 작아진다. 그래서 **환산 방식을")
    print("못박기 전에는 T 숫자만 합의해도 소용이 없다.**")


if __name__ == "__main__":
    main()
