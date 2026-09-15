# -*- coding: utf-8 -*-
"""그룹 안 지표 비중을 AI 가 정해 주면 축 점수가 나아지나 — pae_feature_weight 의 쓸모.

## 무엇이 문제인가

현재 계약은 축 점수를 **그룹 안 지표 점수의 산술 평균**으로 만든다
(EntryFeatureScoreAxisProvider: "그룹 안 항목 점수의 산술 평균"). 그러면
    ABILITY = mean(rating 11.2%, ebv_prize 18.1%, inbreeding 8.2%, ...)
처럼 무작위 수준(9.4%)인 지표가 좋은 지표와 **같은 무게**로 들어간다.

스키마에는 `pae_feature_weight`(항목 가중치) 테이블이 이미 있는데 쓰는 코드가 없다.
MP-03 으로 빠진 건 *유저가 항목을 직접 조절하는 기능*이지, 항목 가중치 자체가 아니다.
→ **유저는 그룹만 만지고, 그룹 안 비중은 AI 가 측정해서 채운다.**

    최종 항목 가중치 = groupWeight[g] × innerWeight[f]     ( Σ_{f∈g} innerWeight = 1 )

## 비중을 어떻게 정하나 — 셋을 비교한다

    균등      현행. 1/n
    파워비례  train 에서 잰 단독 판별력에 비례
    적합      train 에서 축 점수의 logloss 를 최소화하도록 맞춘다 (비음수·합 1)

**전부 train 에서 정하고 valid 에서만 잰다.** valid 로 비중을 맞추면 그 숫자로 다시
valid 를 평가할 수 없다.

    python tools/fit_group_inner_weights.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_temperature import AXES, AXIS_FEATURES, pct_within_race, race_index, nll, C  # noqa: E402


def rank100(v, rid):
    return (pd.Series(v).groupby(rid.values).rank(pct=True) * 100).to_numpy(float)


def main():
    tr, va = C.load("train"), C.load("valid")
    # 비중 적합은 반복 최적화라 train 전체(38만행)면 못 끝난다. 경주를 무작위로
    # 표본해 쓴다 — 비중 6×n 개를 정하는 데 8천 경주면 충분하다 (적합 대상이 작다).
    _r = np.random.default_rng(20260914)
    _keep = set(_r.choice(tr["race_id"].unique(), size=8000, replace=False))
    tr = tr[tr["race_id"].isin(_keep)].copy()
    print("비중 적합용 train 표본 %s (%d경주)" % (tr.shape, tr["race_id"].nunique()))
    tr_win = tr["y_win"].to_numpy(bool)
    q_tr, q_va = race_index(tr), race_index(va)
    nq_tr, nq_va = q_tr.max() + 1, q_va.max() + 1
    w_tr, w_va = tr["y_win"].to_numpy(bool), va["y_win"].to_numpy(bool)

    # 지표별 방향·판별력 — train 에서만
    direction, power, cols = {}, {}, {}
    for ax in AXES:
        keep = []
        for f in AXIS_FEATURES[ax]:
            if f in tr.columns and pd.api.types.is_numeric_dtype(tr[f]):
                p = pct_within_race(tr, f)
                if np.nanstd(p) > 1e-6:
                    direction[f] = 1.0 if p[tr_win].mean() >= 50.0 else -1.0
                    power[f] = abs(p[tr_win].mean() - 50.0)
                    keep.append(f)
        cols[ax] = keep

    def mat(df, ax):
        return np.stack([(lambda p: p if direction[f] > 0 else 100.0 - p)(pct_within_race(df, f))
                         for f in cols[ax]], 1)

    M_tr = {ax: mat(tr, ax) for ax in AXES}
    M_va = {ax: mat(va, ax) for ax in AXES}

    schemes = {}
    schemes["균등 (현행)"] = {ax: np.ones(len(cols[ax])) / len(cols[ax]) for ax in AXES}
    schemes["파워 비례"] = {ax: (lambda v: v / v.sum())(
        np.array([power[f] for f in cols[ax]])) for ax in AXES}

    # 적합 — 축마다 그 축 점수 하나만으로 train logloss 최소화. 비음수·합1
    #
    # ⚠ Nelder-Mead 를 0 벡터에서 시작하면 안 된다. scipy 의 기본 초기 심플렉스는
    #   0 인 성분을 0.00025 로만 밀어서 사실상 제자리에 머문다 (첫 시도에서 적합
    #   비중이 균등값과 소수 둘째 자리까지 같게 나왔다 — 최적화가 아니라 버그였다).
    #   파워비례의 로그에서 출발하고 초기 심플렉스를 직접 준다.
    fitted = {}
    for ax in AXES:
        n = len(cols[ax])
        p0 = np.log(np.array([power[f] for f in cols[ax]]) + 1e-6)
        p0 -= p0.mean()

        def obj(u, ax=ax):
            w = np.exp(u - u.max()); w /= w.sum()
            return nll(1.0, rank100(M_tr[ax] @ w, tr["race_id"]) * 0.04, q_tr, w_tr, nq_tr)

        simplex = np.vstack([p0] + [p0 + 0.7 * np.eye(n)[i] for i in range(n)])
        r = optimize.minimize(obj, p0, method="Nelder-Mead",
                              options=dict(initial_simplex=simplex, maxiter=300 * n,
                                           fatol=1e-6, xatol=1e-3))
        w = np.exp(r.x - r.x.max()); w /= w.sum()
        fitted[ax] = w
        print("  %-12s 적합 %4d회 · train nll %.5f → %.5f · 최대비중 %.3f (균등 %.3f)"
              % (ax, r.nfev, obj(p0), r.fun, w.max(), 1 / n))
    schemes["적합 (train)"] = fitted

    # PRIMARY 만 — `_z`/`_rk` 중복을 빼면 그 지표에 3배 무게가 붙는 일이 사라진다
    prim_idx = {ax: [i for i, f in enumerate(cols[ax]) if not f.endswith(("_z", "_rk"))]
                for ax in AXES}
    schemes["PRIMARY만 균등"] = {ax: (lambda v: v / v.sum())(
        np.isin(np.arange(len(cols[ax])), prim_idx[ax]).astype(float)) for ax in AXES}

    rng = np.random.default_rng(20260914)
    W = []
    while len(W) < 100:
        g = rng.dirichlet(np.ones(6)) * 100
        if g.max() <= 40:
            W.append(g)
    W = np.array(W)
    eq = np.ones(6) * 100 / 6

    print("=" * 86)
    print("그룹 안 지표 비중 3종 — valid 1,254경주 (비중은 train 에서 정했다)")
    print("=" * 86)
    print("%-16s%12s%12s%14s%12s" % ("비중", "top-1(균등축)", "top-3", "logloss(T*)", "T*"))
    print("-" * 86)
    for name, sch in schemes.items():
        AX = np.stack([rank100(M_va[ax] @ sch[ax], va["race_id"]) for ax in AXES], 1)
        raws = [AX @ g / 100.0 for g in W]
        def total(t):
            return float(np.mean([nll(t, r, q_va, w_va, nq_va) for r in raws]))
        T = float(optimize.minimize_scalar(total, bounds=(1e-4, 5.0), method="bounded",
                                           options=dict(xatol=1e-7)).x)
        m = C.evaluate(va, AX @ eq / 100.0)
        print("%-16s%11.1f%%%11.1f%%%14.4f%12.4f"
              % (name, m["top1"], m["top3"], total(T), T))
    print("-" * 86)
    print("")
    print("적합된 비중 — 그룹 안에서 어느 지표에 무게가 갔나 (상위 3개만)")
    for ax in AXES:
        w = fitted[ax]
        o = np.argsort(-w)[:3]
        print("  %-12s " % ax + " · ".join("%s %.2f" % (cols[ax][i].split("_", 1)[1], w[i])
                                           for i in o)
              + "   (균등이면 각 %.2f)" % (1 / len(cols[ax])))


if __name__ == "__main__":
    main()
