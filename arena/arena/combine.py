# -*- coding: utf-8 -*-
"""결합 — 계열 앙상블 + Benter 2단계(펀더멘털 → 시장).

레포가 반복해서 같은 결론에 도달한 지점이다:
  · 정원님 `exp/tower-hist` §6-3 — "축 개선으로는 해결되지 않는다. **결합식 문제다**"
  · basemodel/ledger.md 판정 §3 — "**결합식이 전부다**. 축 구조가 아니라 시장 신호의 양"
  · 정원님 장부 — top-1 최고 기록이 단일 모델이 아니라 **LGB + S3 앙상블 35.3**

그래서 여기서는 계열을 늘린 뒤 **결합**을 제대로 재는 것까지 한 묶음으로 본다.

⚠ 결합 계수는 **valid 2-fold 교차적합**으로 적합한다. train 에서 적합하면 안 되는 이유:
  기저 모델이 train 을 학습했으므로 그 구간에서 펀더멘털 점수가 실제보다 날카롭고,
  결합 계수는 "그 날카로움과 시장의 교환비"라서 in-sample 에서 고르면 과하게 기운다.
  (basemodel 역배형 m 에서 실제로 겪었다 — train 적합 m=−0.50 이 valid 로 전이 안 됨)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from .evaluate import _race_index, race_softmax, race_z   # race_z 는 evaluate 가 원본


def ens_z(df: pd.DataFrame, scores: list[np.ndarray], w=None) -> np.ndarray:
    """경주 내 z-score 평균. 순위만 쓰는 지표(top-1/top-3)에 적합."""
    Z = np.stack([race_z(df, s) for s in scores])
    w = np.ones(len(Z)) if w is None else np.asarray(w, float)
    return (Z * (w / w.sum())[:, None]).sum(axis=0)


def ens_prob(df: pd.DataFrame, scores: list[np.ndarray], w=None) -> np.ndarray:
    """경주 내 softmax 확률 평균 → log. 확률 품질(logloss)까지 볼 때."""
    P = np.stack([race_softmax(df, s) for s in scores])
    w = np.ones(len(P)) if w is None else np.asarray(w, float)
    p = (P * (w / w.sum())[:, None]).sum(axis=0)
    return np.log(np.clip(p, 1e-15, None))


# ── 조건부 로짓 (Benter 1994) ─────────────────────────────────────────
def _nll(beta, X, q, win, n):
    eta = X @ beta
    mx = np.full(n, -np.inf); np.maximum.at(mx, q, eta)
    e = np.exp(eta - mx[q])
    den = np.zeros(n); np.add.at(den, q, e)
    lse = np.log(den) + mx
    return float(-(eta[win] - lse[q[win]]).sum())


def fit_conditional_logit(df: pd.DataFrame, X: np.ndarray) -> np.ndarray:
    """경주 단위 조건부 로짓 MLE. X = (행, 특성). 반환 계수.

    가드 — 1착이 없는 경주가 끼면 logsumexp 가 마스크값을 그대로 더해 우도비가 0 이 된다
    (basemodel/ledger.md 실수 기록). 여기서는 경주 단위로 걸러 그 상황을 만들지 않는다.
    """
    q = _race_index(df)
    win = df["y_win"].to_numpy(bool)
    n = q.max() + 1
    have = np.zeros(n, bool); have[q[win]] = True
    keep = have[q]
    if not keep.all():
        sub = df[keep]
        q = _race_index(sub); win = sub["y_win"].to_numpy(bool)
        X = X[keep]; n = q.max() + 1
    r = optimize.minimize(_nll, np.zeros(X.shape[1]), args=(X, q, win, n), method="BFGS")
    return r.x


def benter2(df: pd.DataFrame, base: np.ndarray, folds: int = 2, seed: int = 0):
    """2단계 결합 — eta = a·log(시장확률) + b·(경주내 z 표준화한 펀더멘털 점수).

    valid 를 경주 단위로 fold 로 갈라, 한쪽에서 (a,b) 적합 → 다른 쪽 점수 산출.
    반환 (결합점수, 적합계수 목록).
    """
    from .evaluate import market_logit
    mk = market_logit(df)
    z = race_z(df, base)
    q = _race_index(df)
    rng = np.random.default_rng(seed)
    fold_of_race = rng.integers(0, folds, size=q.max() + 1)
    fold = fold_of_race[q]
    out = np.zeros(len(df)); betas = []
    for k in range(folds):
        fit_m, app_m = fold != k, fold == k
        d_fit = df[fit_m]
        X_fit = np.column_stack([mk[fit_m], z[fit_m]])
        b = fit_conditional_logit(d_fit, X_fit)
        betas.append(b)
        out[app_m] = np.column_stack([mk[app_m], z[app_m]]) @ b
    return out, betas
