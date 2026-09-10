# -*- coding: utf-8 -*-
"""채점 — 적중률은 팀 채점기 규칙, 유의성은 쌍체.

출처: 지표·검정 방법은 `basemodel/basemodel/evaluate.py` · `significance.py` 의 것을
같은 규칙으로 다시 구현했다. 여기서 지키는 관례 두 개(둘 다 basemodel/ledger.md 에 근거):

  · **표에 적는 적중률은 팀 채점기 그대로**(`common.evaluate`, 동률이면 첫 행).
    레포의 다른 표와 직접 비교하려면 같은 자여야 한다.
  · **유의성 검정은 동률 1/k 기대값**으로 한다. 인기 1위가 2두인 경주에서 idxmax 는
    첫 행을 집어 시장 기준선이 0.16%p 낮게 나온다(valid 16경주, 1.3%).

절대 적중률의 표준오차 ±0.85%p 는 경주 1,254개라는 표본에서 오는 한계라 시드를 더 돌려도
줄지 않는다. 두 모델은 **같은 경주**를 맞히므로 쌍으로 묶으면 공통 분산이 상쇄되고
검출력이 3배 예민해진다 — 그래서 판정은 쌍체로만 한다.
검출 한계: 짝 공식 기준 top-1 ±2.39%p (80% 검정력, α=.05).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .harness import C


def _race_index(df: pd.DataFrame) -> np.ndarray:
    return pd.factorize(df["race_id"], sort=False)[0]


def table_metrics(df: pd.DataFrame, scores: np.ndarray) -> dict:
    """팀 채점기 그대로 — 레포의 다른 표와 같은 자."""
    return C.evaluate(df, scores)


def hit_expected(df: pd.DataFrame, scores: np.ndarray, col: str = "y_win") -> np.ndarray:
    """경주별 적중 기대값. 최고점이 k두 동률이면 1/k. 검정용."""
    q = _race_index(df)
    s = np.asarray(scores, float)
    y = df[col].to_numpy(float)
    n = q.max() + 1
    best = np.full(n, -np.inf)
    np.maximum.at(best, q, s)
    top = s >= best[q] - 1e-12
    cnt = np.zeros(n); np.add.at(cnt, q, top.astype(float))
    won = np.zeros(n); np.add.at(won, q, (top & (y > 0)).astype(float))
    return won / np.maximum(cnt, 1)


def race_z(df: pd.DataFrame, s: np.ndarray) -> np.ndarray:
    """경주 내 z-score. 계열마다 점수 척도가 달라 비교 전에 맞춰야 한다."""
    q = _race_index(df)
    s = np.asarray(s, float)
    n = q.max() + 1
    cnt = np.bincount(q, minlength=n).astype(float)
    mean = np.bincount(q, weights=s, minlength=n) / cnt
    var = np.bincount(q, weights=(s - mean[q]) ** 2, minlength=n) / np.maximum(cnt, 1)
    return (s - mean[q]) / np.sqrt(np.maximum(var[q], 1e-12))


def race_softmax(df: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    """경주 안에서만 softmax — 확률 합이 경주마다 1 이 된다."""
    q = _race_index(df)
    s = np.asarray(scores, float)
    n = q.max() + 1
    mx = np.full(n, -np.inf); np.maximum.at(mx, q, s)
    e = np.exp(s - mx[q])
    tot = np.zeros(n); np.add.at(tot, q, e)
    return e / tot[q]


def logloss(df: pd.DataFrame, scores: np.ndarray) -> float:
    p = race_softmax(df, scores)
    w = df["y_win"].to_numpy(bool)
    return float(-np.log(np.clip(p[w], 1e-15, 1)).mean())


def ece(df: pd.DataFrame, scores: np.ndarray, bins: int = 10) -> float:
    p = race_softmax(df, scores)
    y = df["y_win"].to_numpy(float)
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(out)


def _pl_nll(T, z, q, win, n) -> float:
    eta = T * z
    mx = np.full(n, -np.inf); np.maximum.at(mx, q, eta)
    e = np.exp(eta - mx[q])
    den = np.zeros(n); np.add.at(den, q, e)
    return float(-(eta[win] - (np.log(den) + mx)[q[win]]).mean())


def logloss_calibrated(df: pd.DataFrame, scores: np.ndarray,
                       folds: int = 2, seed: int = 0):
    """온도 보정 후 logloss. 반환 (logloss, 평균 T).

    ★ 이게 없으면 이 열은 비교가 안 된다. 계열마다 점수 척도가 다른데 온도 1 의
      경주 내 softmax 는 그 척도를 그대로 확률로 읽는다. 확률을 그대로 내놓는 계열
      (lgb_binary)은 softmax 가 한 번 더 걸려 거의 균등이 되고, 회귀 계열은 출력
      범위가 좁아 같은 일이 생긴다 — **모델이 나쁜 게 아니라 척도가 다른 것**이다.
      같은 함정을 시장 확률에서 이미 겪었다(basemodel/ledger.md 실수 기록).

    T 는 valid 2-fold 교차적합으로 적합한다 — 한쪽에서 고르고 다른 쪽에서 잰다.
    top-1/top-3 은 순위만 보므로 이 보정과 무관하다(값이 안 바뀐다).
    """
    z = race_z(df, scores)
    q = _race_index(df)
    win = df["y_win"].to_numpy(bool)
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, size=q.max() + 1)[q]
    lls, Ts = [], []
    for k in range(folds):
        fm, am = fold != k, fold == k
        qf = pd.factorize(df.loc[fm, "race_id"], sort=False)[0]
        r = optimize.minimize_scalar(_pl_nll, bounds=(1e-3, 20.0), method="bounded",
                                     args=(z[fm], qf, win[fm], qf.max() + 1))
        Ts.append(float(r.x))
        qa = pd.factorize(df.loc[am, "race_id"], sort=False)[0]
        lls.append(_pl_nll(float(r.x), z[am], qa, win[am], qa.max() + 1))
    return float(np.mean(lls)), float(np.mean(Ts))


def paired(df: pd.DataFrame, s_a: np.ndarray, s_b: np.ndarray,
           col: str = "y_win", n_boot: int = 10000, seed: int = 0) -> dict:
    """B − A. 쌍체 부트스트랩 CI(경주 단위 재표집) + McNemar 정확검정."""
    a = hit_expected(df, s_a, col)
    b = hit_expected(df, s_b, col)
    d = b - a
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1) * 100
    lo, hi = np.percentile(bs, [2.5, 97.5])
    only_a = float(np.clip(a - b, 0, None).sum())
    only_b = float(np.clip(b - a, 0, None).sum())
    tot = only_a + only_b
    # 동률 1/k 를 쓰므로 불일치 수가 정수가 아닐 수 있다. binomtest 는 정수를 받으니
    # 반올림한다 — 근사가 걸리는 건 **점수에 동률이 있는 쪽뿐**이다. 연속 점수를 내는
    # 모델은 동률이 사실상 없고(정수), valid 에서 동률이 나는 건 시장(-mkt_rank) 16경주
    # (1.3%) 다. 즉 이 근사는 '시장 대비' 비교에만, 그것도 1경주 미만 규모로 영향한다.
    p = stats.binomtest(int(round(only_b)), int(round(tot)), 0.5).pvalue if tot >= 1 else 1.0
    return {"diff": d.mean() * 100, "lo": lo, "hi": hi,
            "only_a": only_a, "only_b": only_b, "p": p}


def market_scores(df: pd.DataFrame) -> np.ndarray:
    return C.market_scores(df)


def market_logit(df: pd.DataFrame) -> np.ndarray:
    """시장을 **점수**로 쓸 때는 log(내재확률)이다.

    확률을 그대로 넣으면 경주 내 softmax 가 한 번 더 걸려 거의 균등이 된다
    (그렇게 재면 시장 logloss 가 2.21 로 무작위 2.38 근처가 나온다 — 실수 기록 참조).
    """
    p = df["F6_mkt_prob"].to_numpy(float)
    return np.log(np.clip(p, 1e-9, None))


def fmt_row(name: str, df: pd.DataFrame, scores: np.ndarray, width: int = 30) -> str:
    m = table_metrics(df, scores)
    ll, T = logloss_calibrated(df, scores)
    return (f"{name:<{width}}{m['top1']:>7.2f}{m['top3']:>8.2f}"
            f"{logloss(df, scores):>10.4f}{ll:>10.4f}{T:>7.2f}")


def header(width: int = 30) -> str:
    """logloss  = 온도 1 (레포의 다른 표와 같은 조건 — 척도 차이가 섞인다)
       logloss* = 온도 보정 후 (valid 2-fold 교차적합 — 계열 간 비교는 이 열로)
       T        = 적합된 온도. 1 에서 멀면 그 계열의 점수 척도가 다른 것이다."""
    return (f"{'모델':<{width}}{'top-1':>7}{'top-3':>8}{'logloss':>10}{'logloss*':>10}{'T':>7}"
            + chr(10) + "-" * (width + 42))
