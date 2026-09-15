# -*- coding: utf-8 -*-
"""제품 계약의 온도 T 를 실측한다 — 명세 0.0445 인가, 백엔드 구현 0.1 인가.

## 왜 필요한가

같은 점수를 놓고 세 곳이 다른 확률을 낸다:
    docs/명세서 · BotConfigDefaults.TEMPERATURE      T = 0.0445
    back/…/room/service/PredictionService.java:57    exp(score / 10.0)  → T = 0.1
    front/src/room/rules.js                          softmax 자체가 없다

T 는 취향이 아니라 **측정값**이다. 검증셋에서 logloss 를 최소로 만드는 값이 하나 있다.

## 무엇을 재나

제품이 실제로 돌릴 경로를 그대로 만든다:
    ① 지표 점수  entry_feature_score.score = "0~100. 경주 내 순위 기반 환산" (schema.sql:307)
    ② 축 점수    그 그룹 지표들의 산술 평균 (EntryFeatureScoreAxisProvider 와 같은 방식)
    ③ 점수       raw = Σ( groupWeights[k] × ax_k ) / 100
    ④ 확률       p = softmax(raw × T), 경주 안에서

basemodel 의 신경망 축 점수는 경주 내 z-score(±2 부근)라 척도가 다르다. 0.0445 는
그 자로 잰 값일 수 없다 — export.py::fit_temperature 의 탐색 구간이 0.2~6.0 이다.
여기서는 **DB 에 실제로 들어갈 0~100 척도**로 잰다.

## 규약

    · 방향(높을수록 좋은가)은 **train** 에서 정한다. valid 에 적합시키지 않는다
    · T 는 valid 에서 맞춘다 — 비교·튜닝은 valid 에서 한다는 팀 규칙
    · test 는 열지 않는다. game 은 쓰지 않는다
    · 채점은 팀 채점기 docs/model/common.py 를 거친다
    · 정렬을 바꾸지 않는다

    python tools/fit_temperature.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

sys.stdout.reconfigure(encoding="utf-8")

TEAM = Path(r"c:/Users/SSAFY/Desktop/말고리즘/S15P21A304")
BASE = Path(r"c:/Users/SSAFY/Desktop/말고리즘/horse-pred-engine/basemodel")
sys.path.insert(0, str(TEAM / "docs" / "model"))
sys.path.insert(0, str(BASE))
import common as C                                            # noqa: E402
from basemodel.config import AXIS_FEATURES, MARKET            # noqa: E402

AXES = ["CONDITION", "SPEED", "RUNNING", "JOCKEY", "ENVIRONMENT", "ABILITY"]
SPEC_T = 0.0445          # docs/명세서 · BotConfigDefaults
IMPL_T = 0.1             # PredictionService exp(score / 10.0)

# 화면 프리셋 (front/src/mock/fixtures.js:113~117). 합 100
PRESETS = {
    "균등":     {k: 100 / 6 for k in AXES},
    "기본형":   dict(CONDITION=17, SPEED=17, RUNNING=12, JOCKEY=22, ENVIRONMENT=19, ABILITY=13),
    "사람형":   dict(CONDITION=15, SPEED=13, RUNNING=12, JOCKEY=35, ENVIRONMENT=15, ABILITY=10),
    "상승세형": dict(CONDITION=32, SPEED=16, RUNNING=12, JOCKEY=20, ENVIRONMENT=12, ABILITY=8),
    "한 축 몰기(기수 100)": dict(CONDITION=0, SPEED=0, RUNNING=0, JOCKEY=100, ENVIRONMENT=0, ABILITY=0),
}


def race_index(df):
    return pd.factorize(df["race_id"], sort=False)[0]


def pct_within_race(df, col):
    """경주 안에서 순위 백분위 0~100. 결측은 50(중립). schema.sql:307 '경주 내 순위 기반 환산'."""
    v = pd.to_numeric(df[col], errors="coerce")
    r = v.groupby(df["race_id"]).rank(method="average", pct=True)
    return (r * 100).fillna(50.0).to_numpy(float)


def nll(T, raw, q, win, n_races):
    """경주 내 softmax 의 승자 음의 로그가능도 (Plackett-Luce 1단계)."""
    z = raw * T
    m = np.zeros(n_races)
    np.maximum.at(m, q, z)
    e = np.exp(z - m[q])
    s = np.zeros(n_races)
    np.add.at(s, q, e)
    lp = z - m[q] - np.log(s[q])
    return -lp[win].mean()


def ece(p, y, bins=10):
    """기대 보정 오차 — 예측 확률과 실제 승률의 평균 괴리."""
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.digitize(p, edges[1:-1])
    out = 0.0
    for k in range(bins):
        m = b == k
        if m.sum():
            out += m.mean() * abs(p[m].mean() - y[m].mean())
    return out


def probs(raw, q, n_races, T):
    z = raw * T
    m = np.zeros(n_races)
    np.maximum.at(m, q, z)
    e = np.exp(z - m[q])
    s = np.zeros(n_races)
    np.add.at(s, q, e)
    return e / s[q]


def main():
    tr, va = C.load("train"), C.load("valid")
    print("train %s · valid %s (%d경주)"
          % (tr.shape, va.shape, va["race_id"].nunique()))

    # ── 지표 방향 — train 에서 정한다 ────────────────────────────────
    # 승리마의 평균 백분위가 50 보다 크면 '높을수록 좋다'. 작으면 뒤집는다.
    direction, skipped = {}, []
    tr_win = tr["y_win"].to_numpy(bool)
    for ax in AXES:
        for f in AXIS_FEATURES[ax]:
            if f not in tr.columns:
                skipped.append((f, "컬럼 없음"))
                continue
            if not pd.api.types.is_numeric_dtype(tr[f]):
                skipped.append((f, "범주형"))
                continue
            p = pct_within_race(tr, f)
            if np.nanstd(p) < 1e-6:
                skipped.append((f, "경주 내 상수"))
                continue
            direction[f] = 1.0 if p[tr_win].mean() >= 50.0 else -1.0

    n_used = len(direction)
    print("지표 %d개 채택 · %d개 제외" % (n_used, len(skipped)))
    for f, why in skipped:
        print("    제외 %-24s %s" % (f, why))
    print("")

    # ── 축 점수 0~100 (그룹 안 지표 점수의 산술 평균) ────────────────
    ax_va = {}
    for ax in AXES:
        cols = [f for f in AXIS_FEATURES[ax] if f in direction]
        M = np.stack([pct_within_race(va, f) if direction[f] > 0
                      else 100.0 - pct_within_race(va, f) for f in cols], 1)
        ax_va[ax] = M.mean(1)
        print("  %-12s 지표 %2d개   축 점수 평균 %.1f  표준편차 %.1f  범위 %.0f~%.0f"
              % (ax, len(cols), ax_va[ax].mean(), ax_va[ax].std(),
                 ax_va[ax].min(), ax_va[ax].max()))

    q = race_index(va)
    nq = q.max() + 1
    win = va["y_win"].to_numpy(bool)
    y = va["y_win"].to_numpy(float)

    print("")
    print("=" * 96)
    print("온도 T — valid 1,254경주에서 logloss 최소화")
    print("=" * 96)
    print("%-22s%8s%12s%12s%12s%10s%10s"
          % ("가중치", "T*", "ll(T*)", "ll(0.0445)", "ll(0.1)", "ECE(T*)", "top-1"))
    print("-" * 96)

    rows = []
    for name, w in PRESETS.items():
        raw = sum(w[k] * ax_va[k] for k in AXES) / 100.0
        r = optimize.minimize_scalar(nll, bounds=(1e-3, 3.0), method="bounded",
                                     args=(raw, q, win, nq),
                                     options=dict(xatol=1e-5))
        T = float(r.x)
        ll_t, ll_s, ll_i = (nll(T, raw, q, win, nq), nll(SPEC_T, raw, q, win, nq),
                            nll(IMPL_T, raw, q, win, nq))
        e = ece(probs(raw, q, nq, T), y)
        top1 = C.evaluate(va, raw)["top1"]          # 순위만 보므로 T 와 무관
        rows.append((name, T, ll_t, ll_s, ll_i, e, top1))
        print("%-22s%8.4f%12.4f%12.4f%12.4f%10.4f%9.1f%%"
              % (name, T, ll_t, ll_s, ll_i, e, top1))

    print("-" * 96)
    base = C.evaluate(va, np.zeros(len(va)))
    print("무작위 기준 logloss %.4f (경주 평균 %.1f두)"
          % (np.log(len(va) / nq), len(va) / nq))

    Ts = np.array([r[1] for r in rows])
    print("")
    print("=" * 96)
    print("판정")
    print("=" * 96)
    print("  T* 범위 %.4f ~ %.4f  (가중치 5종)" % (Ts.min(), Ts.max()))
    for label, t in (("명세 0.0445", SPEC_T), ("구현 0.1", IMPL_T)):
        d = np.array([nll(t, sum(PRESETS[n][k] * ax_va[k] for k in AXES) / 100.0,
                          q, win, nq) - r[2] for n, r in zip(PRESETS, rows)])
        print("  %-12s → T* 대비 logloss 손해 평균 %+.4f  (최대 %+.4f)"
              % (label, d.mean(), d.max()))


if __name__ == "__main__":
    main()
