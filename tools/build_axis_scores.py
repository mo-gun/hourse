# -*- coding: utf-8 -*-
"""축 점수 산출기 — entry_feature_score 에 넣을 0~100 을 모델로 만든다 (AI-05).

## 왜 필요한가

지표값을 경주 안에서 순위로 바꿔 평균하면 top-1 28.3% 가 나온다. 같은 데이터로
모델을 돌리면 33.5% 다 (tools/fit_group_inner_weights.py · basemodel/README.md §5).
**5%p 는 표준오차 ±1.27%p 의 4배다.** 축 점수를 순위 환산으로 채우면 그만큼 버린다.

그래서 축마다 랭커를 하나씩 학습한다 — 그 축에 속한 지표만 보는 랭커다. 출력값을
경주 안에서 순위 환산해 0~100 으로 만들면 그게 축 점수다.

## 왜 축마다 따로 학습하나 (전체 모델 하나가 아니라)

제품 계약이 **유저가 축 비율을 실시간으로 바꾸는 것**이라, 축 점수가 서로 독립으로
나와야 한다. 전체 모델 하나를 돌려 놓고 축별로 쪼개면 그 쪼갠 값은 "다른 축이 이렇게
들어왔을 때의 기여"라서 비율을 바꾸는 순간 틀린다.

## 왕복 검증 — 이게 이 스크립트의 요점이다

만든 점수를 **제품 공식에 그대로 태워** 성적을 잰다.
    raw = Σ( groupWeights[k] × ax_k ) / 100
    p   = softmax(raw × T),  경주 안에서
28.3% 가 아니라 33% 대가 나와야 산출기가 맞은 것이다.

## 규약
    · train 으로 학습, valid 로만 평가. test 는 열지 않고 game 은 쓰지 않는다
    · 시드 3개, 평균±표준편차
    · 정렬을 바꾸지 않는다 (race_id 연속 블록 — LightGBM group 이 그걸 가정한다)
    · 채점은 팀 채점기 docs/model/common.py 를 거친다

    python tools/build_axis_scores.py
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import optimize

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_temperature import AXES, AXIS_FEATURES, race_index, nll, C, TEAM   # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"
SEEDS = (20260901, 20260902, 20260903)
N_ROUND = 200
PARAMS = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[3],
              lambdarank_truncation_level=5, learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)


def rank100(v, rid):
    return (pd.Series(v).groupby(rid.values).rank(pct=True) * 100).to_numpy(float)


def axis_cols(tr):
    """축별 지표 — 범주형은 뺀다 (LightGBM 이 코드값을 크기로 읽으면 안 된다)."""
    out = {}
    for ax in AXES:
        out[ax] = [f for f in AXIS_FEATURES[ax]
                   if f in tr.columns and pd.api.types.is_numeric_dtype(tr[f])]
    return out


def main():
    tr, va = C.load("train"), C.load("valid")
    cols = axis_cols(tr)
    rid_va = va["race_id"]
    g_tr = C.race_groups(tr)
    y_tr = tr["y_rel"].to_numpy()
    print("학습 %s / 평가 %s (%d경주)" % (tr.shape, va.shape, va["race_id"].nunique()))
    print("축별 지표 수  " + "  ".join("%s %d" % (a, len(cols[a])) for a in AXES))
    print("")

    # ── 축마다 랭커 3시드 ────────────────────────────────────────────
    raw_axis = {}
    for ax in AXES:
        Xtr = tr[cols[ax]].to_numpy(float)
        Xva = va[cols[ax]].to_numpy(float)
        preds = []
        for sd in SEEDS:
            ds = lgb.Dataset(Xtr, label=y_tr, group=g_tr)   # 순서 그대로
            m = lgb.train(dict(PARAMS, seed=sd, bagging_seed=sd, feature_fraction_seed=sd),
                          ds, num_boost_round=N_ROUND)
            preds.append(m.predict(Xva))
        P = np.stack(preds)
        raw_axis[ax] = P.mean(0)
        t1 = [C.evaluate(va, p)["top1"] for p in preds]
        print("  %-12s 지표 %2d개   축 단독 top-1 %.1f%% ± %.1f"
              % (ax, len(cols[ax]), np.mean(t1), np.std(t1)))

    AX = np.stack([rank100(raw_axis[ax], rid_va) for ax in AXES], 1)   # 0~100

    # ── 왕복 검증: 제품 공식을 그대로 태운다 ─────────────────────────
    q = race_index(va); nq = q.max() + 1
    win = va["y_win"].to_numpy(bool)
    presets = {
        "균등": {k: 100 / 6 for k in AXES},
        "기본형": dict(CONDITION=17, SPEED=17, RUNNING=12, JOCKEY=22, ENVIRONMENT=19, ABILITY=13),
        "사람형": dict(CONDITION=15, SPEED=13, RUNNING=12, JOCKEY=35, ENVIRONMENT=15, ABILITY=10),
        "상승세형": dict(CONDITION=32, SPEED=16, RUNNING=12, JOCKEY=20, ENVIRONMENT=12, ABILITY=8),
    }
    rng = np.random.default_rng(20260914)
    W = []
    while len(W) < 100:
        g = rng.dirichlet(np.ones(6)) * 100
        if g.max() <= 40:
            W.append(g)
    raws = [AX @ np.array(g) / 100.0 for g in W]

    def total(t):
        return float(np.mean([nll(t, r, q, win, nq) for r in raws]))

    T = float(optimize.minimize_scalar(total, bounds=(1e-4, 5.0), method="bounded",
                                       options=dict(xatol=1e-7)).x)

    print("")
    print("=" * 78)
    print("왕복 검증 — 만든 축 점수를 제품 공식에 그대로 태운다")
    print("=" * 78)
    print("%-14s%10s%10s%12s" % ("가중치", "top-1", "top-3", "logloss"))
    print("-" * 78)
    for name, w in presets.items():
        raw = AX @ np.array([w[k] for k in AXES]) / 100.0
        m = C.evaluate(va, raw)
        print("%-14s%9.1f%%%9.1f%%%12.4f"
              % (name, m["top1"], m["top3"], nll(T, raw, q, win, nq)))
    print("-" * 78)
    print("비교  순위환산 축(현행안) 28.3% · LGB 73피처 33.5% · 시장 39.3%")
    print("적합 T = %.4f   (명세 0.0445 · 순위환산 축에서 쟀던 값과 같은 자리인지 본다)" % T)

    np.save(OUT / "axis_scores_valid.npy", AX)
    (OUT / "axis_meta.json").write_text(json.dumps(
        {"axes": AXES, "temperature": round(T, 4), "seeds": list(SEEDS),
         "n_round": N_ROUND, "axis_features": {a: cols[a] for a in AXES},
         "score_contract": "raw = (Σ_k groupWeights[k] * ax_k) / 100 ; p = softmax(raw * T) 경주 내",
         "score_scale": "축 점수 = 축 랭커 출력의 경주 내 순위 백분위 × 100"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("→ out/axis_scores_valid.npy · out/axis_meta.json")


if __name__ == "__main__":
    main()
