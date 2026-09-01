# -*- coding: utf-8 -*-
"""window_study 의 학습·평가 함수를 실험들이 공유한다."""
import numpy as np, pandas as pd, lightgbm as lgb
SEED = 42
PARAMS = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[3],
              learning_rate=0.05, num_leaves=63, min_child_samples=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, verbose=-1, seed=SEED, num_threads=0)
ROUNDS = 800

def prep(df, cols):
    X = df[cols].copy()
    for c in X.columns:
        if str(X[c].dtype) == "category":
            X[c] = X[c].cat.codes.replace(-1, np.nan)
    return X.astype("float32")

def groups(df):
    return df.groupby("race_id", sort=False).size().to_numpy()

def fit(tr, va, cols, rounds=ROUNDS):
    dtr = lgb.Dataset(prep(tr, cols), label=tr["y_rel"], group=groups(tr))
    dva = lgb.Dataset(prep(va, cols), label=va["y_rel"], group=groups(va), reference=dtr)
    return lgb.train(PARAMS, dtr, num_boost_round=rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(80, verbose=False)])

def evaluate(model, te, cols, tag):
    s = model.predict(prep(te, cols))
    d = pd.DataFrame({"race_id": te["race_id"].to_numpy(), "score": s,
                      "y_ord": te["y_ord"].to_numpy()})
    d["rank"] = d.groupby("race_id")["score"].rank(ascending=False, method="first")
    top = d[d["rank"] == 1]
    def ndcg(g):
        g = g.sort_values("rank"); rel = (g["y_ord"].head(3) <= 3).to_numpy().astype(float)
        dcg = (rel / np.log2(np.arange(2, len(rel) + 2))).sum()
        n = min(3, len(g)); ideal = (np.ones(n) / np.log2(np.arange(2, n + 2))).sum()
        return dcg / ideal if ideal else np.nan
    return {"tag": tag, "races": int(d["race_id"].nunique()),
            "top1": float((top["y_ord"] == 1).mean()),
            "top3": float((top["y_ord"] <= 3).mean()),
            "ndcg3": float(d.groupby("race_id").apply(ndcg, include_groups=False).mean())}

def market_baseline(te):
    fav = te[te["F6_mkt_rank"] == 1]
    return {"tag": "시장 (인기1위)", "races": int(te["race_id"].nunique()),
            "top1": float((fav["y_ord"] == 1).mean()),
            "top3": float((fav["y_ord"] <= 3).mean()), "ndcg3": float("nan")}
