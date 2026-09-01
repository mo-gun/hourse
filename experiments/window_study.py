# -*- coding: utf-8 -*-
"""
학습 기간 실험 — "데이터는 많을수록 좋은가?" 를 실측으로 답한다.

두 가지를 본다.
  A. 학습 **시작점**을 언제로 잡을 것인가 (2010 vs 2015 vs 2018 vs 2021 vs 2023)
  B. **최신 데이터**를 학습에 넣는 효과 (train 만 vs train+valid)

경마는 정적인 문제가 아니다. 기수·조교사·말이 계속 바뀌고, 등급 체계·상금·주로도
바뀐다. 오래된 데이터가 도움이 될지 방해가 될지는 선험적으로 알 수 없으므로 측정한다.

test 셋(2026-05-15~08-30, 802경주)은 모든 조건에서 동일하다.

실행: python experiments/window_study.py
"""
import os, sys, json, time
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import lightgbm as lgb
import schema_v2 as S

SEED = 42
PARAMS = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[3],
              learning_rate=0.05, num_leaves=63, min_child_samples=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, verbose=-1, seed=SEED, num_threads=0)
ROUNDS = 600


def load():
    tr = pd.read_parquet(f"{S.MODEL_DIR}/train.parquet")
    va = pd.read_parquet(f"{S.MODEL_DIR}/valid.parquet")
    te = pd.read_parquet(f"{S.MODEL_DIR}/test.parquet")
    return tr, va, te


def prep(df, cols):
    X = df[cols].copy()
    for c in X.columns:
        if str(X[c].dtype) == "category":
            X[c] = X[c].cat.codes.replace(-1, np.nan)
    return X.astype("float32")


def groups(df):
    return df.groupby("race_id", sort=False).size().to_numpy()


def evaluate(model, te, cols, tag):
    """경주 단위 지표. 예측 점수 최상위가 실제 1착인가."""
    s = model.predict(prep(te, cols))
    d = pd.DataFrame({"race_id": te["race_id"].to_numpy(),
                      "score": s, "y_ord": te["y_ord"].to_numpy(),
                      "mkt": te["F6_mkt_rank"].to_numpy()})
    d["rank"] = d.groupby("race_id")["score"].rank(ascending=False, method="first")
    top = d[d["rank"] == 1]
    top1 = (top["y_ord"] == 1).mean()
    top3 = (top["y_ord"] <= 3).mean()
    # NDCG@3
    def ndcg(g):
        g = g.sort_values("rank")
        rel = (g["y_ord"].head(3) <= 3).to_numpy().astype(float)
        dcg = (rel / np.log2(np.arange(2, len(rel) + 2))).sum()
        ideal = (np.ones(min(3, len(g))) / np.log2(np.arange(2, min(3, len(g)) + 2))).sum()
        return dcg / ideal if ideal else np.nan
    nd = d.groupby("race_id").apply(ndcg, include_groups=False).mean()
    return {"tag": tag, "races": int(d["race_id"].nunique()),
            "top1": float(top1), "top3": float(top3), "ndcg3": float(nd)}


def market_baseline(te):
    fav = te[te["F6_mkt_rank"] == 1]
    return {"tag": "시장 (인기1위)", "races": int(te["race_id"].nunique()),
            "top1": float((fav["y_ord"] == 1).mean()),
            "top3": float((fav["y_ord"] <= 3).mean()), "ndcg3": np.nan}


def fit(tr, va, cols):
    dtr = lgb.Dataset(prep(tr, cols), label=tr["y_rel"], group=groups(tr))
    dva = lgb.Dataset(prep(va, cols), label=va["y_rel"], group=groups(va), reference=dtr)
    return lgb.train(PARAMS, dtr, num_boost_round=ROUNDS, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(60, verbose=False)])


def main():
    tr, va, te = load()
    cols = [c for c in S.features() if c in tr.columns]
    print(f"피처 {len(cols)}개 · test {te['race_id'].nunique():,}경주\n")
    rows = [market_baseline(te)]

    print("── A. 학습 시작점 (전부 2025-11 까지, valid 로 early stop) ──")
    for y0 in (2010, 2015, 2018, 2021, 2023):
        sub = tr[tr["rcDate"] >= y0 * 10000]
        if sub["race_id"].nunique() < 500:
            continue
        t = time.time()
        m = fit(sub, va, cols)
        r = evaluate(m, te, cols, f"{y0}~ ({sub['race_id'].nunique():,}경주)")
        r["races_tr"] = int(sub["race_id"].nunique())
        r["iters"] = m.best_iteration
        rows.append(r)
        print(f"  {r['tag']:<24s} top1 {r['top1']*100:5.1f}%  top3 {r['top3']*100:5.1f}%  "
              f"ndcg3 {r['ndcg3']:.4f}  ({m.best_iteration}it, {time.time()-t:.0f}s)")

    print("\n── B. 최신 데이터 포함 효과 ──")
    #   train 만 vs train+valid. 후자는 early stop 을 못 쓰므로 A 의 최적 iter 를 쓴다.
    best_iter = max((r.get("iters", ROUNDS) for r in rows if "iters" in r), default=300)
    full = pd.concat([tr, va], ignore_index=True).sort_values(["rcDate", "meet", "rcNo"])
    for name, data in (("train 만 (~2025-11)", tr), ("train+valid (~2026-05)", full)):
        d = lgb.Dataset(prep(data, cols), label=data["y_rel"], group=groups(data))
        m = lgb.train({**PARAMS}, d, num_boost_round=best_iter)
        r = evaluate(m, te, cols, name)
        rows.append(r)
        print(f"  {name:<24s} top1 {r['top1']*100:5.1f}%  top3 {r['top3']*100:5.1f}%  "
              f"ndcg3 {r['ndcg3']:.4f}  ({best_iter}it 고정)")

    b = rows[0]
    print(f"\n  {b['tag']:<24s} top1 {b['top1']*100:5.1f}%  top3 {b['top3']*100:5.1f}%")
    se = np.sqrt(b["top1"] * (1 - b["top1"]) / b["races"])
    print(f"  ※ test {b['races']}경주 기준 top1 표준오차 ≈ {se*100:.1f}%p — "
          f"이보다 작은 차이는 우열로 읽지 말 것")

    os.makedirs("experiments/out", exist_ok=True)
    json.dump(rows, open("experiments/out/window_study.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n저장 experiments/out/window_study.json")


if __name__ == "__main__":
    main()
