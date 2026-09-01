# -*- coding: utf-8 -*-
"""
진단 2단계 — 의심 3건을 하나씩 고쳐서 뭐가 진짜 원인인지 가린다.

의심
  1. 라벨 — y_rel = max(0, dusu−ord) 는 1착(9점)과 2착(8점) 차이가 11%뿐이다.
     Top-1 이 목표인데 라벨이 그걸 강조하지 않는다. 게다가 두수가 다르면
     같은 착순의 점수가 달라져 경주 간 기준이 어긋난다.
  2. 범주형 — cat.codes 를 float 로 넘기면 LightGBM 이 '순서 있는 수'로 오해한다.
     F2_sire_id(935종) 에서 특히 해롭다.
  3. 경주 내 상수 피처 8개 — 랭킹 모델에 직접 기여가 0 (상호작용으로만 쓰임).

F6(인기도) 를 뺀 조건에서 본다. 주말 실시간 모델이 그 조건이고,
F6 가 있으면 시장 신호에 가려 다른 문제가 안 보인다.

실행: python experiments/fix_test.py
"""
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import lightgbm as lgb
import schema_v2 as S

SEED = 42
BASE = dict(metric="ndcg", ndcg_eval_at=[1, 3], learning_rate=0.05, num_leaves=63,
            min_child_samples=40, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=SEED, num_threads=0)

tr = pd.read_parquet(f"{S.MODEL_DIR}/train.parquet")
va = pd.read_parquet(f"{S.MODEL_DIR}/valid.parquet")
te = pd.read_parquet(f"{S.MODEL_DIR}/test.parquet")
COLS = [c for c in S.features(exclude_tier=("G",)) if c in tr.columns]
CATS = [c for c in S.CATEGORICAL if c in COLS]
RACE_CONST = ["F4_month", "F4_track_moist", "F4_weather", "F5_race_n_front",
              "X_dusu", "X_grade", "X_rcDist"]


def grade_label(df):
    """1착을 확실히 띄우는 등급 라벨. 두수와 무관해 경주 간 기준이 같다."""
    o = df["y_ord"].to_numpy()
    return np.where(o == 1, 4, np.where(o == 2, 2, np.where(o == 3, 1, 0))).astype(int)


def X(df, cols, native_cat):
    d = df[cols].copy()
    for c in cols:
        if str(d[c].dtype) == "category":
            if native_cat:
                d[c] = d[c].cat.codes.astype("int32")     # LightGBM 이 범주로 취급
            else:
                d[c] = d[c].cat.codes.replace(-1, np.nan).astype("float32")
        else:
            d[c] = d[c].astype("float32")
    return d


def grp(df):
    return df.groupby("race_id", sort=False).size().to_numpy()


def run(tag, cols, label, native_cat, objective="lambdarank"):
    ytr = grade_label(tr) if label == "grade" else tr["y_rel"].to_numpy()
    yva = grade_label(va) if label == "grade" else va["y_rel"].to_numpy()
    p = {**BASE, "objective": objective}
    cat = [c for c in CATS if c in cols] if native_cat else []
    if objective == "binary":
        ytr = (tr["y_ord"] == 1).astype(int).to_numpy()
        yva = (va["y_ord"] == 1).astype(int).to_numpy()
        p = {**BASE, "objective": "binary", "metric": "auc"}
        dtr = lgb.Dataset(X(tr, cols, native_cat), ytr, categorical_feature=cat)
        dva = lgb.Dataset(X(va, cols, native_cat), yva, categorical_feature=cat, reference=dtr)
    else:
        dtr = lgb.Dataset(X(tr, cols, native_cat), ytr, group=grp(tr), categorical_feature=cat)
        dva = lgb.Dataset(X(va, cols, native_cat), yva, group=grp(va),
                          categorical_feature=cat, reference=dtr)
    m = lgb.train(p, dtr, num_boost_round=1500, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(120, verbose=False)])
    s = m.predict(X(te, cols, native_cat))
    d = pd.DataFrame({"r": te["race_id"].to_numpy(), "s": s, "o": te["y_ord"].to_numpy()})
    d["rk"] = d.groupby("r")["s"].rank(ascending=False, method="first")
    top = d[d["rk"] == 1]
    return {"tag": tag, "top1": float((top["o"] == 1).mean()),
            "top3": float((top["o"] <= 3).mean()), "iters": m.best_iteration,
            "nfeat": len(cols)}


fav = te[te["F6_mkt_rank"] == 1]
mkt = {"tag": "시장 (인기1위)", "top1": float((fav["y_ord"] == 1).mean()),
       "top3": float((fav["y_ord"] <= 3).mean()), "iters": 0, "nfeat": 0}
rows = [mkt]

TESTS = [
    ("① 현재 그대로 (y_rel + cat.codes)",  COLS, "rel",   False, "lambdarank"),
    ("② 라벨만 수정 (등급 라벨)",           COLS, "grade", False, "lambdarank"),
    ("③ 범주형만 수정 (native cat)",       COLS, "rel",   True,  "lambdarank"),
    ("④ 둘 다 수정",                      COLS, "grade", True,  "lambdarank"),
    ("⑤ ④ + 경주내 상수 제거",             [c for c in COLS if c not in RACE_CONST],
                                          "grade", True,  "lambdarank"),
    ("⑥ 이진분류 (y_win)",                COLS, "-",     True,  "binary"),
]
print(f"F6 제외 조건 · 피처 {len(COLS)}개 · test {te['race_id'].nunique():,}경주\n")
print(f"  {'조건':<34s} {'top1':>7s} {'top3':>7s} {'iter':>6s}")
print("  " + "-" * 58)
print(f"  {mkt['tag']:<34s} {mkt['top1']*100:6.1f}% {mkt['top3']*100:6.1f}%      -")
for tag, cols, lab, nc, obj in TESTS:
    r = run(tag, cols, lab, nc, obj)
    rows.append(r)
    print(f"  {r['tag']:<34s} {r['top1']*100:6.1f}% {r['top3']*100:6.1f}% {r['iters']:>6d}")

se = np.sqrt(mkt["top1"] * (1 - mkt["top1"]) / te["race_id"].nunique())
print(f"\n  ※ 표준오차 ≈ {se*100:.1f}%p")
os.makedirs("experiments/out", exist_ok=True)
json.dump(rows, open("experiments/out/fix_test.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
