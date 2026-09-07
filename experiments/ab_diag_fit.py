# -*- coding: utf-8 -*-
"""
A/B 진단 1단계 — 학습하며 valid 예측값·피처 기여도를 전부 저장한다.
쌍체 검정과 하위군 분석은 저장된 결과로 재학습 없이 한다.

    python experiments/ab_diag_fit.py
산출: experiments/out/ab_diag.npz , ab_diag_meta.json
"""
import sys, json, io, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

sys.stdout.reconfigure(encoding="utf-8")
TEAM = Path(r"c:/Users/SSAFY/Desktop/말고리즘/S15P21A304")
sys.path.insert(0, str(TEAM / "docs" / "model"))
import common as C                                     # noqa: E402

HERE = Path(r"c:/Users/SSAFY/Desktop/주제선정")
ARMS = {"A": HERE/"dataset"/"v2_before_2004", "B": HERE/"dataset"/"v2"}
SEEDS = (20260901, 20260902, 20260903)
PARAMS = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[3],
              lambdarank_truncation_level=5, learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, verbose=-1, num_threads=4)
N_ROUND = 200

preds, imps, meta = {}, {}, {}
t0 = time.time()
for arm, path in ARMS.items():
    C.DATASET = path
    tr, va = C.load("train"), C.load("valid")
    meta[arm] = {"n_train": int(len(tr)), "n_races_train": int(tr.race_id.nunique())}
    if "valid_meta" not in meta:                       # valid 는 A/B 동일 — 한 번만 저장
        keep = ["race_id", "y_win", "y_plc", "X_dusu", "X_rcDist", "X_grade",
                "X_rating", "F6_mkt_rank", "F6_mkt_prob", "rcDate", "meet"]
        keep = [c for c in keep if c in va.columns]
        va[keep].to_parquet(HERE/"experiments"/"out"/"valid_meta.parquet")
        meta["valid_meta"] = keep
        meta["n_valid_rows"] = int(len(va))
    # train 쪽 진단용 요약 (희석 확인)
    meta[arm]["fill"] = {c: float(tr[c].notna().mean())
                         for c in ("X_rating", "F1_tr_sessions_28d", "F2_ebv_prize",
                                   "F5_g3f_time", "F1_layoff_days") if c in tr.columns}
    for xp, tag in ((False, "77"), (True, "73")):
        cols = C.feature_cols(exclude_pop=xp)
        a, b = C.encode(tr, va, cols)
        Xa = a[cols].to_numpy(float); Xb = b[cols].to_numpy(float)
        ds = lgb.Dataset(Xa, label=a["y_rel"].to_numpy(), group=C.race_groups(a))
        for s in SEEDS:
            p = dict(PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s)
            m = lgb.train(p, ds, num_boost_round=N_ROUND)
            preds[f"{arm}_{tag}_{s}"] = m.predict(Xb)
            imps[f"{arm}_{tag}_{s}"] = m.feature_importance("gain").tolist()
            print(f"  {arm} {tag}피처 seed={s}  {time.time()-t0:.0f}s", flush=True)
        meta[f"cols_{tag}"] = cols

np.savez_compressed(HERE/"experiments"/"out"/"ab_diag.npz", **preds)
meta["importance"] = imps
io.open(HERE/"experiments"/"out"/"ab_diag_meta.json", "w", encoding="utf-8").write(
    json.dumps(meta, ensure_ascii=False, indent=1))
print(f"완료 {time.time()-t0:.0f}s")
