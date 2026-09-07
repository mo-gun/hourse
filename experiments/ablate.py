# -*- coding: utf-8 -*-
"""어느 제거가 성능을 깎았는지 하나씩 분리. 인기도 제외(실시간) 조건."""
import sys, warnings, numpy as np, pandas as pd, lightgbm as lgb
warnings.filterwarnings("ignore")
D = sys.argv[1]; sys.path.insert(0, D)
import schema_v2 as S

SEED = 20260901
G = ["F1_prize_life", "F1_prize_life_z", "F1_prize_life_rk"]
M = ["F4_hr_meet_win_rate"]
T = ["F4_month"]

def load(s):
    return S.usable(S.clean(pd.read_parquet(f"{D}/model/{s}.parquet")))

def grp(d): return d.groupby("race_id", sort=False).size().to_numpy()

def run(tr, va, cols, seed):
    m = lgb.LGBMRanker(objective="lambdarank", n_estimators=200, learning_rate=0.05,
                       num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=5.0,
                       random_state=seed, n_jobs=-1, verbose=-1)
    m.fit(tr[cols], tr["y_rel"], group=grp(tr))
    p = m.predict(va[cols])
    t = pd.DataFrame({"r": va.race_id.to_numpy(), "s": p, "w": va.y_win.to_numpy()})
    mx = t.groupby("r")["s"].transform("max")
    return t[t.s >= mx - 1e-12].groupby("r")["w"].mean().mean() * 100

tr, va = load("train"), load("valid")
base = S.features(exclude_tier=("G",))            # 73 (현재)
full = base + G + M + T                           # 78 (이전)
cases = [("전부 포함 (이전 78)", full),
         ("− prize_life 3개", base + M + T),
         ("− hr_meet_win_rate", base + G + T),
         ("− month", base + G + M),
         ("전부 제거 (현재 73)", base)]
print(f"valid {va.race_id.nunique():,}경주 · seed 3개 평균\n")
print(f"{'구성':24s} {'피처':>4s}  {'top-1':>7s}  {'표준편차':>7s}")
print("-" * 50)
res = {}
for lab, cols in cases:
    cols = [c for c in cols if c in tr.columns]
    v = [run(tr, va, cols, s) for s in (SEED, SEED + 1, SEED + 2)]
    res[lab] = np.mean(v)
    print(f"{lab:24s} {len(cols):>4}  {np.mean(v):6.2f}%  ±{np.std(v):5.2f}")
print()
b = res["전부 포함 (이전 78)"]
for lab in list(res)[1:]:
    print(f"  {lab:24s} {res[lab] - b:+.2f}%p")
