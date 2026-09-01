# -*- coding: utf-8 -*-
"""인기도(F6) 유무 비교 — 펀더멘털 피처가 실제로 기여하는가.

주말 실시간 모델은 F6 를 못 쓴다(발주 전 배당 미제공, 실측). 그래서 F6 없는 성능이
곧 실시간 예측의 실력이다. F6 있는 모델이 잘 나오는 건 시장을 따라한 것일 수 있다.
"""
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np, pandas as pd
import schema_v2 as S
from window_study_helpers import fit, evaluate, market_baseline

tr = pd.read_parquet(f"{S.MODEL_DIR}/train.parquet")
va = pd.read_parquet(f"{S.MODEL_DIR}/valid.parquet")
te = pd.read_parquet(f"{S.MODEL_DIR}/test.parquet")
allc = [c for c in S.features() if c in tr.columns]
noF6 = [c for c in S.features(exclude_tier=("G",)) if c in tr.columns]
onlyF6 = [c for c in allc if c.startswith("F6_")]

rows = [market_baseline(te)]
for tag, cols in (("전체 (F6 포함)", allc),
                  ("F6 제외 = 주말 실시간용", noF6),
                  ("F6 만", onlyF6)):
    m = fit(tr, va, cols)
    r = evaluate(m, te, cols, f"{tag} [{len(cols)}피처]")
    r["iters"] = m.best_iteration
    rows.append(r)
    print(f"  {r['tag']:<32s} top1 {r['top1']*100:5.1f}%  top3 {r['top3']*100:5.1f}%  "
          f"ndcg3 {r['ndcg3']:.4f}  ({m.best_iteration}it)")
    if tag == "F6 제외 = 주말 실시간용":
        imp = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False)
        print("\n    상위 기여 피처 (gain)")
        for k, v in imp.head(12).items():
            print(f"      {k:28s} {v/imp.sum()*100:5.1f}%")
        print()

b = rows[0]
se = np.sqrt(b["top1"] * (1 - b["top1"]) / b["races"])
print(f"  {b['tag']:<32s} top1 {b['top1']*100:5.1f}%  top3 {b['top3']*100:5.1f}%")
print(f"  ※ 표준오차 ≈ {se*100:.1f}%p")
os.makedirs("experiments/out", exist_ok=True)
json.dump(rows, open("experiments/out/ablation.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
