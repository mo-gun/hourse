# -*- coding: utf-8 -*-
"""
v2.2.0(77피처) 재학습 — EDA 반영이 성능에 어떤 영향을 줬는지 확인.

README §5 표를 같은 조건에서 다시 만들고, 이전 82피처와 나란히 비교한다.
평가는 valid 로 한다. **test 는 열지 않는다** (§3-③ 최종 1회 규칙).

    python retrain_v220.py <dataset_dir>
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
D = sys.argv[1] if len(sys.argv) > 1 else "."
sys.path.insert(0, D)
import schema_v2 as S                                    # noqa: E402

SEED = 20260901
DROPPED = ["F1_prize_life", "F1_prize_life_z", "F1_prize_life_rk",
           "F4_month", "F4_hr_meet_win_rate"]


def load(split):
    df = pd.read_parquet(f"{D}/model/{split}.parquet")
    return S.usable(S.clean(df))


def groups(df):
    """race_id 연속 블록의 크기. 정렬을 바꾸지 않는다 (§3-④)."""
    return df.groupby("race_id", sort=False).size().to_numpy()


def top_k(df, score, k=1):
    """경주마다 score 최대인 말을 골라 적중률. 동점이면 1/k 가중."""
    t = pd.DataFrame({"r": df["race_id"].to_numpy(), "s": np.asarray(score),
                      "w": df["y_win"].to_numpy(), "p": df["y_plc"].to_numpy()})
    mx = t.groupby("r")["s"].transform("max")
    top = t[t["s"] >= mx - 1e-12]
    g = top.groupby("r")
    return (g["w"].mean().mean() * 100, g["p"].mean().mean() * 100)


def market(df):
    return top_k(df, df["F6_mkt_prob"].fillna(-1).to_numpy())


def fit(tr, va, cols, rounds=200, seed=SEED):
    m = lgb.LGBMRanker(
        objective="lambdarank", n_estimators=rounds, learning_rate=0.05,
        num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=5.0,
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    m.fit(tr[cols], tr["y_rel"], group=groups(tr))
    return m, m.predict(va[cols])


def main():
    t0 = time.time()
    tr, va = load("train"), load("valid")
    print(f"train {len(tr):,}행 / {tr.race_id.nunique():,}경주")
    print(f"valid {len(va):,}행 / {va.race_id.nunique():,}경주")
    print(f"schema {S.SCHEMA_VERSION}\n")

    new_all = S.features()                       # 77
    new_nog = S.features(exclude_tier=("G",))    # 73
    old_all = new_all + DROPPED                  # 82 (재현용)
    old_nog = new_nog + DROPPED                  # 78
    f6 = [c for c in new_all if c.startswith("F6_")]

    m1, m3 = market(va)
    rows = [("시장 (인기 1위마)", "—", m1, m3)]

    for label, cols in [
        ("인기도 포함 — v2.2.0", new_all),
        ("인기도 포함 — v2.1.0(이전)", old_all),
        ("인기도만 (F6 4개)", f6),
        ("인기도 제외 — v2.2.0", new_nog),
        ("인기도 제외 — v2.1.0(이전)", old_nog),
    ]:
        cols = [c for c in cols if c in tr.columns]
        vs = [top_k(va, fit(tr, va, cols, seed=SEED + k)[1]) for k in range(3)]
        a = float(np.mean([v[0] for v in vs])); b = float(np.mean([v[1] for v in vs]))
        sd = float(np.std([v[0] for v in vs]))
        rows.append((label, len(cols), a, b))
        print(f"  {label:30s} {len(cols):>3}피처  top-1 {a:5.2f}% (±{sd:.2f})  top-3 {b:5.2f}%")

    print(f"\n{'':30s} {'피처':>4s}  {'top-1':>7s}  {'top-3':>7s}")
    print("-" * 56)
    for lab, n, a, b in rows:
        print(f"{lab:30s} {str(n):>4s}  {a:6.2f}%  {b:6.2f}%")

    d = dict((r[0], r[2]) for r in rows)
    print(f"\nv2.2.0 − v2.1.0 (인기도 포함): "
          f"{d['인기도 포함 — v2.2.0'] - d['인기도 포함 — v2.1.0(이전)']:+.2f}%p")
    print(f"v2.2.0 − v2.1.0 (인기도 제외): "
          f"{d['인기도 제외 — v2.2.0'] - d['인기도 제외 — v2.1.0(이전)']:+.2f}%p")
    print(f"모델 − 시장  (인기도 포함)    : {d['인기도 포함 — v2.2.0'] - m1:+.2f}%p")
    se = np.sqrt(m1 / 100 * (1 - m1 / 100) / va.race_id.nunique()) * 100
    print(f"valid {va.race_id.nunique():,}경주 표준오차 ±{se:.2f}%p")
    print(f"\n{time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
