# -*- coding: utf-8 -*-
"""
A/B — 2004~2009 원장 백필이 성능을 올리는가.

팀 공통 모듈(docs/model/common.py)을 그대로 쓴다. 데이터 디렉터리만 갈아끼운다.
평가는 valid 로만 한다 (test 는 열지 않는다).

  A  2010~     기존 train 385,514행 / 36,586경주   = 팀 레포 현재 parquet 과 값까지 동일
  B  2004~     백필 train 538,898행 / 51,476경주
  C  2004~ +w  B 와 같은 데이터, 최근 경주에 가중치 (지수감쇠, 반감기 8년)

시드 3개 × {77피처, 73피처} × 3구성 = 18회 학습.

    python experiments/ab_2004.py
"""
import sys, json, time, io
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.stdout.reconfigure(encoding="utf-8")

TEAM = Path(r"c:/Users/SSAFY/Desktop/말고리즘/S15P21A304")
sys.path.insert(0, str(TEAM / "docs" / "model"))
import common as C                                    # noqa: E402

HERE = Path(r"c:/Users/SSAFY/Desktop/주제선정")
ARMS = [
    ("A  2010~ (기존)",   HERE / "dataset" / "v2_before_2004", False),
    ("B  2004~ (백필)",   HERE / "dataset" / "v2",             False),
    ("C  2004~ +가중치",  HERE / "dataset" / "v2",             True),
]
SEEDS = (20260901, 20260902, 20260903)
HALFLIFE_Y = 8.0          # 최근 데이터 가중 — 반감기 8년

PARAMS = dict(                       # baseline_lgbm.py 와 동일
    objective="lambdarank",
    metric="ndcg", ndcg_eval_at=[3],
    lambdarank_truncation_level=5,
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=1,
    verbose=-1, num_threads=4,
)
N_ROUND = 200


def decay_weight(df):
    """경주일 기준 지수감쇠. 같은 경주 안에서는 값이 같으므로 group 가중과 동치."""
    d = pd.to_datetime(df["rcDate"].astype(str), format="%Y%m%d")
    age_y = (d.max() - d).dt.days / 365.25
    return np.power(0.5, age_y / HALFLIFE_Y).to_numpy()


def run(tr, va, exclude_pop, seed, weighted):
    cols = C.feature_cols(exclude_pop=exclude_pop)
    a, b = C.encode(tr, va, cols)
    ds = lgb.Dataset(a[cols].to_numpy(float),
                     label=a["y_rel"].to_numpy(),
                     group=C.race_groups(a),
                     weight=decay_weight(a) if weighted else None)
    p = dict(PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
    m = lgb.train(p, ds, num_boost_round=N_ROUND)
    return C.evaluate(b, m.predict(b[cols].to_numpy(float)))


def main():
    t0 = time.time()
    out = {}
    # 시장 기준선은 데이터셋과 무관하나, valid 를 한 번 읽어야 하므로 A 에서 잡는다
    C.DATASET = ARMS[0][1]
    va0 = C.load("valid")
    mkt = C.evaluate(va0, C.market_scores(va0))
    print(f"valid {mkt['n_races']:,}경주 · top-1 표준오차 ±{mkt['se_top1']:.2f}%p "
          f"(2배 = {2*mkt['se_top1']:.2f}%p 를 넘어야 우열)\n")
    print(f"시장 (인기 1위마)   top-1 {mkt['top1']:.2f}%   top-3 {mkt['top3']:.2f}%\n")
    print(f"{'구성':<22}{'피처':>5}{'top-1':>9}{'표준편차':>9}{'top-3':>9}   시드별")
    print("-" * 82)

    for label, path, weighted in ARMS:
        C.DATASET = path
        tr, va = C.load("train"), C.load("valid")
        for xp, tag in ((False, 77), (True, 73)):
            vals, t3s = [], []
            for s in SEEDS:
                r = run(tr, va, xp, s, weighted)
                vals.append(r["top1"]); t3s.append(r["top3"])
            key = f"{label}|{tag}"
            out[key] = {"top1_mean": float(np.mean(vals)), "top1_std": float(np.std(vals)),
                        "top3_mean": float(np.mean(t3s)), "seeds": vals,
                        "n_train": int(len(tr)), "n_races_train": int(tr.race_id.nunique())}
            print(f"{label:<22}{tag:>5}{np.mean(vals):>8.2f}%{np.std(vals):>8.2f}%"
                  f"{np.mean(t3s):>8.2f}%   {' '.join(f'{v:.2f}' for v in vals)}")
        print()

    print("-" * 82)
    print(f"\n시장 기준선 {mkt['top1']:.2f}% · 표준오차 ±{mkt['se_top1']:.2f}%p\n")
    print("=== A 대비 차이 ===")
    for tag in (77, 73):
        base = out[f"{ARMS[0][0]}|{tag}"]["top1_mean"]
        for label, _, _ in ARMS[1:]:
            d = out[f"{label}|{tag}"]["top1_mean"] - base
            verdict = "유의" if abs(d) > 2 * mkt["se_top1"] else "효과 없음(표준오차 2배 이내)"
            print(f"  {tag}피처  {label:<22} {d:+.2f}%p   {verdict}")

    out["_market"] = mkt
    out["_halflife_years"] = HALFLIFE_Y
    Path(HERE / "experiments" / "out").mkdir(parents=True, exist_ok=True)
    io.open(HERE / "experiments" / "out" / "ab_2004.json", "w", encoding="utf-8").write(
        json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n총 {time.time()-t0:.0f}s · 저장 experiments/out/ab_2004.json")


if __name__ == "__main__":
    main()
