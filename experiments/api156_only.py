# -*- coding: utf-8 -*-
"""
실험 (a) — API156 필드만으로 만들 수 있는 피처로 어디까지 맞추나.

API 를 새로 백필하지 않고, 기존 데이터셋에서 **API156 으로 재현 가능한 컬럼만 남겨**
같은 분할·같은 방식으로 학습한다. 분할은 절대 다시 나누지 않는다 (docs/dataset README §3).

이 결과는 'API156 단독 파이프라인'의 상한선에 가깝다 — API156 에 있는 원천 필드
(착순·주파기록·기수·조교사·배당·주로·날씨)로 지금과 같은 롤링 집계를 만들 수 있기 때문이다.

    python experiments/api156_only.py
"""
import sys, json, io, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

sys.stdout.reconfigure(encoding="utf-8")
TEAM = Path(r"c:/Users/SSAFY/Desktop/말고리즘/S15P21A304")
sys.path.insert(0, str(TEAM / "docs" / "model"))
import common as C                                          # noqa: E402
C.DATASET = TEAM / "docs" / "dataset"                       # 팀 표준 데이터셋
import schema_v2 as S                                       # noqa: E402

SEEDS = (20260901, 20260902, 20260903)
PARAMS = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[3],
              lambdarank_truncation_level=5, learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, verbose=-1, num_threads=4)
N_ROUND = 200

# ── API156 44필드로 재현 불가능한 피처 ────────────────────────────────
#   F2_*          혈통·육종가. API156 에 부마/모마/육종가가 없다
#   F5_*          구간기록. API156 에 구간 통과기록이 전혀 없다
#   F1_tr_*       조교. 별도 API(trcontihi) 소관
UNAVAILABLE_PREFIX = ("F2_", "F5_")
UNAVAILABLE_EXACT = ("F1_tr_sessions_28d", "F1_tr_jk_ridden_28d")

# 참고 — 재현 가능한 근거 (API156 필드 → 데이터셋 피처)
SOURCE_NOTE = {
    "X_age": "pthrAg", "X_sex": "pthrGndr", "X_prd_cty": "pthrNtnlty(국적)",
    "X_rcDist": "cndRaceDs", "X_grade": "cndRaceClas", "X_dusu": "경주별 행 수",
    "X_chulNo": "pthrGtno", "X_wgBudam": "pthrBurdWgt", "X_rating": "pthrRatg",
    "F1_*": "rsutRk·rsutRaceRcd·rsutRkPurse 이력 롤링", "F1_layoff_days": "pthrLatstPtinDt",
    "F3_*": "hrmJckyId·hrmTrarId + rsutRk 이력 롤링",
    "F4_track_moist": "rsutTrckStus", "F4_weather": "rsutWetr",
    "F6_*": "rsutWinPrice(단승 배당)",
}


def avail(cols):
    return [c for c in cols
            if not c.startswith(UNAVAILABLE_PREFIX) and c not in UNAVAILABLE_EXACT]


def run(tr, va, cols, seed):
    a, b = C.encode(tr, va, cols)
    ds = lgb.Dataset(a[cols].to_numpy(float), label=a["y_rel"].to_numpy(),
                     group=C.race_groups(a))
    p = dict(PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
    m = lgb.train(p, ds, num_boost_round=N_ROUND)
    return m.predict(b[cols].to_numpy(float))


def main():
    t0 = time.time()
    tr, va = C.load("train"), C.load("valid")
    full77, full73 = C.feature_cols(), C.feature_cols(exclude_pop=True)
    api50, api46 = avail(full77), avail(full73)

    print(f"train {len(tr):,}두 · valid {len(va):,}두 {va.race_id.nunique():,}경주\n")
    print("API156 으로 재현 불가능한 피처")
    lost = [c for c in full77 if c not in api50]
    for grp, pfx in (("F2 혈통·육종가", "F2_"), ("F5 구간기록", "F5_")):
        got = [c for c in lost if c.startswith(pfx)]
        print(f"  {grp:<16} {len(got):>2}개")
    print(f"  {'F1 조교':<16} {len([c for c in lost if c in UNAVAILABLE_EXACT]):>2}개")
    print(f"  {'합계':<16} {len(lost):>2}개 제외\n")

    CASES = [
        ("전체 77피처",             full77, False),
        ("API156 유도 50피처",      api50,  False),
        ("전체 73피처 (실시간)",     full73, True),
        ("API156 유도 46피처 (실시간)", api46, True),
    ]
    out, ens = {}, {}
    print(f"{'구성':<30}{'피처':>5}{'top-1':>9}{'편차':>8}{'top-3':>9}   시드별")
    print("-" * 84)
    for label, cols, rt in CASES:
        sc, t1, t3 = [], [], []
        for s in SEEDS:
            p = run(tr, va, cols, s); sc.append(p)
            m = C.evaluate(va, p); t1.append(m["top1"]); t3.append(m["top3"])
        e = C.evaluate(va, np.mean(sc, axis=0))
        out[label] = dict(n=len(cols), top1=float(np.mean(t1)), std=float(np.std(t1)),
                          top3=float(np.mean(t3)), seeds=t1, ens_top1=float(e["top1"]),
                          realtime=rt)
        ens[label] = e["top1"]
        print(f"{label:<30}{len(cols):>5}{np.mean(t1):>8.2f}%{np.std(t1):>7.2f}%"
              f"{np.mean(t3):>8.2f}%   {' '.join(f'{v:.2f}' for v in t1)}")

    mkt = C.evaluate(va, C.market_scores(va))
    print("-" * 84)
    print(f"{'시장 (인기 1위마)':<30}{'—':>5}{mkt['top1']:>8.2f}%{'':>8}{mkt['top3']:>8.2f}%")
    print(f"\n표준오차 ±{mkt['se_top1']:.2f}%p · 그 2배({2*mkt['se_top1']:.2f}%p)를 넘어야 우열\n")

    print("=== 피처를 뺐을 때의 손실 ===")
    for a, b in (("전체 77피처", "API156 유도 50피처"),
                 ("전체 73피처 (실시간)", "API156 유도 46피처 (실시간)")):
        d = out[b]["top1"] - out[a]["top1"]
        verdict = "유의한 손실" if abs(d) > 2 * mkt["se_top1"] else "표준오차 안"
        print(f"  {a:<24} → {b:<28} {d:+.2f}%p  {verdict}")

    print("\n=== 시드 앙상블 (점수 평균) ===")
    for k, v in ens.items():
        print(f"  {k:<30}{v:>7.2f}%")

    out["_market"] = dict(top1=float(mkt["top1"]), top3=float(mkt["top3"]),
                          se=float(mkt["se_top1"]))
    out["_lost_features"] = lost
    io.open(Path(r"c:/Users/SSAFY/Desktop/주제선정/experiments/out/api156_only.json"),
            "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n총 {time.time()-t0:.0f}s · 저장 experiments/out/api156_only.json")


if __name__ == "__main__":
    main()
