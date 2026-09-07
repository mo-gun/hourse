# -*- coding: utf-8 -*-
"""
A/B 진단 2단계 — 저장된 예측으로 쌍체 검정 + 원인 분석. 재학습 없음.

  1) McNemar 쌍체 검정 (A vs B, 같은 1,254경주)
  2) 하위군 분해 — 어디서 이기고 어디서 지는가
  3) 피처 기여도 이동 — 모델이 무엇에 기대게 됐는가
  4) 상관구조 · PCA — 피처가 서로 중복인가
  5) 이원 ANOVA — 데이터셋 / 피처셋 / 상호작용
  6) PSI — 분포 이동 가설 검증

    python experiments/ab_diag_analyze.py
"""
import sys, json, io
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(r"c:/Users/SSAFY/Desktop/주제선정")
OUT = HERE / "experiments" / "out"
SEEDS = (20260901, 20260902, 20260903)

Z = np.load(OUT / "ab_diag.npz")
META = json.load(io.open(OUT / "ab_diag_meta.json", encoding="utf-8"))
V = pd.read_parquet(OUT / "valid_meta.parquet").reset_index(drop=True)
rid = V["race_id"].to_numpy()
win = V["y_win"].to_numpy()
RESULT = {}


def picks(score):
    """경주별 최고점 말의 인덱스 (동점이면 첫 번째 — common.evaluate 와 동일)."""
    d = pd.DataFrame({"r": rid, "s": np.asarray(score, float)})
    return d.groupby("r", sort=False)["s"].idxmax().to_numpy()


def hits(score):
    """경주 단위 적중 벡터 (길이 = 경주 수)."""
    return win[picks(score)]


def mean_score(arm, tag):
    return np.mean([Z[f"{arm}_{tag}_{s}"] for s in SEEDS], axis=0)


race_ids = pd.Index(pd.unique(rid))
print(f"valid {len(race_ids):,}경주 · {len(V):,}두")
print(f"train  A {META['A']['n_train']:,}행 / B {META['B']['n_train']:,}행\n")

# ── 1) McNemar ────────────────────────────────────────────────────────
print("=" * 74)
print("1) McNemar 쌍체 검정 — 같은 경주를 A 와 B 가 각각 맞혔는가")
print("=" * 74)
mcn = {}
for tag in ("77", "73"):
    ha, hb = hits(mean_score("A", tag)), hits(mean_score("B", tag))
    n11 = int(((ha == 1) & (hb == 1)).sum()); n00 = int(((ha == 0) & (hb == 0)).sum())
    n10 = int(((ha == 1) & (hb == 0)).sum()); n01 = int(((ha == 0) & (hb == 1)).sum())
    nd = n10 + n01
    p = stats.binomtest(n01, nd, 0.5).pvalue if nd else 1.0
    diff = (hb.mean() - ha.mean()) * 100
    se_paired = np.sqrt(nd) / len(ha) * 100 if nd else 0.0
    mcn[tag] = dict(n11=n11, n00=n00, n10=n10, n01=n01, p=float(p),
                    diff=float(diff), se_paired=float(se_paired),
                    a=float(ha.mean()*100), b=float(hb.mean()*100))
    print(f"\n[{tag}피처]  A {ha.mean()*100:.2f}%  →  B {hb.mean()*100:.2f}%   ({diff:+.2f}%p)")
    print(f"  둘 다 적중 {n11:>4}    둘 다 실패 {n00:>4}")
    print(f"  A만 적중  {n10:>4}    B만 적중  {n01:>4}   ← 불일치 {nd}경주만이 정보를 가진다")
    print(f"  쌍체 표준오차 ±{se_paired:.2f}%p  (절대 표준오차 ±1.38%p 보다 훨씬 작다)")
    print(f"  McNemar 정확검정 p = {p:.4f}   {'유의 (p<0.05)' if p < 0.05 else '유의하지 않음'}")
RESULT["mcnemar"] = mcn

# 시드별 재현성
print("\n[시드별 McNemar p]")
for tag in ("77", "73"):
    ps = []
    for s in SEEDS:
        ha, hb = hits(Z[f"A_{tag}_{s}"]), hits(Z[f"B_{tag}_{s}"])
        n10 = int(((ha == 1) & (hb == 0)).sum()); n01 = int(((ha == 0) & (hb == 1)).sum())
        ps.append(stats.binomtest(n01, n10+n01, 0.5).pvalue if n10+n01 else 1.0)
    print(f"  {tag}피처  " + "  ".join(f"{p:.3f}" for p in ps))
    RESULT.setdefault("mcnemar_seeds", {})[tag] = [float(x) for x in ps]

# ── 2) 하위군 분해 ────────────────────────────────────────────────────
print("\n" + "=" * 74)
print("2) 하위군 분해 — B 가 어디서 이기고 어디서 지는가 (73피처 기준)")
print("=" * 74)
rv = V.groupby("race_id", sort=False).first().reindex(race_ids)
ha, hb = hits(mean_score("A", "73")), hits(mean_score("B", "73"))
rv = rv.assign(A=ha, B=hb, d=hb.astype(int) - ha.astype(int))
sub = {}
def show(name, key):
    g = rv.groupby(key, observed=True).agg(n=("A", "size"), A=("A", "mean"), B=("B", "mean"))
    g = g[g["n"] >= 40].sort_values("n", ascending=False)
    print(f"\n[{name}]")
    print(f"  {'구간':<14}{'경주':>6}{'A':>8}{'B':>8}{'차이':>8}")
    for k, r in g.iterrows():
        print(f"  {str(k):<14}{int(r['n']):>6}{r['A']*100:>7.1f}%{r['B']*100:>7.1f}%"
              f"{(r['B']-r['A'])*100:>+7.1f}%p")
    sub[name] = {str(k): dict(n=int(r["n"]), A=float(r["A"]*100), B=float(r["B"]*100))
                 for k, r in g.iterrows()}

rv["_dist"] = pd.cut(rv["X_rcDist"], [0, 1100, 1300, 1600, 9999],
                     labels=["~1100m", "1200~1300m", "1400~1600m", "1700m+"])
rv["_dusu"] = pd.cut(rv["X_dusu"], [0, 8, 10, 12, 99],
                     labels=["~8두", "9~10두", "11~12두", "13두+"])
rv["_rating"] = np.where(rv["X_rating"].notna(), "rating 있음", "rating 없음")
show("거리", "_dist"); show("출주두수", "_dusu")
show("레이팅 유무", "_rating"); show("등급", "X_grade")
RESULT["subgroup_73"] = sub

# ── 3) 피처 기여도 이동 ───────────────────────────────────────────────
print("\n" + "=" * 74)
print("3) 피처 기여도(gain) 이동 — 모델이 무엇에 기대게 됐는가 (73피처)")
print("=" * 74)
cols = META["cols_73"]
def gain(arm):
    g = np.mean([META["importance"][f"{arm}_73_{s}"] for s in SEEDS], axis=0)
    return pd.Series(g / g.sum() * 100, index=cols)
ga, gb = gain("A"), gain("B")
sh = pd.DataFrame({"A": ga, "B": gb, "diff": gb - ga}).sort_values("diff")
print(f"\n{'피처':<26}{'A':>8}{'B':>8}{'변화':>9}")
print("  ↓ B 에서 덜 쓰게 된 것")
for c, r in sh.head(7).iterrows():
    print(f"  {c:<24}{r['A']:>7.2f}%{r['B']:>7.2f}%{r['diff']:>+8.2f}%p")
print("  ↑ B 에서 더 쓰게 된 것")
for c, r in sh.tail(7).iloc[::-1].iterrows():
    print(f"  {c:<24}{r['A']:>7.2f}%{r['B']:>7.2f}%{r['diff']:>+8.2f}%p")
RESULT["gain_shift_73"] = {c: dict(A=float(r["A"]), B=float(r["B"]), diff=float(r["diff"]))
                           for c, r in sh.iterrows()}
io.open(OUT / "ab_diag_result.json", "w", encoding="utf-8").write(
    json.dumps(RESULT, ensure_ascii=False, indent=1))
print("\n저장 experiments/out/ab_diag_result.json")
