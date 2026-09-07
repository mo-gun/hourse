# -*- coding: utf-8 -*-
"""
A/B 진단 3단계 — 구조 분석.
  4) 피처 상관구조 · PCA (중복도 / 유효 차원)
  5) 이원 ANOVA (데이터셋 × 피처셋)
  6) PSI — 분포 이동 가설 검증

    python experiments/ab_diag_struct.py
"""
import sys, json, io
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
TEAM = Path(r"c:/Users/SSAFY/Desktop/말고리즘/S15P21A304")
sys.path.insert(0, str(TEAM / "docs" / "model"))
import common as C                                     # noqa: E402
HERE = Path(r"c:/Users/SSAFY/Desktop/주제선정")
OUT = HERE / "experiments" / "out"
R = {}

C.DATASET = HERE / "dataset" / "v2"
trB, va = C.load("train"), C.load("valid")
C.DATASET = HERE / "dataset" / "v2_before_2004"
trA = C.load("train")
cols73 = C.feature_cols(exclude_pop=True)
trA_e, va_e = C.encode(trA, va, cols73)
trB_e, _    = C.encode(trB, va, cols73)

# ── 4) 상관구조 · PCA ─────────────────────────────────────────────────
print("=" * 74); print("4) 상관구조 · PCA — 73피처가 서로 얼마나 중복인가"); print("=" * 74)
X = trB_e[cols73].astype(float)
num = [c for c in cols73 if X[c].notna().mean() > 0.5]
Xn = X[num].fillna(X[num].median())
corr = Xn.corr()
st = corr.stack()
st = st[st.index.get_level_values(0) < st.index.get_level_values(1)]   # 대각·중복 제외
pairs = st[st.abs() > 0.90].abs().sort_values(ascending=False).reset_index()
pairs.columns = ["a", "b", "r"]
print(f"\n결측 50% 미만인 피처 {len(num)}개로 계산")
print(f"|r| > 0.90 인 쌍: {len(pairs)}개")
for _, r in pairs.head(12).iterrows():
    print(f"  {r.a:<26}{r.b:<26} r={corr.loc[r.a, r.b]:+.3f}")
R["corr_pairs"] = [dict(a=r.a, b=r.b, r=float(corr.loc[r.a, r.b])) for _, r in pairs.iterrows()]

Z = StandardScaler().fit_transform(Xn)
p = PCA().fit(Z)
ev = np.cumsum(p.explained_variance_ratio_)
k90 = int(np.searchsorted(ev, 0.90) + 1); k95 = int(np.searchsorted(ev, 0.95) + 1)
print(f"\nPCA: 분산 90% 설명에 {k90}개 주성분, 95% 에 {k95}개 (원 피처 {len(num)}개)")
print("  상위 5개 주성분 설명력: " + " ".join(f"{v*100:.1f}%" for v in p.explained_variance_ratio_[:5]))
print(f"  → 유효 차원이 {len(num)} 이 아니라 {k90}~{k95} 수준. 피처를 더 넣어도")
print("    새 정보가 아니라 같은 축의 반복일 수 있다는 뜻이다.")
R["pca"] = dict(n_features=len(num), k90=k90, k95=k95,
                evr=[float(v) for v in p.explained_variance_ratio_[:15]],
                cum=[float(v) for v in ev[:15]])

# ── 5) 이원 ANOVA ─────────────────────────────────────────────────────
print("\n" + "=" * 74); print("5) 이원 ANOVA — top-1 변동을 무엇이 설명하는가"); print("=" * 74)
res = json.load(io.open(OUT / "ab_2004.json", encoding="utf-8"))
rows = []
for key, v in res.items():
    if key.startswith("_"): continue
    label, tag = key.split("|")
    arm = label.strip().split()[0]
    if arm not in ("A", "B"): continue
    for s, val in zip(("s1", "s2", "s3"), v["seeds"]):
        rows.append(dict(arm=arm, feat=tag, seed=s, top1=val))
d = pd.DataFrame(rows)
gm = d.top1.mean(); n = len(d)
ss_t = ((d.top1 - gm) ** 2).sum()
ss_a = sum(len(g) * (g.top1.mean() - gm) ** 2 for _, g in d.groupby("arm"))
ss_b = sum(len(g) * (g.top1.mean() - gm) ** 2 for _, g in d.groupby("feat"))
cell = d.groupby(["arm", "feat"]).top1.agg(["mean", "size"])
ss_cells = sum(r["size"] * (r["mean"] - gm) ** 2 for _, r in cell.iterrows())
ss_ab = ss_cells - ss_a - ss_b
ss_e = ss_t - ss_cells
df_a = df_b = df_ab = 1; df_e = n - 4
ms_e = ss_e / df_e
print(f"\n{'요인':<22}{'제곱합':>10}{'자유도':>6}{'F':>9}{'p':>9}{'설명비율':>9}")
for nm, ss, df_ in (("데이터셋 (A/B)", ss_a, df_a), ("피처셋 (77/73)", ss_b, df_b),
                    ("상호작용", ss_ab, df_ab)):
    F = (ss / df_) / ms_e; pv = 1 - stats.f.cdf(F, df_, df_e)
    print(f"{nm:<22}{ss:>10.3f}{df_:>6}{F:>9.2f}{pv:>9.4f}{ss/ss_t*100:>8.1f}%")
    R.setdefault("anova", {})[nm] = dict(ss=float(ss), F=float(F), p=float(pv),
                                         share=float(ss/ss_t*100))
print(f"{'잔차 (시드)':<22}{ss_e:>10.3f}{df_e:>6}{'':>9}{'':>9}{ss_e/ss_t*100:>8.1f}%")
R["anova"]["잔차(시드)"] = dict(ss=float(ss_e), share=float(ss_e/ss_t*100))

# ── 6) PSI ────────────────────────────────────────────────────────────
print("\n" + "=" * 74); print("6) PSI — train 이 valid 와 얼마나 다른가 (분포 이동 가설)"); print("=" * 74)
def psi(exp, act, bins=10):
    e = pd.to_numeric(exp, errors="coerce").dropna()
    a = pd.to_numeric(act, errors="coerce").dropna()
    if len(e) < 100 or len(a) < 100 or e.nunique() < 5: return np.nan
    q = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(q) < 3: return np.nan
    q[0], q[-1] = -np.inf, np.inf
    pe = np.histogram(e, q)[0] / len(e); pa = np.histogram(a, q)[0] / len(a)
    pe, pa = np.clip(pe, 1e-6, None), np.clip(pa, 1e-6, None)
    return float(((pa - pe) * np.log(pa / pe)).sum())

added = trB[pd.to_numeric(trB.rcDate, errors="coerce") // 10000 < 2010]
watch = [c for c in ("X_rating", "F1_tr_sessions_28d", "F2_ebv_prize", "F5_g3f_time",
                     "F1_layoff_days", "F1_win_rate_life", "F1_ord_avg3",
                     "F3_jk_win_rate_life", "F4_dist_gap") if c in trB.columns]
print(f"\n{'피처':<24}{'A train':>10}{'B train':>10}{'추가분만':>10}   판정")
psis = {}
for c in watch:
    pa_, pb_, pn_ = psi(trA[c], va[c]), psi(trB[c], va[c]), psi(added[c], va[c])
    flag = "" if not (pb_ == pb_) else ("심각" if pb_ > 0.25 else "주의" if pb_ > 0.10 else "안정")
    print(f"{c:<24}{pa_:>10.3f}{pb_:>10.3f}{pn_:>10.3f}   {flag}")
    psis[c] = dict(A=pa_, B=pb_, added=pn_)
R["psi"] = psis
print("\n  PSI 해석: 0.10 미만 안정 · 0.10~0.25 주의 · 0.25 초과 심각")
print(f"  A 평균 {np.nanmean([v['A'] for v in psis.values()]):.3f} → "
      f"B 평균 {np.nanmean([v['B'] for v in psis.values()]):.3f} · "
      f"추가분만 {np.nanmean([v['added'] for v in psis.values()]):.3f}")

io.open(OUT / "ab_struct_result.json", "w", encoding="utf-8").write(
    json.dumps(R, ensure_ascii=False, indent=1))
print("\n저장 experiments/out/ab_struct_result.json")
