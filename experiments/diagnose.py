# -*- coding: utf-8 -*-
"""
진단 1단계 — 데이터가 멀쩡한가.

모델이 안 나올 때 튜닝부터 손대면 안 된다. 피처가 깨져 있으면 어떤 설정으로도 안 나온다.
여기서 보는 것:
  1. 죽은 피처 — 상수, 전부 결측, 사실상 단일값
  2. 단변량 신호 — 각 피처가 **경주 내에서** 착순과 상관이 있나
     (전체 상관은 무의미하다. 경마는 경주 내 상대비교 문제다)
  3. 범주형 처리 — cat.codes 를 float 로 넘기면 LightGBM 이 순서형으로 오해한다
  4. 라벨 분포 — y_rel 이 1착을 충분히 강조하나

실행: python experiments/diagnose.py
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from scipy import stats
import schema_v2 as S

tr = pd.read_parquet(f"{S.MODEL_DIR}/train.parquet")
cols = [c for c in S.features() if c in tr.columns]
print(f"train {len(tr):,}행 · 피처 {len(cols)}개\n")

# ── 1. 죽은 피처 ────────────────────────────────────────────────────
print("1. 죽은 피처")
dead = []
for c in cols:
    s = tr[c]
    na = s.isna().mean()
    nu = s.nunique(dropna=True)
    if na > 0.98:
        dead.append((c, f"결측 {na*100:.0f}%")); continue
    if nu <= 1:
        dead.append((c, f"고유값 {nu}개")); continue
    if s.dtype.kind in "if":
        v = s.dropna()
        if len(v) and (v == v.mode().iloc[0]).mean() > 0.98:
            dead.append((c, f"단일값 {(v==v.mode().iloc[0]).mean()*100:.0f}%"))
print(f"   {len(dead)}개" + ("" if dead else " — 없음"))
for c, why in dead:
    print(f"     {c:28s} {why}")

# ── 2. 단변량 신호 — 경주 내 상관 ───────────────────────────────────
print("\n2. 경주 내 단변량 신호 (Spearman: 피처 순위 vs 착순 순위)")
print("   |rho| 이 클수록 그 피처만으로도 경주 내 우열을 가릴 수 있다는 뜻")
sig = []
g = tr.groupby("race_id", sort=False)
yr = g["y_ord"].rank()
for c in cols:
    s = tr[c]
    if s.dtype.kind not in "if" or s.notna().sum() < 5000:
        continue
    xr = s.groupby(tr["race_id"]).rank()
    m = xr.notna() & yr.notna()
    if m.sum() < 5000:
        continue
    rho = stats.spearmanr(xr[m], yr[m]).statistic
    sig.append((c, rho, s.notna().mean()))
sig.sort(key=lambda x: -abs(x[1]))
print(f"\n   {'피처':<28s} {'rho':>7s} {'충전':>7s}")
for c, r, f in sig[:16]:
    print(f"   {c:<28s} {r:>7.3f} {f*100:>6.0f}%")
print("   …")
weak = [x for x in sig if abs(x[1]) < 0.02]
print(f"\n   |rho| < 0.02 인 사실상 무신호 피처: {len(weak)}/{len(sig)}개")
for c, r, f in weak[:12]:
    print(f"     {c:28s} rho={r:+.3f} 충전 {f*100:.0f}%")

# ── 3. 범주형 처리 ──────────────────────────────────────────────────
print("\n3. 범주형")
for c in S.CATEGORICAL:
    if c in tr.columns:
        print(f"   {c:20s} dtype={str(tr[c].dtype):10s} 고유 {tr[c].nunique():>6,} "
              f"결측 {tr[c].isna().mean()*100:5.1f}%")
print("   ⚠ 학습 코드가 cat.codes → float 로 넘기면 LightGBM 이 '순서 있는 수'로 오해한다.")
print("     F2_sire_id 같은 고카디널리티에서 특히 해롭다. categorical_feature 로 넘겨야 한다.")

# ── 4. 라벨 ─────────────────────────────────────────────────────────
print("\n4. 라벨 y_rel = max(0, dusu − ord)")
d = tr[tr["X_dusu"] == 10]
vc = d["y_rel"].value_counts().sort_index(ascending=False)
print("   10두 경주에서 착순별 라벨:")
for rel, n in list(vc.items())[:5]:
    ordv = 10 - rel
    print(f"     {int(ordv)}착 → {int(rel)}점  ({n:,}행)")
print(f"   1착(9점) 대 2착(8점) 차이 = {1/9*100:.0f}%. "
      "Top-1 을 목표로 하는데 1·2착 구분이 거의 없다.")
print(f"   두수가 다르면 같은 착순의 점수도 달라진다: "
      f"8두 1착={8}점 vs 12두 1착={11}점 — 경주 간 비교가 어긋난다.")

print("\n" + "=" * 66)
print("요약")
print(f"  죽은 피처 {len(dead)}개 · 무신호 피처 {len(weak)}개 / {len(sig)}개")
if sig:
    print(f"  최강 단변량 신호: {sig[0][0]} rho={sig[0][1]:+.3f}")
