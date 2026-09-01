# -*- coding: utf-8 -*-
"""
보조 데이터 조인 — 조교(F1) · 장제(F1) · 혈통(F2).

build_dataset.py 가 import 해서 쓴다. 여기 있는 것도 전부 as-of 다:
조교·장제는 **경주일 이전** 기록만 창(window)에 넣고,
혈통 자마 성적은 롤링 루프 안에서 emit→update 순서로 누적한다.

혈통 커버리지 주의
  마필종합 API 가 벌크 조회를 막아놔서 말 단위로 1콜씩 받아야 한다(개발계정 3,000콜/일).
  현재 상태에서 train 커버리지가 test 보다 크게 낮으면 F2 는 학습/평가 분포가 어긋난다.
  build_dataset 이 빌드 때마다 구간별 커버리지를 출력하니 그 숫자를 보고 판단할 것.
"""
import os, sys, bisect
from collections import defaultdict

import pandas as pd
import numpy as np

TRAIN_CSV = "data/raw/aux_train.csv"
SHOE_CSV = "data/raw/aux_shoe.csv"
PED_CSV = "data/raw/aux_pedigree.csv"

W28 = 28


def _to_ordinal(s):
    """'20260810' 과 '2026-08-10' 두 형식을 모두 받는다.

    조교 API 는 ISO('2026-08-10'), 원장은 YYYYMMDD 라 형식이 섞인다.
    숫자 변환만 하면 ISO 쪽이 통째로 NaN 이 되어 조교 피처가 조용히 전부 결측이 된다.
    """
    d = pd.to_datetime(s.astype(str).str.strip(), format="mixed",
                       errors="coerce")
    out = np.full(len(s), np.nan)
    ok = d.notna().to_numpy()
    out[ok] = d[ok].map(lambda x: x.toordinal()).to_numpy()
    return out


# ══════════════════════════════════════════════════════════════════════
def load_train_index():
    """(마번 → 정렬된 [일자, 횟수, 분, 수영]) 인덱스."""
    if not os.path.exists(TRAIN_CSV):
        print("  조교 파일 없음 — F1_tr_* 는 결측 유지")
        return None
    t = pd.read_csv(TRAIN_CSV, dtype={"hrNo": str}, encoding="utf-8-sig",
                    low_memory=False)
    t["d"] = _to_ordinal(t["trDate"])
    t = t[t["d"].notna()]
    for c in ("n_sessions", "minutes", "swim_n", "swim_laps", "jk_ridden", "entered"):
        if c not in t.columns:
            t[c] = 0
        t[c] = pd.to_numeric(t[c], errors="coerce").fillna(0)
    t = t.sort_values(["hrNo", "d"])
    idx = {}
    for hn, g in t.groupby("hrNo", sort=False):
        idx[hn] = (g["d"].to_numpy(),
                   g["n_sessions"].to_numpy(),
                   g["minutes"].to_numpy(),
                   g["swim_n"].to_numpy())
    print(f"  조교 인덱스 {len(idx):,}두 / {len(t):,}일")
    return idx


def train_features(idx, hrNo, day):
    """경주일 이전 28일 / 그 이전 28일 창."""
    if idx is None:
        return (np.nan,) * 5
    e = idx.get(hrNo)
    if e is None:
        return (np.nan,) * 5
    d, n, m, s = e
    hi = bisect.bisect_left(d, day)          # 경주일 당일은 제외 (< day)
    if hi == 0:
        return (0.0, 0.0, np.nan, 0.0, np.nan)
    lo1 = bisect.bisect_left(d, day - W28)
    lo2 = bisect.bisect_left(d, day - 2 * W28)
    n1 = float(n[lo1:hi].sum()); m1 = float(m[lo1:hi].sum())
    sw = float(s[lo1:hi].sum())
    n2 = float(n[lo2:lo1].sum())
    since = float(day - d[hi - 1])
    chg = (n1 / n2) if n2 > 0 else np.nan
    return (n1, m1, since, sw, chg)


# ══════════════════════════════════════════════════════════════════════
def load_shoe_index():
    if not os.path.exists(SHOE_CSV):
        print("  장제 파일 없음 — F1_shoe_* 는 결측 유지")
        return None
    s = pd.read_csv(SHOE_CSV, dtype={"hrNo": str}, encoding="utf-8-sig",
                    low_memory=False)
    s["d"] = _to_ordinal(s["shoeDate"])
    s = s[s["d"].notna()].sort_values(["hrNo", "d"])
    idx = {}
    for hn, g in s.groupby("hrNo", sort=False):
        idx[hn] = (g["d"].to_numpy(), g["codeName2"].astype(str).to_numpy())
    print(f"  장제 인덱스 {len(idx):,}두 / {len(s):,}건")
    return idx


def shoe_features(idx, hrNo, day):
    """(최종 장제 경과일, 형태변경 여부, 현재 형태)"""
    if idx is None:
        return (np.nan, np.nan, None)
    e = idx.get(hrNo)
    if e is None:
        return (np.nan, np.nan, None)
    d, ty = e
    hi = bisect.bisect_left(d, day)
    if hi == 0:
        return (np.nan, np.nan, None)
    since = float(day - d[hi - 1])
    cur = ty[hi - 1]
    chg = np.nan
    if hi >= 2:
        chg = 1.0 if ty[hi - 2] != cur else 0.0
    return (since, chg, cur)


# ══════════════════════════════════════════════════════════════════════
def load_pedigree():
    """마번 → (부마, 모마, 외조부). 없는 말은 키 자체가 없다."""
    if not os.path.exists(PED_CSV):
        print("  혈통 파일 없음 — F2_* 는 결측 유지")
        return {}
    p = pd.read_csv(PED_CSV, dtype=str, encoding="utf-8-sig").fillna("")
    out = {}
    for r in p.itertuples(index=False):
        hn = str(r.hrNo).strip()
        if not hn:
            continue
        fa = str(getattr(r, "faHrNo", "")).strip()
        mo = str(getattr(r, "moHrNo", "")).strip()
        ds = str(getattr(r, "damsireNo", "")).strip()
        out[hn] = (fa or None, mo or None, ds or None)
    print(f"  혈통 {len(out):,}두")
    return out


class SireState:
    """부마/모마/외조부별 자마 성적 as-of 누적기.

    emit 시점에는 '이 경주 이전'까지만 반영돼 있고,
    update 는 행 처리 끝에서 호출된다. 롤링 루프와 같은 규약.
    """

    def __init__(self):
        self.n = defaultdict(int)          # 자마 출주수
        self.w = defaultdict(int)          # 자마 1착수
        self.spd = defaultdict(float)      # 스피드지수 합
        self.spd_n = defaultdict(int)
        self.wdist = defaultdict(float)    # 우승 거리 합
        self.wdist_n = defaultdict(int)
        self.dn = defaultdict(int)         # (부마, 거리대) 출주
        self.dw = defaultdict(int)
        self.wn = defaultdict(int)         # (부마, 불량주로) 출주
        self.ww = defaultdict(int)
        self.prog = defaultdict(set)       # 자마 두수

    def emit(self, sire, band, wet):
        if not sire:
            return (np.nan,) * 7
        n = self.n[sire]
        if n == 0:
            return (0, np.nan, np.nan, np.nan, np.nan, np.nan, len(self.prog[sire]))
        wr = self.w[sire] / n
        sp = self.spd[sire] / self.spd_n[sire] if self.spd_n[sire] else np.nan
        awd = self.wdist[sire] / self.wdist_n[sire] if self.wdist_n[sire] else np.nan
        dn = self.dn[(sire, band)]
        df_ = self.dw[(sire, band)] / dn if dn else np.nan
        wn = self.wn[sire] if wet else 0
        wf = (self.ww[sire] / self.wn[sire]) if self.wn[sire] else np.nan
        return (n, wr, sp, df_, wf, awd, len(self.prog[sire]))

    def update(self, sire, hrNo, band, wet, win, spd, dist):
        if not sire:
            return
        self.n[sire] += 1
        self.w[sire] += win
        self.prog[sire].add(hrNo)
        if spd == spd:
            self.spd[sire] += spd; self.spd_n[sire] += 1
        if win and dist == dist:
            self.wdist[sire] += dist; self.wdist_n[sire] += 1
        self.dn[(sire, band)] += 1
        self.dw[(sire, band)] += win
        if wet:
            self.wn[sire] += 1
            self.ww[sire] += win


F2_EMIT_COLS = ["F2_sire_starts", "F2_sire_win_rate", "F2_sire_speed_avg",
                "F2_sire_dist_fit", "F2_sire_wet_fit", "F2_sire_avg_win_dist",
                "F2_sire_prog_n"]


def report_coverage(df, ped):
    """구간별 혈통 커버리지 — train/test 가 크게 어긋나면 F2 는 신뢰할 수 없다."""
    have = df["hrNo"].astype(str).isin(ped.keys())
    print("  혈통 커버리지 (구간별 행 비율)")
    rows = []
    for sp in ("train", "valid", "test"):
        m = df["split"] == sp
        if m.any():
            c = have[m].mean() * 100
            rows.append((sp, c))
            print(f"    {sp:6s} {c:5.1f}%")
    if len(rows) >= 2:
        lo = min(c for _, c in rows)
        hi = max(c for _, c in rows)
        if hi - lo > 25:
            print(f"    ⚠ 구간 간 격차 {hi-lo:.0f}%p — F2 는 학습/평가 분포가 어긋난다.")
            print("       tools/backfill_aux.py pedigree --topN 으로 과거마를 더 채우거나,")
            print("       F2 담당은 커버리지가 높은 구간으로 학습 범위를 좁힐 것.")
