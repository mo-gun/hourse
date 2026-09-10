# -*- coding: utf-8 -*-
"""train 내부 시간분할 교차검증 — valid·test 를 건드리지 않고 검정력을 얻는다.

**왜 필요한가.** valid 1,254경주는 2026-09-10 하루에 약 25번 측정했다(계열 16종×3시드,
PCA 2종, H1/H2/H1+H2, 앙상블, Benter2 6종). 게다가 top-1 이 안 움직인 것을 보고 나서
logloss 를 보고했다 — 지표를 결과에 맞춰 고른 셈이다. 같은 자를 계속 쓰면 숫자가
좋아 보이는 쪽으로 흐른다.

`test` 802경주는 한 번만 열 수 있고, H1+H2 효과(logloss −0.0128, 경주별 sd 0.1325)를
80% 검정력으로 잡으려면 **경주 834개**가 필요해 **경계선(~78%)** 이다. 지금 하나에
쓰기 아깝다.

**그래서 train 안에서 시간을 접는다.** train 은 36,586경주(2010-01~2025-11) 다.
확장창(expanding window) 으로 K분할하면 **valid 의 약 20배 표본**을 얻고,
`valid`·`test`·`game` 을 전혀 건드리지 않는다.

    fold1 학습 → fold2 평가
    fold1+2   → fold3 평가
    fold1+2+3 → fold4 평가
    ...

미래를 절대 보지 않는다(확장창이라 학습 구간이 항상 평가 구간보다 과거다).
비교는 **같은 fold 안에서 쌍체**로 하므로 fold 마다 학습량·시대가 다른 것이 상쇄된다.

    uv run python -m arena.cv                              # 기준선 vs H1+H2
    uv run python -m arena.cv --arms lgb_lambdarank,lgb_margin,lgb_field,lgb_margin_field
    uv run python -m arena.cv --folds 6 --seeds 0,1
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

from pathlib import Path

from . import evaluate as E
from . import harness as H
from .models import FAMILIES, LABEL

CACHE = Path(__file__).resolve().parents[1] / "artifacts" / "cv"


def race_dates(df):
    """경주 단위 (race_id, rcDate) — 분할은 경주 단위여야 한다."""
    g = df.groupby("race_id", sort=False)["rcDate"].first()
    return g


def make_folds(df, k):
    """개최일 기준 k 등분. 경주가 fold 를 걸치지 않는다."""
    rd = race_dates(df)
    days = np.sort(rd.unique())
    edges = [days[int(len(days) * i / k)] for i in range(1, k)]
    fold_of_day = np.searchsorted(edges, days, side="right")
    day2fold = dict(zip(days, fold_of_day))
    return df["rcDate"].map(day2fold).to_numpy(), days, edges


def per_race_ll(ev, scores, T):
    q = pd.factorize(ev["race_id"], sort=False)[0]
    p = E.race_softmax(ev, T * E.race_z(ev, scores))
    w = ev["y_win"].to_numpy(bool)
    return -np.log(np.clip(p[w], 1e-15, 1)), q[w]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="lgb_lambdarank,lgb_margin_field")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--pop", action="store_true")
    a = ap.parse_args()
    arms = [x.strip() for x in a.arms.split(",")]
    seeds = tuple(int(x) for x in a.seeds.split(","))
    tr_all, _ = H.splits()
    feats = H.cols(a.pop)

    fold, days, edges = make_folds(tr_all, a.folds)
    print("train %d경주 %s~%s · %d분할"
          % (tr_all["race_id"].nunique(), int(days[0]), int(days[-1]), a.folds))
    for f in range(a.folds):
        m = fold == f
        print("  fold %d  %7d두 %6d경주  %s~%s"
              % (f, int(m.sum()), tr_all.loc[m, "race_id"].nunique(),
                 int(tr_all.loc[m, "rcDate"].min()), int(tr_all.loc[m, "rcDate"].max())))
    print("  ★ valid·test·game 을 전혀 쓰지 않는다\n")

    t0 = time.time()
    ll = {n: [] for n in arms}          # 경주별 logloss 를 fold 에 걸쳐 이어붙인다
    hit = {n: [] for n in arms}
    for f in range(1, a.folds):
        fit = tr_all[fold < f]
        ev = tr_all[fold == f]
        for name in arms:
            fn = FAMILIES[name]
            cache = CACHE / ("cv%d_%s_%s_%s.npy" % (a.folds, name, "77" if a.pop else "73",
                                                     "-".join(map(str, seeds)) + "_f%d" % f))
            if cache.exists():
                sc = np.load(cache)
            else:
                sc = np.mean([np.asarray(fn(fit, ev, feats, s), float) for s in seeds], axis=0)
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, sc)
            T = E.logloss_calibrated(ev, sc)[1]
            l, _ = per_race_ll(ev, sc, T)
            ll[name].append(l)
            hit[name].append(E.hit_expected(ev, sc))
            print("  fold %d  %-20s 학습 %6d경주 → 평가 %5d경주 · top-1 %5.2f · logloss %.4f  [%.0fs]"
                  % (f, name, fit["race_id"].nunique(), ev["race_id"].nunique(),
                     E.table_metrics(ev, sc)["top1"], l.mean(), time.time() - t0), flush=True)

    print("\n" + "=" * 88)
    print("종합 — fold 2..%d 를 이어붙인 평가 표본" % (a.folds - 1))
    print("=" * 88)
    base = arms[0]
    L = {n: np.concatenate(v) for n, v in ll.items()}
    Hh = {n: np.concatenate(v) for n, v in hit.items()}
    n_race = len(L[base])
    print("평가 경주 %d개 (valid 1,254 의 %.1f배)\n" % (n_race, n_race / 1254))
    print("%-28s%10s%10s%12s" % ("팔", "top-1", "logloss", ""))
    print("-" * 62)
    for n in arms:
        print("%-28s%9.2f%%%10.4f" % (LABEL[n], Hh[n].mean() * 100, L[n].mean()))

    print("\n%-28s%12s%12s%26s%12s" % ("기준선 대비 (쌍체)", "top-1", "logloss", "logloss 95% CI", "Wilcoxon p"))
    print("-" * 92)
    rng = np.random.default_rng(0)
    for n in arms[1:]:
        dh = (Hh[n] - Hh[base]).mean() * 100
        d = L[n] - L[base]
        bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p = stats.wilcoxon(L[base], L[n]).pvalue
        sig = "★" if not (lo <= 0 <= hi) else " "
        print("%-28s%+10.2f%%p%+11.4f   [%+.4f, %+.4f]%s%11.2e"
              % (LABEL[n], dh, d.mean(), lo, hi, sig, p))
    print("\n총 %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
