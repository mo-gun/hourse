# -*- coding: utf-8 -*-
"""계열 비교 실행 — 같은 데이터·같은 자로 한 표에 올린다.

    uv run python -m arena.run                       # 73피처, 시드 0 (선별용 1회)
    uv run python -m arena.run --seeds 0,1,2         # 시드 3개 평균±sd (판정용)
    uv run python -m arena.run --pop                 # 77피처(인기도 포함, 게임 리플레이)
    uv run python -m arena.run --families cat_querysoftmax,xgb_ndcg

예측은 `artifacts/pred/<family>_<77|73>_<seed>.npy` 로 저장된다 — 앙상블·2단계 결합이
다시 학습하지 않고 집어 쓴다(`arena.combine`).

규칙: valid 로만 비교한다. test 는 열지 않는다. game 은 하네스가 아예 안 내준다.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from . import evaluate as E
from . import harness as H
from .models import FAMILIES, LABEL

PRED = Path(__file__).resolve().parents[1] / "artifacts" / "pred"


def run(names, seeds, pop):
    tr, ev = H.splits()
    feats = H.cols(pop)
    tag = "77" if pop else "73"
    PRED.mkdir(parents=True, exist_ok=True)
    print(f"학습 {len(tr):,}두 {tr['race_id'].nunique():,}경주 / "
          f"평가 {len(ev):,}두 {ev['race_id'].nunique():,}경주 · {len(feats)}피처 [{tag}]")
    print(f"시드 {list(seeds)}\n")

    out = {}
    for name in names:
        fn = FAMILIES[name]
        got = []
        for sd in seeds:
            f = PRED / f"{name}_{tag}_{sd}.npy"
            if f.exists():
                s = np.load(f); mark = "(캐시)"
            else:
                t0 = time.time()
                s = np.asarray(fn(tr, ev, feats, sd), float)
                np.save(f, s); mark = f"({time.time()-t0:.0f}s)"
            got.append(s)
            m = E.table_metrics(ev, s)
            print(f"  {name:<18} seed {sd}  top-1 {m['top1']:5.2f}  "
                  f"top-3 {m['top3']:5.2f}  {mark}", flush=True)
        out[name] = got
    return tr, ev, out, tag


def report(ev, out, tag):
    print("\n" + "=" * 82)
    print(f"[{tag}피처] 시드 평균 ± sd — 적중률은 팀 채점기 그대로")
    print("=" * 82)
    print(f"{'계열':<34}{'top-1':>15}{'top-3':>15}{'logloss*':>10}{'T':>7}")
    print("-" * 89)
    rows = []
    for name, ss in out.items():
        t1 = np.array([E.table_metrics(ev, s)["top1"] for s in ss])
        t3 = np.array([E.table_metrics(ev, s)["top3"] for s in ss])
        mean = np.mean(ss, axis=0)
        ll, T = E.logloss_calibrated(ev, mean)
        sd1 = t1.std(ddof=1) if len(t1) > 1 else 0.0
        sd3 = t3.std(ddof=1) if len(t3) > 1 else 0.0
        print(f"{LABEL[name]:<34}{t1.mean():>9.2f} ±{sd1:<4.2f}{t3.mean():>9.2f} ±{sd3:<4.2f}"
              f"{ll:>10.4f}{T:>7.2f}")
        rows.append((name, t1.mean(), mean))
    mk = E.market_scores(ev)
    m = E.table_metrics(ev, mk)
    mll, mT = E.logloss_calibrated(ev, E.market_logit(ev))
    print("-" * 89)
    print(f"{'시장 (인기 1위마)':<34}{m['top1']:>9.2f}      {m['top3']:>9.2f}"
          f"      {mll:>10.4f}{mT:>7.2f}")
    print("  logloss* = 온도 보정 후(valid 2-fold 교차적합). 계열 간 비교는 이 열로 —")
    print("  보정 없이 재면 확률을 그대로 내놓는 계열이 '거의 균등'으로 잘못 나온다")
    base = E.table_metrics(ev, np.mean(out['lgb_lambdarank'], axis=0))["top1"] \
        if 'lgb_lambdarank' in out else None
    if base is not None:
        print(f"\n넘어야 할 선 = 기준선 {base:.2f} + 2×SE(0.85) = **{base + 1.7:.2f}**  "
              f"· 검출 한계 짝 기준 ±2.39%p")

    if 'lgb_lambdarank' in out:
        print("\n" + "=" * 82)
        print("쌍체 검정 — 시드평균 점수, 기준선(LightGBM lambdarank) 대비. 동률 1/k")
        print("=" * 82)
        a = np.mean(out['lgb_lambdarank'], axis=0)
        for name, _, mean in rows:
            if name == 'lgb_lambdarank':
                continue
            r = E.paired(ev, a, mean)
            flag = "" if (r['lo'] <= 0 <= r['hi']) else "  ★유의"
            print(f"{LABEL[name]:<34}{r['diff']:>+7.2f}%p  "
                  f"CI [{r['lo']:+6.2f}, {r['hi']:+6.2f}]  p={r['p']:.3f}{flag}")
        r = E.paired(ev, a, E.market_scores(ev))
        print(f"{'시장 − 기준선':<34}{r['diff']:>+7.2f}%p  "
              f"CI [{r['lo']:+6.2f}, {r['hi']:+6.2f}]  p={r['p']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--pop", action="store_true", help="77피처(인기도 포함)")
    a = ap.parse_args()
    names = [n.strip() for n in a.families.split(",") if n.strip()]
    bad = [n for n in names if n not in FAMILIES]
    if bad:
        raise SystemExit(f"모르는 계열 {bad}. 가능: {list(FAMILIES)}")
    seeds = tuple(int(s) for s in a.seeds.split(","))
    tr, ev, out, tag = run(names, seeds, a.pop)
    report(ev, out, tag)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
