# -*- coding: utf-8 -*-
"""결합 단계 — 저장된 예측을 집어 앙상블·Benter 2단계를 잰다 (재학습 없음).

    uv run python -m arena.run --seeds 0,1,2            # 먼저 계열을 돌려 예측을 남긴다
    uv run python -m arena.run_combine --seeds 0,1,2

두 갈래를 따로 본다 — 갈래를 섞으면 "시장을 쓰나 안 쓰나"가 흐려진다.

  **73 갈래 (주말 실시간)** — 배당을 못 쓴다. 여기서 할 수 있는 결합은 계열 앙상블뿐.
    레포 최고 기록이 단일 모델이 아니라 LGB+S3 앙상블(35.3)이었으므로 1차 표적이다.

  **77 갈래 (게임 리플레이)** — 배당이 확정돼 있다. Benter(1994) 2단계 =
    73피처 펀더멘털 점수를 시장 확률과 조건부 로짓으로 결합. 레포는 이 결합을
    AxisRanker 기저로만 재봤다(적중률 −0.16%p). 기저를 GBDT 로 바꿔 다시 잰다.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np

from . import combine as K
from . import evaluate as E
from . import harness as H
from .models import LABEL

PRED = Path(__file__).resolve().parents[1] / "artifacts" / "pred"


def load(tag, seeds):
    out = {}
    for f in sorted(PRED.glob(f"*_{tag}_*.npy")):
        name = f.stem.rsplit(f"_{tag}_", 1)[0]
        sd = int(f.stem.rsplit("_", 1)[1])
        if sd in seeds:
            out.setdefault(name, []).append(np.load(f))
    return {k: np.mean(v, axis=0) for k, v in out.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--top", type=int, default=4, help="앙상블 후보 상위 몇 계열")
    a = ap.parse_args()
    seeds = {int(s) for s in a.seeds.split(",")}
    tr, ev = H.splits()
    P73 = load("73", seeds)
    P77 = load("77", seeds)
    if not P73:
        raise SystemExit("73 예측이 없다. 먼저 `python -m arena.run --seeds ...` 를 돌려라.")

    rank = sorted(P73, key=lambda n: -E.table_metrics(ev, P73[n])["top1"])
    base73 = P73.get("lgb_lambdarank")

    # ── 진단: 계열끼리 얼마나 다른가 ────────────────────────────────
    # 앙상블이 이득을 내려면 **틀리는 곳이 달라야** 한다. 레포 최고 기록(LGB+S3 35.3)은
    # GBDT 와 신경망을 섞은 것이었다. 트리 구현끼리는 상관이 높을 것으로 보이는데,
    # 그러면 계열을 늘려도 앙상블 이득이 안 나온다 — 그 예측을 먼저 확인한다.
    names = sorted(P73, key=lambda n: -E.table_metrics(ev, P73[n])["top1"])
    Z = np.stack([K.race_z(ev, P73[n]) for n in names])
    Cm = np.corrcoef(Z)
    print("=" * 84)
    print("진단 — 계열 간 예측 상관 (경주 내 z-score 기준)")
    print("=" * 84)
    print("       " + " ".join(f"{i:>5}" for i in range(len(names))))
    for i, n in enumerate(names):
        print(f"{i:>2} {n[:16]:<16}" + " ".join(f"{Cm[i, j]:>5.2f}" for j in range(len(names))))
    off = Cm[np.triu_indices(len(names), 1)]
    print("")
    print(f"비대각 상관: 중앙값 {np.median(off):.3f} · 최소 {off.min():.3f} · 최대 {off.max():.3f}")
    print("  상관이 높으면 같은 경주에서 같이 틀린다 = 앙상블로 얻을 게 없다.")
    print("  ★ 실측: 가장 탈상관되는 건 '모델 계열'(mlp_pl 0.87~0.95)이 아니라")
    print("    '타깃'(lgb_speedfig 0.72~0.85)이다. 구조를 바꾸는 것보다 무엇을 맞히게")
    print("    할지를 바꾸는 쪽이 다르게 틀린다 — 앙상블 재료는 그쪽에서 찾을 것.")
    print("")
    print("=" * 84)
    print("73 갈래 (주말 실시간) — 계열 앙상블. 경주 내 z-score 평균")
    print("=" * 84)
    print(E.header(38))
    for n in rank:
        print(E.fmt_row(LABEL[n], ev, P73[n], 38))
    print("-" * 71)

    cands, best = rank[:a.top], None
    for k in (2, 3, min(4, len(cands))):
        for combo in combinations(cands, k):
            s = K.ens_z(ev, [P73[c] for c in combo])
            m = E.table_metrics(ev, s)
            if best is None or m["top1"] > best[1]:
                best = (combo, m["top1"], s)
    allz = K.ens_z(ev, [P73[n] for n in rank])
    print(E.fmt_row(f"앙상블 전 계열 {len(rank)}개", ev, allz, 38))
    if best:
        print(E.fmt_row("앙상블 " + "+".join(best[0]), ev, best[2], 38))
    print("-" * 71)
    if base73 is not None:
        b = E.table_metrics(ev, base73)["top1"]
        print(f"기준선 {b:.2f} · 넘어야 할 선 **{b + 1.7:.2f}**")
        print("\n쌍체 (기준선 대비, 동률 1/k):")
        for nm, s in [("앙상블 전 계열", allz)] + ([("앙상블 " + "+".join(best[0]), best[2])] if best else []):
            r = E.paired(ev, base73, s)
            ci_excl = not (r["lo"] <= 0 <= r["hi"])
            flag = ("  ★유의" if (ci_excl and r["p"] < 0.05)
                    else "  (경계)" if (ci_excl or r["p"] < 0.05) else "")
            print(f"  {nm:<34}{r['diff']:>+7.2f}%p  CI [{r['lo']:+6.2f}, {r['hi']:+6.2f}]  p={r['p']:.3f}{flag}")

    print("\n" + "=" * 84)
    print("77 갈래 (게임 리플레이) — Benter 2단계: 73 펀더멘털 + log(시장확률), valid 2-fold 교차적합")
    print("=" * 84)
    # 시장 행은 **log(내재확률)** 을 점수로 쓴다. -mkt_rank 는 순위라서 top-1 은 같지만
    # logloss 가 척도 의미를 잃는다(내 장부 실수 기록: 확률/순위를 점수로 넣으면 softmax
    # 가 한 번 더 걸린다). top-1 픽은 단조변환이라 어느 쪽이든 동일하다.
    mk = E.market_logit(ev)
    print(E.header(38))
    print(E.fmt_row("시장 (인기 1위마)", ev, mk, 38))
    if P77:
        for n in sorted(P77, key=lambda n: -E.table_metrics(ev, P77[n])["top1"]):
            print(E.fmt_row(LABEL[n] + " [77]", ev, P77[n], 38))
    print("-" * 71)
    rows = []
    for n in rank[:a.top]:
        s, betas = K.benter2(ev, P73[n])
        print(E.fmt_row(f"Benter2({n})", ev, s, 38))
        rows.append((n, s, betas))
    s_ens, betas = K.benter2(ev, allz)
    print(E.fmt_row("Benter2(앙상블 전 계열)", ev, s_ens, 38))
    rows.append(("앙상블", s_ens, betas))
    print("-" * 71)
    print("계수 (a=log시장확률, b=펀더멘털 z) — fold 별:")
    for n, _, bs in rows:
        print(f"  {n:<22}" + " · ".join(f"a={b[0]:+.3f} b={b[1]:+.3f}" for b in bs))
    print("\n쌍체 (시장 대비, 동률 1/k):")
    for n, s, _ in rows:
        r = E.paired(ev, mk, s)
        ci_excl = not (r["lo"] <= 0 <= r["hi"])
        # 부트스트랩 CI 와 McNemar 가 엇갈릴 수 있다(다른 통계량이다). 둘 다 나올 때만
        # 유의로 읽고, 한쪽만이면 '경계'로 적는다 — 한쪽만 보고 단정하면 안 된다.
        flag = ("  ★유의" if (ci_excl and r["p"] < 0.05)
                else "  (경계)" if (ci_excl or r["p"] < 0.05) else "")
        print(f"  Benter2({n}){'':<{max(0, 22-len(n))}}{r['diff']:>+7.2f}%p  "
              f"CI [{r['lo']:+6.2f}, {r['hi']:+6.2f}]  p={r['p']:.3f}{flag}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
