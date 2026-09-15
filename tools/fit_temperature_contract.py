# -*- coding: utf-8 -*-
"""온도 T 를 상수 하나로 둘 수 있나 — 계약 3종 비교.

fit_temperature.py 에서 가중치마다 T* 가 달랐다 (0.031 ~ 0.076). 상수 하나를
박아 두는 현재 계약은 유저가 슬라이더를 극단으로 밀면 무너진다는 뜻이다.
세 가지 계약을 같은 자로 재고 고른다.

    A) 현행    p = softmax(raw × T)                       raw = Σ(w·ax)/100
    B) 경주내 표준화  z = (raw - mean_race) / std_race ;  p = softmax(z × T)
    C) 축 표준화      축마다 경주 내 표준화 후 가중합 ;   p = softmax(raw × T)

고르는 자는 **T* 의 흔들림**이다. 가중치를 바꿔도 T* 가 거의 안 변하는 계약이라야
상수 하나를 내려보낼 수 있다.

    python tools/fit_temperature_contract.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_temperature import (AXES, PRESETS, SPEC_T, IMPL_T, AXIS_FEATURES,   # noqa: E402
                             pct_within_race, race_index, nll, probs, ece, C)


def race_stat(df, v, q, nq, how):
    s = np.zeros(nq)
    n = np.bincount(q, minlength=nq).astype(float)
    np.add.at(s, q, v)
    mu = s / n
    d = v - mu[q]
    if how == "mean":
        return d
    s2 = np.zeros(nq)
    np.add.at(s2, q, d * d)
    return d / (np.sqrt(s2 / n)[q] + 1e-6)


def main():
    va = C.load("valid")
    q, nq = race_index(va), race_index(va).max() + 1
    win = va["y_win"].to_numpy(bool)
    y = va["y_win"].to_numpy(float)
    tr = C.load("train")
    tr_win = tr["y_win"].to_numpy(bool)

    direction = {}
    for ax in AXES:
        for f in AXIS_FEATURES[ax]:
            if f in tr.columns and pd.api.types.is_numeric_dtype(tr[f]):
                p = pct_within_race(tr, f)
                if np.nanstd(p) > 1e-6:
                    direction[f] = 1.0 if p[tr_win].mean() >= 50.0 else -1.0

    ax_raw, ax_std = {}, {}
    for ax in AXES:
        cols = [f for f in AXIS_FEATURES[ax] if f in direction]
        M = np.stack([pct_within_race(va, f) if direction[f] > 0
                      else 100.0 - pct_within_race(va, f) for f in cols], 1)
        ax_raw[ax] = M.mean(1)
        ax_std[ax] = race_stat(va, ax_raw[ax], q, nq, "z") * 25.0 + 50.0   # 축마다 같은 폭으로

    print("=" * 92)
    print("계약 3종 — 가중치를 바꿔도 T 가 버티나  (valid 1,254경주)")
    print("=" * 92)
    print("%-22s%10s%10s%10s%12s" % ("가중치", "A 현행", "B 경주표준화", "C 축표준화", "A logloss"))
    print("-" * 92)

    res = {k: [] for k in "ABC"}
    for name, w in PRESETS.items():
        line = [name]
        for key in "ABC":
            src = ax_std if key == "C" else ax_raw
            raw = sum(w[k] * src[k] for k in AXES) / 100.0
            if key == "B":
                raw = race_stat(va, raw, q, nq, "z")
            r = optimize.minimize_scalar(nll, bounds=(1e-3, 20.0), method="bounded",
                                         args=(raw, q, win, nq), options=dict(xatol=1e-6))
            res[key].append((float(r.x), float(r.fun)))
            line.append(float(r.x))
        print("%-22s%10.4f%10.4f%10.4f%12.4f"
              % (line[0], line[1], line[2], line[3], res["A"][-1][1]))

    print("-" * 92)
    print("%-22s%10s%10s%10s" % ("", "", "", ""))
    for key, label in (("A", "A 현행 softmax(raw×T)"),
                       ("B", "B 경주 내 표준화 후"),
                       ("C", "C 축 표준화 후")):
        T = np.array([t for t, _ in res[key]])
        ll = np.array([l for _, l in res[key]])
        spread = T.max() / T.min()
        print("  %-24s T* %.4f~%.4f  (최대/최소 %4.1f배)  평균 logloss %.4f"
              % (label, T.min(), T.max(), spread, ll.mean()))

    # 고정 T 를 썼을 때의 실제 손해 — 계약별로 '전 가중치 공통 T' 를 하나 뽑아 잰다
    print("")
    print("=" * 92)
    print("상수 하나를 박았을 때의 손해 (전 가중치 공통 T 를 최적화해서 고름)")
    print("=" * 92)
    for key, label in (("A", "A 현행"), ("B", "B 경주 내 표준화"), ("C", "C 축 표준화")):
        src = ax_std if key == "C" else ax_raw
        raws = []
        for w in PRESETS.values():
            r = sum(w[k] * src[k] for k in AXES) / 100.0
            raws.append(race_stat(va, r, q, nq, "z") if key == "B" else r)

        def total(t):
            return float(np.mean([nll(t, r, q, win, nq) for r in raws]))

        g = optimize.minimize_scalar(total, bounds=(1e-3, 20.0), method="bounded",
                                     options=dict(xatol=1e-6))
        Tg = float(g.x)
        worst = max(nll(Tg, r, q, win, nq) - res[key][i][1] for i, r in enumerate(raws))
        print("  %-16s 공통 T = %7.4f   평균 손해 %+.4f   최악 가중치 손해 %+.4f"
              % (label, Tg, total(Tg) - np.mean([l for _, l in res[key]]), worst))

    print("")
    print("참고 — 현행 계약(A)에서 명세 0.0445 / 구현 0.1 의 최악 손해")
    for t, lab in ((SPEC_T, "0.0445"), (IMPL_T, "0.1   ")):
        worst = max(nll(t, sum(w[k] * ax_raw[k] for k in AXES) / 100.0, q, win, nq)
                    - res["A"][i][1] for i, w in enumerate(PRESETS.values()))
        print("    T = %s → 최악 %+.4f" % (lab, worst))


if __name__ == "__main__":
    main()
