# -*- coding: utf-8 -*-
"""슬라이더로 **실제 도달 가능한** 가중치만 놓고 T 를 고른다.

앞 두 스크립트는 '한 축 100' 을 극단 사례로 넣었는데, 그건 화면에서 만들 수 없는
값이다 — feature_group.max_weight = 40 (front/src/mock/fixtures.js:107, schema.sql:91).
못 만드는 상황에 대비해 계약을 고르면 멀쩡한 계약을 버리게 된다.

그래서 제약(각 축 0~40, 합 100)을 만족하는 가중치를 무작위로 400개 뽑아
    · T* 가 얼마나 흔들리나
    · 상수 하나를 박으면 최악 몇 경주가 얼마나 손해인가
를 계약 A(현행)·B(경주 내 표준화) 두 가지로 잰다.

    python tools/fit_temperature_reachable.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_temperature import (AXES, PRESETS, SPEC_T, IMPL_T, AXIS_FEATURES,   # noqa: E402
                             pct_within_race, race_index, nll, C)
from fit_temperature_contract import race_stat                                # noqa: E402

MAXW, TOTAL, N_SAMPLE = 40.0, 100.0, 400


def sample_weights(rng, n):
    """각 축 0~40, 합 100 인 가중치. 거절 샘플링 — 제약이 좁아 이게 가장 정직하다."""
    out = []
    while len(out) < n:
        w = rng.dirichlet(np.ones(6)) * TOTAL
        if w.max() <= MAXW:
            out.append(w)
    return np.array(out)


def main():
    tr, va = C.load("train"), C.load("valid")
    q = race_index(va)
    nq = q.max() + 1
    win = va["y_win"].to_numpy(bool)
    tr_win = tr["y_win"].to_numpy(bool)

    direction = {}
    for ax in AXES:
        for f in AXIS_FEATURES[ax]:
            if f in tr.columns and pd.api.types.is_numeric_dtype(tr[f]):
                p = pct_within_race(tr, f)
                if np.nanstd(p) > 1e-6:
                    direction[f] = 1.0 if p[tr_win].mean() >= 50.0 else -1.0

    A = np.stack([np.stack([pct_within_race(va, f) if direction[f] > 0
                            else 100.0 - pct_within_race(va, f)
                            for f in AXIS_FEATURES[ax] if f in direction], 1).mean(1)
                  for ax in AXES], 1)                       # (n, 6) 축 점수 0~100

    rng = np.random.default_rng(20260914)
    W = sample_weights(rng, N_SAMPLE)
    W = np.vstack([W, np.array([[PRESETS[p][k] for k in AXES] for p in PRESETS
                                if max(PRESETS[p].values()) <= MAXW])])
    print("도달 가능한 가중치 %d개 (각 축 0~%g, 합 %g) + 프리셋"
          % (len(W), MAXW, TOTAL))

    for key, label in (("A", "A 현행  p = softmax(raw × T)"),
                       ("B", "B 표준화 p = softmax(z × T)")):
        Ts, lls, raws = [], [], []
        for w in W:
            raw = A @ w / 100.0
            if key == "B":
                raw = race_stat(va, raw, q, nq, "z")
            raws.append(raw)
            r = optimize.minimize_scalar(nll, bounds=(1e-3, 20.0), method="bounded",
                                         args=(raw, q, win, nq), options=dict(xatol=1e-6))
            Ts.append(float(r.x)); lls.append(float(r.fun))
        Ts, lls = np.array(Ts), np.array(lls)

        def total(t):
            return float(np.mean([nll(t, r, q, win, nq) for r in raws]))

        g = optimize.minimize_scalar(total, bounds=(1e-3, 20.0), method="bounded",
                                     options=dict(xatol=1e-6))
        Tg = float(g.x)
        loss = np.array([nll(Tg, r, q, win, nq) - lls[i] for i, r in enumerate(raws)])
        print("")
        print("  " + label)
        print("    가중치별 T*      %.4f ~ %.4f   (중앙값 %.4f · 최대/최소 %.2f배)"
              % (Ts.min(), Ts.max(), np.median(Ts), Ts.max() / Ts.min()))
        print("    공통 상수 T      %.4f" % Tg)
        print("    그때의 손해      평균 %+.4f · 95%%분위 %+.4f · 최악 %+.4f"
              % (loss.mean(), np.quantile(loss, 0.95), loss.max()))

    # 현행 계약에서 세 후보값의 성적
    print("")
    print("=" * 74)
    print("현행 계약(A) — 후보 T 셋을 도달 가능한 가중치 %d개에 전부 적용" % len(W))
    print("=" * 74)
    raws = [A @ w / 100.0 for w in W]
    best = np.array([optimize.minimize_scalar(nll, bounds=(1e-3, 20.0), method="bounded",
                                              args=(r, q, win, nq),
                                              options=dict(xatol=1e-6)).fun for r in raws])
    print("  %-22s%12s%12s%12s" % ("T", "평균 손해", "95%분위", "최악"))
    print("  " + "-" * 58)
    for t, lab in ((SPEC_T, "0.0445  명세·BE상수"), (IMPL_T, "0.1     PredictionService"),
                   (0.075, "0.075   실측 중앙값")):
        d = np.array([nll(t, r, q, win, nq) - best[i] for i, r in enumerate(raws)])
        print("  %-22s%+12.4f%+12.4f%+12.4f" % (lab, d.mean(), np.quantile(d, 0.95), d.max()))


if __name__ == "__main__":
    main()
