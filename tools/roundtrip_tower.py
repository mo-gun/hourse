# -*- coding: utf-8 -*-
"""왕복 검증 — basemodel 6축 타워의 축 점수를 제품 계약에 태워 성적을 확인한다.

## 재는 것

타워가 낸 축 점수(경주 내 z-score)를 우리가 표준으로 정한 ②(경주 내 순위 환산 0~100)
로 바꾸면 성적이 유지되는가. 순위 환산은 축마다 **단조 변환**이라 축 안의 순서는 그대로지만
축끼리의 상대 폭이 달라지므로, 가중합 결과는 바뀔 수 있다. 공짜가 아니라는 뜻이라 재야 한다.

    z 그대로     raw = Σ(w · z_k)/100        타워가 학습한 형태
    ② 순위 환산  raw = Σ(w · rank100_k)/100  DB 에 넣을 형태 (schema.sql:307)

같이 T 도 다시 맞춰 0.041~0.045 자리에 또 떨어지는지 본다 — 다섯 번째 확인이다.

## 규약
    valid 로만 잰다 · test 는 열지 않는다 · 채점은 팀 채점기를 거친다

    python tools/roundtrip_tower.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_temperature import AXES, race_index, nll, C            # noqa: E402

CKPT = Path(r"c:/Users/SSAFY/Desktop/말고리즘/horse-pred-engine/basemodel/artifacts/runs")
STEM = "axis_73_s20260901_final"
OUT = Path(__file__).resolve().parent.parent / "out"
SPEC_T = 0.0445

PRESETS = {
    "균등": {k: 100 / 6 for k in AXES},
    "기본형": dict(CONDITION=17, SPEED=17, RUNNING=12, JOCKEY=22, ENVIRONMENT=19, ABILITY=13),
    "사람형": dict(CONDITION=15, SPEED=13, RUNNING=12, JOCKEY=35, ENVIRONMENT=15, ABILITY=10),
    "상승세형": dict(CONDITION=32, SPEED=16, RUNNING=12, JOCKEY=20, ENVIRONMENT=12, ABILITY=8),
}


def rank100(v, rid):
    return (pd.Series(v).groupby(rid.values).rank(pct=True) * 100).to_numpy(float)


def main():
    va = C.load("valid")
    Z = np.load(CKPT / (STEM + ".valid_axes.npy"))
    meta = json.loads((CKPT / (STEM + ".json")).read_text(encoding="utf-8"))
    print("체크포인트 %s" % STEM)
    print("축 점수 배열 %s   메타 축 %s" % (Z.shape, meta.get("axes", "?")))
    assert len(Z) == len(va), "행 수가 valid 와 다르다 — 정렬이 어긋났다"
    A = Z.shape[1]
    if A > 6:
        print("  ⚠ 축이 %d개다 — 앞 6개(슬라이더 축)만 쓴다. 나머지는 시장 계수용" % A)
        Z = Z[:, :6]

    rid = va["race_id"]
    q = race_index(va); nq = q.max() + 1
    win = va["y_win"].to_numpy(bool)
    R = np.stack([rank100(Z[:, i], rid) for i in range(6)], 1)      # ② 순위 환산

    print("")
    print("축 점수 폭 (표준편차) — ② 를 거치면 여섯 축이 같아진다")
    print("    z 그대로  " + "  ".join("%s %.2f" % (a[:4], s) for a, s in zip(AXES, Z.std(0))))
    print("    ② 환산    " + "  ".join("%s %.1f" % (a[:4], s) for a, s in zip(AXES, R.std(0))))

    rng = np.random.default_rng(20260914)
    W = []
    while len(W) < 100:
        g = rng.dirichlet(np.ones(6)) * 100
        if g.max() <= 40:
            W.append(g)

    print("")
    print("=" * 80)
    print("왕복 검증 — valid 1,254경주 (top-1 표준오차 ±0.85%p)")
    print("=" * 80)
    print("%-14s%22s%22s" % ("", "z 그대로", "② 순위 환산 0~100"))
    print("%-14s%11s%11s%11s%11s" % ("가중치", "top-1", "top-3", "top-1", "top-3"))
    print("-" * 80)
    for name, w in PRESETS.items():
        v = np.array([w[k] for k in AXES])
        mz, mr = C.evaluate(va, Z @ v / 100.0), C.evaluate(va, R @ v / 100.0)
        print("%-14s%10.1f%%%10.1f%%%10.1f%%%10.1f%%"
              % (name, mz["top1"], mz["top3"], mr["top1"], mr["top3"]))
    print("-" * 80)

    for lab, M in (("z 그대로", Z), ("② 순위 환산", R)):
        raws = [M @ np.array(g) / 100.0 for g in W]
        def total(t):
            return float(np.mean([nll(t, r, q, win, nq) for r in raws]))
        T = float(optimize.minimize_scalar(total, bounds=(1e-4, 50.0), method="bounded",
                                           options=dict(xatol=1e-7)).x)
        best = np.mean([optimize.minimize_scalar(
            nll, bounds=(1e-4, 50.0), method="bounded", args=(r, q, win, nq),
            options=dict(xatol=1e-7)).fun for r in raws])
        extra = ""
        if lab.startswith("②"):
            extra = "   명세 0.0445 손해 %+.4f" % (total(SPEC_T) - best)
        print("  %-14s 공통 T %8.4f   logloss %.4f (최적 대비 %+.4f)%s"
              % (lab, T, total(T), total(T) - best, extra))

    print("")
    print("참고  LGB 축 랭커 6개(균등) 31.0% · 순위환산 지표평균 28.3% · LGB 73 통짜 33.5%")
    np.save(OUT / "tower_axis_rank100_valid.npy", R)
    print("→ out/tower_axis_rank100_valid.npy")


if __name__ == "__main__":
    main()
