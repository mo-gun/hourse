# -*- coding: utf-8 -*-
"""게임풀 점수표 산출 — **폐기됨 (2026-09-16). 돌리지 마라.**

## 왜 안 쓰나

도연님이 2026-09-15 에 이미 적재했다 (커밋 "feat(pool): 실경주 1,140건 게임풀 적재").
`back/src/main/resources/pool/pool_races.jsonl` 에 축 점수가 들어 있고, 메타를 보면
**이 문서가 정한 규격을 그대로 따랐다** — `temperature: 0.0445`, 축 점수 0~100,
축 이름 CONDITION…ABILITY.

그리고 도연님 쪽이 **제품에 더 맞다**:

    도연 (적용됨)   1,140경주   8~14두 · 배당 필수 · 완주만 · 동착 제외
    이 스크립트     3,635경주   필터 없음

8~14두 제한은 게임 경험, 배당 필수는 화면 표시 때문이다. 필터 없는 쪽을 넣으면
두수 6짜리 경주나 배당이 없는 경주가 게임에 섞인다.

## 남겨 두는 이유

산출 경로 자체는 검증됐다(조인 100.00%, 파일 재읽기 34.2% 재현). 나중에 게임풀을
다시 만들 일이 생기면 **필터를 도연님 것에 맞춘 뒤** 참고용으로 쓸 수 있다.
그대로 돌리면 제품과 다른 표본이 나오니 주의할 것.

────────────────────────────────────────────────────────────────────────
아래는 원래 설명이다.

게임풀 3,700경주 점수표 산출 (AI-05) — entry_feature_score 에 넣을 값.

## 무엇을 만드나

경주 × 출전마 × 41칸. **긴 형식(1.5M행)이 아니라 넓은 형식(3.8만행 × 44열)** 으로 낸다 —
백엔드가 unpivot 하는 편이 파일도 작고 옮기기도 쉽다.

    race_id · chulNo · hrNo          자연키. 팀 DB 의 race_entry_id 는 여기서 매핑한다
    ax_* 6칸                          점수 계산. basemodel disp_* (경주 내 순위 백분위 0~100)
    표시 지표 35칸 (score/raw_value)  화면용

## 왜 disp_* 인가

basemodel 은 축을 두 벌로 낸다 — `ax_*`(경주 내 z-score)와 `disp_*`(순위 백분위 0~100).
export.py:72 주석은 순위 환산이 "1σ 앞선 말이 같은 값이 되어 확률이 무너진다"고 적고
z 쪽을 확률용으로 뒀다. **재 보니 무너지지는 않는다** — top-1 33.2% → 32.9%,
logloss 1.9223 → 1.9413. 표준오차(±0.85%p) 안이다.

그 0.3%p 를 내주고 얻는 것:
  · 스키마가 적은 `score DECIMAL(5,1)` 0~100 범위에 맞는다
  · **T = 0.0445 가 모델과 무관해진다** (생성기 5종에서 0.0406~0.0478 재현)
  · 모델을 바꿔도 백엔드 상수를 안 고쳐도 된다

## 규약
    game 은 학습·검증·튜닝에 쓰지 않았다. 여기서 하는 일은 **추론**이다.
    게임풀은 학습 시기(2015~2024)와 섞여 있어 valid 보다 낙관적이다 — 표기는 valid 값으로.

    python tools/emit_game_pool.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "tools"))
EXPORT = Path(r"c:/Users/SSAFY/Desktop/말고리즘/horse-pred-engine/basemodel/artifacts/export")
TEAM = Path(r"c:/Users/SSAFY/Desktop/말고리즘/S15P21A304")
OUT = HERE / "out"
AXES = ["CONDITION", "SPEED", "RUNNING", "JOCKEY", "ENVIRONMENT", "ABILITY"]
MODEL_VERSION = "2026-09-15-axis77"
SPEC_T = 0.0445


def main():
    master = json.loads((OUT / "feature_master.json").read_text(encoding="utf-8"))
    shown = [f for f in master["features"] if f["role"] == "PRIMARY"]
    meta = json.loads((EXPORT / "meta.json").read_text(encoding="utf-8"))

    A = pd.read_parquet(EXPORT / "axis_scores.parquet")
    G = pd.read_parquet(TEAM / "docs" / "dataset" / "model" / "game.parquet")
    print("축 점수 %s (%d경주) · 게임풀 원본 %s (%d경주)"
          % (A.shape, A["race_id"].nunique(), G.shape, G["race_id"].nunique()))

    # 자연키로 붙인다 — race_id + chulNo. axis_scores 는 센티널이 빠져 행이 적다
    G["chulNo"] = pd.to_numeric(G["row_id"].astype(str).str.rsplit("_", n=1).str[-1])
    A["chulNo"] = pd.to_numeric(A["chulNo"])
    M = A.merge(G, on=["race_id", "chulNo"], how="inner", suffixes=("", "_g"))
    print("조인 %d행 (%.2f%%)" % (len(M), len(M) / len(A) * 100))
    assert len(M) >= len(A) * 0.99, "조인이 1%% 넘게 샜다 — 키를 확인해라"

    rid = M["race_id"]
    out = M[["race_id", "chulNo", "hrNo"]].copy()

    # ── 축 6칸 — disp_* 가 이미 경주 내 순위 백분위 0~100 이다 ──────────
    for ax in AXES:
        out["ax_" + ax.lower()] = M["disp_" + ax.lower()].round(1)
    out["ax_market"] = M["disp_market"].round(1)     # 제품 스키마 밖. §참고

    # ── 표시 지표 35칸 ─────────────────────────────────────────────────
    n_miss = 0
    for f in shown:
        c = f["source_column"]
        if c not in M.columns:
            n_miss += 1
            continue
        v = pd.to_numeric(M[c], errors="coerce")
        p = (v.groupby(rid).rank(method="average", pct=True) * 100).fillna(50.0)
        out[f["code"]] = (p if f["direction"] > 0 else 100.0 - p).round(1)
        out["rv_" + f["code"]] = v.round(3)
    if n_miss:
        print("  ⚠ 게임풀에 없는 지표 %d개 — 건너뛴다" % n_miss)

    # ── raw_prob — 기본형 프리셋 기준 ──────────────────────────────────
    w = meta["presets"]["BASIC"]["groupWeights"]
    raw = sum(w[a] * out["ax_" + a.lower()] for a in AXES) / 100.0
    z = raw * SPEC_T
    e = np.exp(z - z.groupby(rid).transform("max"))
    out["raw_prob"] = (e / e.groupby(rid).transform("sum")).round(5)
    out["model_version"] = MODEL_VERSION

    # ── 검증 — 이 파일만으로 성적이 재현되나 ──────────────────────────
    p = OUT / "game_pool_scores.parquet"
    out.to_parquet(p, index=False)
    B = pd.read_parquet(p)
    r2 = sum(w[a] * B["ax_" + a.lower()] for a in AXES) / 100.0
    top = r2.groupby(B["race_id"]).transform("max")
    pick = (r2 >= top - 1e-9) & (B.groupby("race_id")["race_id"].transform("size") > 0)
    y = M["y_win"].to_numpy(float)
    first = pick & ~pick.groupby(B["race_id"]).cumsum().gt(1)     # 동점은 첫 행
    hit = y[first.to_numpy()].mean() * 100
    print("")
    print("=" * 66)
    print("게임풀 점수표 — %d경주 %d두 × %d칸" % (out["race_id"].nunique(), len(out),
                                          len([c for c in out.columns
                                               if not c.startswith(("rv_", "race_id", "chulNo",
                                                                    "hrNo", "raw_prob", "model"))])))
    print("=" * 66)
    print("  파일 %s (%.1fMB)" % (p.name, p.stat().st_size / 1e6))
    print("  경주당 합 1.0 인가 — raw_prob 합 평균 %.4f"
          % out.groupby("race_id")["raw_prob"].sum().mean())
    print("  파일에서 다시 채점 (기본형) top-1 %.1f%%" % hit)
    print("  export 보고값 34.3%% · valid 기준 32.9%% ← 표기는 valid 값으로")
    print("")
    print("  ⚠ 게임풀은 학습 시기와 섞여 있어 낙관적이다. 제품 표기에 쓰지 말 것")

    # 백엔드가 확인할 수 있게 앞 2경주만 CSV
    s = out[out["race_id"].isin(out["race_id"].unique()[:2])]
    s.to_csv(OUT / "game_pool_sample.csv", index=False, encoding="utf-8-sig")
    print("  → %s (앞 2경주 %d행, 사람이 보는 용도)" % ("game_pool_sample.csv", len(s)))


if __name__ == "__main__":
    main()
