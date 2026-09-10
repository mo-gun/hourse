# -*- coding: utf-8 -*-
"""팀 하네스 진입점 — **데이터를 직접 읽지 않는다.**

경로 규칙은 `basemodel/basemodel/team_paths.py` 와 같은 것을 쓴다(중복 구현하면 두 곳이
어긋난다). 찾는 순서: TEAM_HARNESS → <repo>/pipeline/data → <repo>/docs → TEAM_REPO.

여기서 고정하는 것 — 바꾸면 레포의 다른 표와 비교가 무효다:
  · 피처는 `common.feature_cols()` 가 주는 77 / 73 그대로. 늘리거나 줄이지 않는다
  · 전처리는 `common.load()` 그대로 (= schema_v2.clean() + usable())
  · 정렬 유지. race_id 연속 블록이고 group/qid 가 그 순서를 가정한다
  · game 은 어디에도 쓰지 않는다. test 는 열지 않는다 (load 가 valid 만 내준다)
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()


def _resolve() -> tuple[Path, Path]:
    cand: list[tuple[Path, Path]] = []
    if env := os.environ.get("TEAM_HARNESS"):
        p = Path(env).expanduser()
        cand.append((p / "model", p / "dataset"))
    for base in _HERE.parents:
        cand.append((base / "pipeline" / "data" / "model", base / "pipeline" / "data" / "dataset"))
        cand.append((base / "docs" / "model", base / "docs" / "dataset"))
    if env := os.environ.get("TEAM_REPO"):
        p = Path(env).expanduser() / "docs"
        cand.append((p / "model", p / "dataset"))
    for m, d in cand:
        if (m / "common.py").exists() and (d / "schema_v2.py").exists():
            return m, d
    raise FileNotFoundError(
        "팀 하네스를 찾지 못했다 (common.py + schema_v2.py).\n"
        "  cd pipeline && TEAM_REPO=<...>/S15P21A304 bash sync_dataset.sh")


TEAM_MODEL, TEAM_DATASET = _resolve()
sys.path.insert(0, str(TEAM_MODEL))
import common as C  # noqa: E402
sys.path.insert(0, str(TEAM_DATASET))
import schema_v2 as S  # noqa: E402

SEEDS = (0, 1, 2)           # 시드 3개 이상 — top-1 은 시드만 바꿔도 ±1%p 움직인다
CATEGORICAL = list(S.CATEGORICAL)


@lru_cache(maxsize=None)
def splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, valid). test 는 최종 1회용이라 이 함수가 아예 내주지 않는다."""
    return C.load("train"), C.load("valid")


def cols(pop: bool) -> list[str]:
    """pop=True → 77피처(인기도 포함, 게임 리플레이) · False → 73피처(주말 실시간)."""
    return C.feature_cols(exclude_pop=not pop)


def qid(df: pd.DataFrame) -> np.ndarray:
    """경주 id → 0,1,2,... 등장 순서. 연속 블록이라 비감소이므로 xgboost qid 로 쓸 수 있다."""
    q = pd.factorize(df["race_id"], sort=False)[0]
    assert (np.diff(q) >= 0).all(), "race_id 정렬이 깨졌다 — 셔플하지 말 것"
    return q


def groups(df: pd.DataFrame) -> np.ndarray:
    return C.race_groups(df)


def numeric(tr: pd.DataFrame, ev: pd.DataFrame, feats: list[str]):
    """범주형 → 정수 코드 (팀 common.encode 그대로). LightGBM·XGBoost·트리 계열용."""
    a, b = C.encode(tr, ev, feats)
    return a[feats].to_numpy(float), b[feats].to_numpy(float)


def with_strings(tr: pd.DataFrame, ev: pd.DataFrame, feats: list[str]):
    """CatBoost 용 — 범주형을 **문자열 그대로** 남긴다.

    팀 README 의 "범주형 native 악화(29.2)" 는 LightGBM 의 categorical_feature 얘기다.
    CatBoost 의 ordered target statistics 는 기제가 달라서 따로 재볼 값이 있다.
    """
    cat = [c for c in feats if c in CATEGORICAL or not pd.api.types.is_numeric_dtype(tr[c])]
    out = []
    for d in (tr, ev):
        x = d[feats].copy()
        for c in cat:
            x[c] = x[c].astype(str).fillna("<na>")
        for c in feats:
            if c not in cat:
                x[c] = pd.to_numeric(x[c], errors="coerce").astype(float)
        out.append(x)
    return out[0], out[1], [feats.index(c) for c in cat]
