# -*- coding: utf-8 -*-
"""피처 기여도 — "지금 무엇을 얼마나 보고 예측하나" 를 직관적으로.

LightGBM gain 기준. gain = 그 피처로 자른 분기들이 손실을 줄인 총량이라, "모델이
실제로 판단에 쓴 양" 에 가장 가깝다(빈도(split)로 보면 값 종류가 많은 피처가 과대평가된다).

세 가지 눈금으로 같이 본다 — 어느 하나만 보면 오해한다.
  · 스키마 그룹 (X·F1~F6)  = 데이터를 만든 사람이 나눈 출처
  · 제품 6축              = 화면 슬라이더 (basemodel/config.py 의 번역표)
  · 개별 피처 상위         = 실제로 어떤 숫자가 판단을 끌고 가나

    uv run python -m arena.importance
"""
from __future__ import annotations

import importlib.util as ilu
import sys
from pathlib import Path

import numpy as np

from . import harness as H
from .models import LGB_BASE, N_ROUND


def _axis_map() -> dict[str, str]:
    """basemodel/config.py 의 AXIS_FEATURES 를 그대로 읽는다 (중복 정의 금지)."""
    p = Path(__file__).resolve().parents[2] / "basemodel" / "basemodel" / "config.py"
    if not p.exists():
        return {}
    sp = ilu.spec_from_file_location("bm_config", p)
    m = ilu.module_from_spec(sp); sys.modules["bm_config"] = m; sp.loader.exec_module(m)
    out = {}
    for axis, feats in m.AXIS_FEATURES.items():
        for f in feats:
            out[f] = axis
    for f in m.RACE_CONSTANT:
        out.setdefault(f, "CORE(경주내상수)")
    return out


def gains(pop: bool, seed: int = 0):
    import lightgbm as lgb
    tr, ev = H.splits()
    feats = H.cols(pop)
    xa, _ = H.numeric(tr, ev, feats)
    ds = lgb.Dataset(xa, label=tr["y_rel"].to_numpy(), group=H.groups(tr))
    m = lgb.train(dict(LGB_BASE, objective="lambdarank", seed=seed), ds,
                  num_boost_round=N_ROUND)
    g = m.feature_importance("gain")
    tot = g.sum() or 1
    return feats, g / tot * 100


def bar(pct: float, width: int = 28) -> str:
    return "█" * int(round(pct / 100 * width)) + "·" * (width - int(round(pct / 100 * width)))


def report(pop: bool):
    feats, pct = gains(pop)
    tag = "77피처 (게임 리플레이 — 배당 포함)" if pop else "73피처 (주말 실시간 — 배당 없음)"
    amap = _axis_map()
    print("=" * 74)
    print(f"[{tag}]  LightGBM gain 기준, 합 100%")
    print("=" * 74)

    print("\n■ 스키마 그룹별")
    grp = {}
    for f, p in zip(feats, pct):
        grp[f.split("_")[0]] = grp.get(f.split("_")[0], 0) + p
    NAME = {"X": "X  공통(마령·거리·등급·게이트·중량·레이팅)",
            "F1": "F1 최근 성적·컨디션·조교", "F2": "F2 혈통·육종가",
            "F3": "F3 기수·조교사", "F4": "F4 거리·주로·날씨 적응",
            "F5": "F5 각질·초반위치·막판각력", "F6": "F6 인기도(배당)"}
    for k, v in sorted(grp.items(), key=lambda x: -x[1]):
        print(f"  {NAME.get(k, k):<42}{v:>6.1f}%  {bar(v)}")

    print("\n■ 제품 6축별 (화면 슬라이더 단위)")
    ax = {}
    for f, p in zip(feats, pct):
        ax[amap.get(f, "CORE(공유)")] = ax.get(amap.get(f, "CORE(공유)"), 0) + p
    KO = {"CONDITION": "컨디션", "SPEED": "스피드", "RUNNING": "주행",
          "JOCKEY": "기수", "ENVIRONMENT": "환경", "ABILITY": "실력", "MARKET": "시장(배당)"}
    for k, v in sorted(ax.items(), key=lambda x: -x[1]):
        print(f"  {KO.get(k, k):<42}{v:>6.1f}%  {bar(v)}")

    print("\n■ 개별 피처 상위 12")
    order = np.argsort(-pct)[:12]
    for i in order:
        print(f"  {feats[i]:<42}{pct[i]:>6.1f}%  {bar(pct[i])}")
    cum = pct[np.argsort(-pct)]
    for n in (3, 5, 10, 20):
        print(f"    상위 {n:>2}개 누적 {cum[:n].sum():>5.1f}%")


def main():
    for pop in (False, True):
        report(pop)
        print()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
