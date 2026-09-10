# -*- coding: utf-8 -*-
"""기법 계열 — 각 함수는 (tr, ev, feats, seed) → ev 행별 점수(클수록 1착 후보).

레포에서 **이미 덮은 것은 넣지 않았다**:
  pipeline/ S1 선형 조건부로짓 · S2 임베딩 · S3 전적 GRU · S4 경주내 어텐션 · S5 타워
  basemodel/ AxisRanker(6축 + 시장계수) · 프리셋 적합 · 온도보정
  팀 baseline_lgbm.py LightGBM lambdarank 200r  ← `lgb_lambdarank` 로 기준선만 재현

여기서 새로 붙이는 축은 셋이다:
  ① **GBDT 다른 구현·다른 목적함수** — 정설이 "표 데이터는 GBDT" 인데 레포는 LightGBM
     lambdarank 한 설정만 재봤다. 특히 `cat_querysoftmax` 는 **조건부 로짓(=Plackett-Luce)
     우도를 트리로 최적화**하는 것으로, 레포의 PL 손실은 전부 신경망에만 얹혀 있었다.
  ② **배깅 계열** — Lessmann 외(2010)가 "경쟁구조 반영 RF 가 조건부로짓을 상회"라고
     보고했고 schema_v2 LIT 에 `LS` 로 등재돼 있는데 아무도 안 돌렸다.
  ③ **범주형 native (CatBoost)** — 팀 README 의 "범주형 native 악화(29.2)" 는 LightGBM
     categorical_feature 얘기다. CatBoost 의 ordered target statistics 는 기제가 다르다.

하이퍼파라미터는 계열마다 **기준선과 대응되는 상식적 한 점**으로 잡았다(깊이·학습률·
라운드·서브샘플을 LightGBM 설정에 맞춤). 계열별 정밀 튜닝은 하지 않았다 — 먼저
"계열을 바꿔서 선을 넘을 여지가 있나"를 보는 게 목적이고, 정원님 장부에 이미
"하이퍼파라미터 탐색: 기본 설정이 최적" 이 있다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import harness as H

EXTRA = Path(__file__).resolve().parents[1] / "artifacts" / "extra"

LGB_BASE = dict(
    metric="ndcg", ndcg_eval_at=[3],
    lambdarank_truncation_level=5,
    learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    verbose=-1, num_threads=8,
)
N_ROUND = 200


# ── ① GBDT ────────────────────────────────────────────────────────────
def _lgb(tr, ev, feats, seed, label="y_rel", rounds=N_ROUND, **over):
    import lightgbm as lgb
    xa, xb = H.numeric(tr, ev, feats)
    ds = lgb.Dataset(xa, label=tr[label].to_numpy(), group=H.groups(tr))
    p = dict(LGB_BASE, objective="lambdarank", seed=seed)
    p.update(over)                      # objective 를 덮어쓰는 계열이 있다
    return lgb.train(p, ds, num_boost_round=rounds).predict(xb)


def lgb_lambdarank(tr, ev, feats, seed):
    """팀 기준선 재현 — 이 표의 기준점."""
    return _lgb(tr, ev, feats, seed)


def lgb_xendcg(tr, ev, feats, seed):
    """rank_xendcg — lambdarank 와 달리 XE-NDCG 대리손실. 잡음 큰 라벨에서 낫다고 보고됨."""
    return _lgb(tr, ev, feats, seed, objective="rank_xendcg")


def lgb_binary(tr, ev, feats, seed):
    """랭킹이 아니라 '이 말이 1착인가' 이진분류. 경주 경쟁구조를 안 쓰는 대조군."""
    import lightgbm as lgb
    xa, xb = H.numeric(tr, ev, feats)
    p = dict(LGB_BASE, objective="binary", seed=seed)
    p.pop("metric"); p.pop("ndcg_eval_at"); p.pop("lambdarank_truncation_level")
    p["metric"] = "binary_logloss"
    ds = lgb.Dataset(xa, label=tr["y_win"].to_numpy())
    return lgb.train(p, ds, num_boost_round=N_ROUND).predict(xb)


def _xgb(tr, ev, feats, seed, objective):
    """LightGBM 설정과 대응시킨다 — 잎 31개(lossguide), lr 0.05, 200라운드, 서브샘플 0.8.

    ⚠ `min_child_weight` 는 **행 수가 아니라 헤시안 합**이다. LightGBM 의
    `min_data_in_leaf=200` 을 그대로 200 으로 옮기면 랭킹 목적함수의 헤시안이 작아서
    사실상 잎당 수천 행을 요구하게 되고 심하게 과소적합한다 — 첫 실행에서 실제로
    그렇게 했고 xgb 가 −0.8%p 로 나왔다. 잎 수 상한으로 용량을 맞추고 이 값은 풀었다.
    """
    import xgboost as xgb
    xa, xb = H.numeric(tr, ev, feats)
    m = xgb.XGBRanker(objective=objective, tree_method="hist",
                      grow_policy="lossguide", max_leaves=31, max_depth=0,
                      n_estimators=N_ROUND, learning_rate=0.05,
                      min_child_weight=1.0, subsample=0.8, colsample_bytree=0.8,
                      random_state=seed, n_jobs=8)
    m.fit(xa, tr["y_rel"].to_numpy(), qid=H.qid(tr))
    return m.predict(xb)


def xgb_pairwise(tr, ev, feats, seed):
    return _xgb(tr, ev, feats, seed, "rank:pairwise")


def xgb_ndcg(tr, ev, feats, seed):
    return _xgb(tr, ev, feats, seed, "rank:ndcg")


def _cat(tr, ev, feats, seed, loss, iters=400):
    from catboost import CatBoostRanker, Pool
    xa, xb, cat_idx = H.with_strings(tr, ev, feats)
    pa = Pool(xa, label=tr["y_rel"].to_numpy(float), group_id=H.qid(tr), cat_features=cat_idx)
    pb = Pool(xb, group_id=H.qid(ev), cat_features=cat_idx)
    m = CatBoostRanker(loss_function=loss, iterations=iters, learning_rate=0.05,
                       depth=6, random_seed=seed, verbose=0, thread_count=8,
                       allow_writing_files=False)
    m.fit(pa)
    return m.predict(pb)


def cat_querysoftmax(tr, ev, feats, seed):
    """★ 조건부 로짓(=Plackett-Luce 1착 항) 우도를 **트리로** 최적화.

    레포의 PL 손실은 전부 신경망(S1~S5·AxisRanker)에만 얹혀 있었다. Benter(1994)의
    모델식을 GBDT 로 푸는 셈이라, "표 데이터는 GBDT 가 강하다"와 "경마는 조건부 로짓"이
    처음으로 같은 모델에서 만난다. 이 폴더에서 가장 기대치가 높은 한 칸.
    """
    return _cat(tr, ev, feats, seed, "QuerySoftMax")


def cat_yetirank(tr, ev, feats, seed):
    """YetiRank — 리스트와이즈. lambdarank 의 CatBoost 대응."""
    return _cat(tr, ev, feats, seed, "YetiRank")


def hist_gb(tr, ev, feats, seed):
    """sklearn HistGradientBoosting — 세 번째 GBDT 구현. 같은 계열이 구현마다 다른지 본다."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    xa, xb = H.numeric(tr, ev, feats)
    m = HistGradientBoostingRegressor(max_iter=N_ROUND, learning_rate=0.05,
                                      max_leaf_nodes=31, min_samples_leaf=200,
                                      random_state=seed)
    m.fit(xa, tr["y_rel"].to_numpy(float))
    return m.predict(xb)


# ── ② 배깅 (Lessmann 2010, LIT=LS) ────────────────────────────────────
def rf(tr, ev, feats, seed):
    from sklearn.ensemble import RandomForestRegressor
    xa, xb = H.numeric(tr, ev, feats)
    xa = np.nan_to_num(xa, nan=-999.0); xb = np.nan_to_num(xb, nan=-999.0)
    m = RandomForestRegressor(n_estimators=300, min_samples_leaf=200, max_features=0.8,
                              random_state=seed, n_jobs=8)
    m.fit(xa, tr["y_rel"].to_numpy(float))
    return m.predict(xb)


def extra_trees(tr, ev, feats, seed):
    from sklearn.ensemble import ExtraTreesRegressor
    xa, xb = H.numeric(tr, ev, feats)
    xa = np.nan_to_num(xa, nan=-999.0); xb = np.nan_to_num(xb, nan=-999.0)
    m = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=200, max_features=0.8,
                            random_state=seed, n_jobs=8)
    m.fit(xa, tr["y_rel"].to_numpy(float))
    return m.predict(xb)


# ── ③ 라벨 프레이밍 — 회귀 계열에는 y_rel 이 불공정하다 ────────────────
def _norm_label(df) -> np.ndarray:
    """경주 내 정규화 착순 (1착=1.0, 최하위=0.0).

    `y_rel = max(0, dusu - ord)` 는 **두수에 비례**한다 — 16두 경주 1착이 15, 8두 경주
    1착이 7 이다. lambdarank 는 group 안에서만 비교하니 무해하지만, RF·HistGB 처럼
    행 단위 회귀로 푸는 계열은 두수를 부분적으로 학습하게 된다. 두수는 경주 내 상수라
    순위를 못 바꾸므로 그만큼 용량이 낭비된다.
    """
    dusu = df["X_dusu"].to_numpy(float)
    ordv = df["y_ord"].to_numpy(float)
    return np.clip((dusu - ordv) / np.maximum(dusu - 1, 1), 0, 1)


def _tree_reg(cls, tr, ev, feats, seed, label, **kw):
    xa, xb = H.numeric(tr, ev, feats)
    xa = np.nan_to_num(xa, nan=-999.0); xb = np.nan_to_num(xb, nan=-999.0)
    y = _norm_label(tr) if label == "norm" else tr["y_rel"].to_numpy(float)
    m = cls(random_state=seed, **kw)
    m.fit(xa, y)
    return m.predict(xb)


def rf_norm(tr, ev, feats, seed):
    """RF + 경주 내 정규화 라벨 — 위 라벨 문제의 대조군."""
    from sklearn.ensemble import RandomForestRegressor
    return _tree_reg(RandomForestRegressor, tr, ev, feats, seed, "norm",
                     n_estimators=300, min_samples_leaf=200, max_features=0.8, n_jobs=8)


def hist_gb_norm(tr, ev, feats, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    xa, xb = H.numeric(tr, ev, feats)
    m = HistGradientBoostingRegressor(max_iter=N_ROUND, learning_rate=0.05,
                                      max_leaf_nodes=31, min_samples_leaf=200,
                                      random_state=seed)
    m.fit(xa, _norm_label(tr))
    return m.predict(xb)


# ── ④ 타깃 프레이밍 — 순위가 아니라 **기록**을 맞힌다 ──────────────────
def lgb_speedfig(tr, ev, feats, seed):
    """스피드지수(`y_speed_fig`)를 회귀로 맞히고 그 예측값으로 줄 세운다.

    레포의 모든 모델은 목적함수가 **순위**(lambdarank/NDCG)거나 **1착 확률**(PL)이다.
    실무 핸디캐핑의 고전 방식은 그게 아니라 "이 말이 몇 초에 달릴까"를 맞히고
    빠른 순으로 세우는 것이다(schema_v2 LIT 의 `PACE` 계열). 타깃이 아예 다르므로
    같은 피처에서도 다른 오차 구조를 갖는다 — 앙상블 재료로 특히 값이 있다.

    ⚠ `y_speed_fig` 는 타깃(tier=P)이라 **입력이 아니라 라벨로만** 쓴다.
    """
    import lightgbm as lgb
    xa, xb = H.numeric(tr, ev, feats)
    y = tr["y_speed_fig"].to_numpy(float)
    ok = np.isfinite(y)
    p = dict(objective="regression", metric="l2", learning_rate=0.05, num_leaves=31,
             min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
             bagging_freq=1, verbose=-1, num_threads=8, seed=seed)
    ds = lgb.Dataset(xa[ok], label=y[ok])
    return lgb.train(p, ds, num_boost_round=N_ROUND).predict(xb)


# -- 5) 표현 변환 — 피처를 섞으면 나아지나 --------------------------------
def _pca_lgb(tr, ev, feats, seed, n_comp):
    """수치화 -> 중앙값 대치 -> 표준화 -> PCA -> LightGBM lambdarank.

    PCA 는 train 에서만 적합한다(valid 를 보면 안 된다).
    n_comp=None 이면 성분을 전부 유지한다 = **압축 없는 순수 회전.** 정보량이 같으므로
    이 칸이 떨어지면 "정보가 줄어서"가 아니라 **트리가 축 정렬 분기를 쓰기 때문**이다.
    """
    import lightgbm as lgb
    from sklearn.decomposition import PCA
    xa, xb = H.numeric(tr, ev, feats)
    med = np.nanmedian(xa, axis=0)
    xa = np.where(np.isnan(xa), med, xa)
    xb = np.where(np.isnan(xb), med, xb)
    mu, sd = xa.mean(0), xa.std(0)
    sd[sd == 0] = 1.0
    xa, xb = (xa - mu) / sd, (xb - mu) / sd
    p = PCA(n_components=n_comp or xa.shape[1], svd_solver="full",
            random_state=seed).fit(xa)
    za, zb = p.transform(xa), p.transform(xb)
    ds = lgb.Dataset(za, label=tr["y_rel"].to_numpy(), group=H.groups(tr))
    m = lgb.train(dict(LGB_BASE, objective="lambdarank", seed=seed), ds,
                  num_boost_round=N_ROUND)
    print("      PCA %d성분 · 설명분산 %.3f" % (za.shape[1], p.explained_variance_ratio_.sum()))
    return m.predict(zb)


def lgb_pca40(tr, ev, feats, seed):
    """PCA 40성분 — 분산 기준 압축."""
    return _pca_lgb(tr, ev, feats, seed, 40)


def lgb_pca_rot(tr, ev, feats, seed):
    """PCA 전 성분 유지 = 순수 회전. 정보량 동일, 축만 바뀐다."""
    return _pca_lgb(tr, ev, feats, seed, None)


# -- 6) 정답지로 만든 새 정보 — 원장에 있는데 안 쓰던 컬럼 -------------------
def _join_extra(tr, ev, feats, name):
    """row_id 로 추가 피처를 붙인다. **순서가 바뀌면 group 이 조용히 틀린다.**"""
    m = pd.read_parquet(EXTRA / (name + ".parquet"))
    add = [c for c in m.columns if c != "row_id"]
    out = []
    for d in (tr, ev):
        j = d.merge(m, on="row_id", how="left")
        assert len(j) == len(d), "조인이 행 수를 바꿨다"
        assert (j["row_id"].to_numpy() == d["row_id"].to_numpy()).all(), "조인이 순서를 바꿨다"
        miss = j[add].isna().all(axis=1).mean()
        out.append(j)
    print("      %s 조인 — 추가 %d개, 전결측 행 %.1f%%" % (name, len(add), miss * 100))
    return out[0], out[1], list(feats) + add


def lgb_margin(tr, ev, feats, seed):
    """H1 — 착차(마신차) 2개를 더한다.

    가설: 착순 백분위는 마신차를 버린다. 머리 차 2착과 10마신 차 2착이 같은 값이다.
    피처: F1_behind_avg5 (최근 5출전 평균 우승마 대비 마신차)
          F1_behind_best5 (최근 5출전 중 최소)
    구성: tools/build_extra_margin.py (개인 폴더 — 원장이 거기 있다)
    검증: 누적마신 vs 기록차 상관 r=0.9932 · y_ord 상관 0.19~0.27 (<0.55) · 조인 100%
    """
    a, b, f = _join_extra(tr, ev, feats, "margin")
    return _lgb(a, b, f, seed)


def lgb_field(tr, ev, feats, seed):
    """H2 — 경쟁 강도 2개를 더한다 (73->75).

    가설: 같은 착순이라도 상대 수준이 다르다. `5,000점초과`의 2착과 `350점이하`의
          2착이 지금 같은 값이다.
    피처: F1_faced_wr_avg5  최근 5출전에서 **만난** 상대의 as-of 통산승률 평균
          F1_beaten_wr_avg5 최근 5출전에서 **이긴** 상대의 as-of 통산승률 평균
    구성: tools/build_extra_field.py — prizeCond 는 246개 이질 표기라 쓰지 않고
          원장에서 상대의 실제 실력을 직접 집계했다(그 근거는 그 파일 주석에).
    """
    a, b, f = _join_extra(tr, ev, feats, "field")
    return _lgb(a, b, f, seed)


def lgb_margin_field(tr, ev, feats, seed):
    """H1+H2 동시 (73->77). 둘이 서로를 대체하는지 보완하는지 본다."""
    a, b, f = _join_extra(tr, ev, feats, "margin")
    a, b, f = _join_extra(a, b, f, "field")
    return _lgb(a, b, f, seed)


FAMILIES = {
    "lgb_lambdarank":  lgb_lambdarank,
    "lgb_xendcg":      lgb_xendcg,
    "lgb_binary":      lgb_binary,
    "xgb_pairwise":    xgb_pairwise,
    "xgb_ndcg":        xgb_ndcg,
    "cat_querysoftmax": cat_querysoftmax,
    "cat_yetirank":    cat_yetirank,
    "hist_gb":         hist_gb,
    "rf":              rf,
    "extra_trees":     extra_trees,
    "rf_norm":         rf_norm,
    "hist_gb_norm":    hist_gb_norm,
    "lgb_speedfig":    lgb_speedfig,
    "lgb_margin":      lgb_margin,
    "lgb_field":       lgb_field,
    "lgb_margin_field": lgb_margin_field,
    "lgb_pca40":       lgb_pca40,
    "lgb_pca_rot":     lgb_pca_rot,
    "mlp_pl":          None,          # neural.py — 아래에서 채운다 (torch 지연 임포트)
}

def _load_mlp():
    from .neural import mlp_pl
    return mlp_pl

FAMILIES["mlp_pl"] = lambda *a: _load_mlp()(*a)

LABEL = {
    "lgb_lambdarank":  "LightGBM lambdarank (기준선)",
    "lgb_xendcg":      "LightGBM rank_xendcg",
    "lgb_binary":      "LightGBM binary (경쟁구조 미사용)",
    "xgb_pairwise":    "XGBoost rank:pairwise",
    "xgb_ndcg":        "XGBoost rank:ndcg",
    "cat_querysoftmax": "CatBoost QuerySoftMax (조건부로짓·트리)",
    "cat_yetirank":    "CatBoost YetiRank",
    "hist_gb":         "sklearn HistGradientBoosting",
    "rf":              "RandomForest (Lessmann 2010)",
    "extra_trees":     "ExtraTrees",
    "rf_norm":         "RandomForest + 정규화라벨",
    "hist_gb_norm":    "HistGB + 정규화라벨",
    "lgb_speedfig":    "LightGBM 스피드지수 회귀 (타깃 교체)",
    "lgb_margin":      "H1) LightGBM + 착차 2개 (73→75)",
    "lgb_field":       "H2) LightGBM + 경쟁강도 2개 (73→75)",
    "lgb_margin_field": "H1+H2) LightGBM + 착차·경쟁강도 (73→77)",
    "lgb_pca40":       "LightGBM + PCA 40성분 (압축)",
    "lgb_pca_rot":     "LightGBM + PCA 전성분 (순수 회전)",
    "mlp_pl":          "MLP + Plackett-Luce (신경망 계열)",
}
