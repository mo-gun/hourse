# -*- coding: utf-8 -*-
"""2026-09-11(금) 경주 사전 예측 — 결과를 보기 **전에** 확정해 봉인한다.

제주 7경주(56두) · 부산경남 9경주(93두) = 16경주 149두.

D-1 에 실제로 쓸 수 있는 것만 쓴다:
  · tier A 71개  D-1 확정. **이게 주 모델이다**
  · tier C 2개   F4_weather / F4_track_moist. 발주 T-10분이라 지금은 없다 → 참고 모델에서 NaN
  · tier B 2개   마체중. 발주 T-61~65분(2026-09-10 실측)이라 지금은 없다 → 안 쓴다
  · tier G 4개   배당. 발주 전 미제공 → 안 쓴다. 시장과의 비교는 **내일 채점 때** 한다

분할 경계를 고정한다(predict_0905.py 와 같은 이유) — assign_split() 은 최신 경주일 기준
상대 구간이라 9월 행을 붙이면 valid/test 창이 밀려 train 구성이 바뀐다. PIN_LAST 를
2026-08-31 로 못박아 기존 데이터셋과 멤버십을 동일하게 유지한다.

★ 예정 경기는 원장에서 ord 가 비어 load_ledger 의 완주 필터에 걸려 사라진다. 그래서
  게이트 번호를 가짜 착순으로 넣어 통과시키는데, 그 가짜가 피처로 새면 예측이 오염된다.
  --leakcheck 는 가짜 착순을 뒤집어 한 번 더 빌드해 9/11 피처가 완전히 동일한지 본다.

    python experiments/predict_20260911.py --leakcheck    # 누수 검증 포함 (2회 빌드)
    python experiments/predict_20260911.py                # 캐시 사용
"""
import argparse
import io
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(r"c:/Users/SSAFY/Desktop/주제선정")
sys.path.insert(0, str(HERE))
import schema_v2 as S          # noqa: E402
import build_v2 as B           # noqa: E402
import ebv_join as EJ          # noqa: E402

TARGET = 20260911
PIN_LAST = date(2026, 8, 31).toordinal()
OUT = HERE / "experiments" / "out"
MEETNM = {1: "서울", 2: "제주", 3: "부산경남", 4: "영천"}
SEEDS = (20260901, 20260902, 20260903)
PARAMS = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[3],
              lambdarank_truncation_level=5, learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, verbose=-1, num_threads=6)
N_ROUND = 200

# 팀 표준 피처 목록 — 단일 원천은 팀 레포다. 로컬 schema_v2 는 별건으로 마체중을
# 등록해 79/75 가 되어 있어 레포의 다른 표와 비교가 안 된다.
TEAMDS = Path(r"c:/Users/SSAFY/Desktop/말고리즘/S15P21A304/docs/dataset")
import importlib.util as _ilu   # noqa: E402
_sp = _ilu.spec_from_file_location("team_schema", TEAMDS / "schema_v2.py")
TS = _ilu.module_from_spec(_sp)
sys.modules["team_schema"] = TS
_sp.loader.exec_module(TS)

LEDGER_CSV = HERE / "data/raw/ledger_2010_2026.csv"
UPCOMING_CSV = HERE / "data/raw/upcoming_20260911.csv"


def assign_split_pinned(df):
    ho = json.load(io.open(S.HOLDOUT_FILE, encoding="utf-8"))
    days = set(ho["days"])
    df["split"] = "train"
    is_game = df["day"].isin(days)
    df.loc[is_game, "split"] = "game"
    t0 = PIN_LAST - S.TEST_WEEKS * 7
    v0 = t0 - S.VALID_WEEKS * 7
    df.loc[~is_game & (df["_ordinal"] > t0), "split"] = "test"
    df.loc[~is_game & (df["_ordinal"] > v0) & (df["_ordinal"] <= t0), "split"] = "valid"
    return df


def make_combined_ledger(reverse):
    """원장 + 9/11 출전표(가짜 착순). reverse=True 면 가짜 착순을 뒤집는다."""
    tag = "rev" if reverse else "fwd"
    out = HERE / ("data/raw/_pred_ledger_%d_%s.csv" % (TARGET, tag))
    if out.exists():
        return out
    cols = list(pd.read_csv(LEDGER_CSV, nrows=0, encoding="utf-8-sig").columns)
    up = pd.read_csv(UPCOMING_CSV, dtype=str, encoding="utf-8-sig")
    gate = pd.to_numeric(up["chulNo"])
    size = up.groupby(["meet", "rcNo"])["chulNo"].transform("size").astype(int)
    fake = (size + 1 - gate) if reverse else gate
    up["ord"] = fake.astype(int).astype(str)
    body = io.open(LEDGER_CSV, encoding="utf-8-sig").read()
    with io.open(out, "w", encoding="utf-8-sig", newline="") as f:
        f.write(body if body.endswith("\n") else body + "\n")
        up.to_csv(f, index=False, header=False, columns=cols)
    return out


def build_frame(reverse):
    tag = "rev" if reverse else "fwd"
    cache = OUT / ("_frame_%d_%s.parquet" % (TARGET, tag))
    if cache.exists():
        print("  캐시 사용: %s" % cache.name)
        return pd.read_parquet(cache)
    path = make_combined_ledger(reverse)
    orig = B.LEDGER
    B.LEDGER = str(path)
    try:
        t0 = time.time()
        df = B.load_ledger(None)
        df = assign_split_pinned(df)
        df = B.add_targets(df)
        df = B.add_speed_fig(df)
        df = B.add_sections(df)
        df = B.rolling(df)
        df, _ = EJ.attach(df)
        df = B.add_base_and_norm(df)
        print("  빌드 %.0fs" % (time.time() - t0))
    finally:
        B.LEDGER = orig
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


# ── 같은 날 누수 차단 ────────────────────────────────────────────────────
# build_v2.rolling 의 update 블록은 **모든 행에 무조건** 돈다:
#     j[0] += 1; j[1] += wv; jw.add(day, wv)      # 기수
#     jd[0] += 1; jd[1] += wv; tw.add(day, wv)    # 기수·거리, 조교사
#     if fa: s[0] += 1; s[1] += wv; s[2].add(hr)  # 부마
# 말은 하루에 한 번만 뛰므로 말 단위 피처는 안전하지만, 기수·조교사·부마는 같은 날
# 여러 경주에 나온다. 그래서 앞 경주의 (예정 경기에서는 **가짜**) 착순이 뒤 경주
# 피처에 들어간다. D-1 예측에서 그 경주들은 아직 안 뛰었으니 미래 정보다.
#
# 고치는 방법 — 재빌드가 필요 없다. load_ledger 가 (_ordinal, meet, rcNo, chulNo) 로
# 정렬하므로 어떤 기수의 **9/11 첫 등장 행**은 9/10 종료 시점 상태를 담고 있다.
# 오염은 그 이후 행에만 생긴다. 첫 등장 값을 같은 키의 모든 9/11 행에 전파한다.
#
# ⚠ 검증의 한계 — fwd/rev 비교는 **착순에 의존하는** 누수만 잡는다. 분모만 부푸는
#   누수(F2_sire_prog_n = 자마 수)는 fwd·rev 가 같은 값이라 검사에 안 걸린다.
#   걸리지 않았어도 같은 이유로 오염이므로 같이 고친다.
SAMEDAY_FIX = [
    ("F3_jk_win_rate_life", ("jkNo",)),
    ("F3_jk_win_rate_365", ("jkNo",)),
    ("F3_jk_win_rate_dist", ("jkNo", "_band")),
    ("F3_tr_win_rate_365", ("trNo",)),
    ("F2_sire_win_rate", ("F2_sire_id",)),
    ("F2_sire_prog_n", ("F2_sire_id",)),      # 검사에 안 걸리지만 같은 오염
]
RENORM = ["F3_jk_win_rate_365", "F3_tr_win_rate_365"]   # NORMALIZE_WITHIN_RACE 대상


def fix_sameday(new):
    """9/11 행의 기수·조교사·부마 집계를 '9/10 종료 시점' 값으로 되돌린다."""
    new = new.sort_values(["meet", "rcNo", "X_chulNo"], kind="stable").copy()
    new["_band"] = pd.to_numeric(new["X_rcDist"], errors="coerce").map(B.dist_band)
    n_fixed = {}
    for col, keys in SAMEDAY_FIX:
        if col not in new.columns:
            continue
        k = list(keys)
        before = new[col].to_numpy(dtype=float, copy=True)
        first = new.groupby(k, dropna=False)[col].transform("first")
        new[col] = first
        after = new[col].to_numpy(dtype=float)
        n_fixed[col] = int((~np.isclose(before, after, equal_nan=True)).sum())
    # 경주 내 정규화는 기반 값이 바뀌었으니 다시 계산한다
    for col in RENORM:
        if col not in new.columns:
            continue
        g = new.groupby("race_id")[col]
        if col + "_z" in new.columns:
            new[col + "_z"] = (new[col] - g.transform("mean")) / g.transform("std")
        if col + "_rk" in new.columns:
            new[col + "_rk"] = g.rank(pct=True)
    print("  같은 날 누수 차단 — 값이 바뀐 행 수:")
    for c, v in n_fixed.items():
        print("    %-24s %3d / %d" % (c, v, len(new)))
    return new.drop(columns=["_band"])


EXTRA = [("margin_911", "H1 착차"), ("field_911", "H2 경쟁강도")]


def join_extra(df):
    """H1·H2 추가 피처를 row_id 로 붙인다. 순서가 바뀌면 group 이 조용히 틀린다.

    9/11 행의 값이 가짜 착순에 반응하지 않는 것은 확인했다 — 말은 하루에 한 번만
    뛰므로 말 단위 as-of 집계는 같은 날 오염이 없다. (rev 원장으로 다시 만들어
    9/11 149행 전 컬럼 일치 확인)
    """
    add = []
    for name, label in EXTRA:
        m = pd.read_parquet(HERE / "dataset" / "v2" / "extra" / (name + ".parquet"))
        cols = [c for c in m.columns if c != "row_id"]
        before = df["row_id"].to_numpy()
        df = df.merge(m, on="row_id", how="left")
        assert (df["row_id"].to_numpy() == before).all(), "조인이 순서를 바꿨다"
        add += cols
        print("  %-12s 조인 %d개 · 전결측 %.1f%%"
              % (label, len(cols), df[cols].isna().all(axis=1).mean() * 100))
    return df, add


def encode_pair(a, b, cols):
    a, b = a.copy(), b.copy()
    for c in cols:
        if c in a.columns and not pd.api.types.is_numeric_dtype(a[c]):
            lv = sorted(set(a[c].dropna().astype(str)) | set(b[c].dropna().astype(str)))
            for d in (a, b):
                d[c] = pd.Categorical(d[c].astype(str), categories=lv).codes
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leakcheck", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("1) 프레임 빌드 (분할 고정 %s)" % date.fromordinal(PIN_LAST))
    print("=" * 78)
    df = build_frame(False)

    F71 = [c for c in TS.features(max_tier="A", exclude_tier=("G",)) if c in df.columns]
    F73 = [c for c in TS.features(exclude_tier=("G",)) if c in df.columns]
    assert len(F71) == 71 and len(F73) == 73, "%d/%d" % (len(F71), len(F73))

    rc = pd.to_numeric(df["rcDate"], errors="coerce")
    raw_new = df[rc == TARGET].copy()
    print("  9/11 대상 %d두 / %d경주" % (len(raw_new), raw_new["race_id"].nunique()))
    new = fix_sameday(raw_new)

    if args.leakcheck:
        print("")
        print("=" * 78)
        print("2) 누수 검증 — 가짜 착순을 뒤집어도 피처가 같은가")
        print("=" * 78)
        df2 = build_frame(True)
        rc2 = pd.to_numeric(df2["rcDate"], errors="coerce")
        raw2 = df2[rc2 == TARGET].copy()

        def diff(u, v):
            u = u.sort_values("row_id").reset_index(drop=True)
            v = v.sort_values("row_id").reset_index(drop=True)
            out = []
            for c in F73:
                x, y = u[c], v[c]
                if pd.api.types.is_numeric_dtype(x):
                    if not np.allclose(x.astype(float), y.astype(float), equal_nan=True):
                        out.append(c)
                elif not x.astype(str).equals(y.astype(str)):
                    out.append(c)
            return out

        before = diff(raw_new, raw2)
        print("  차단 전 — 가짜 착순에 반응한 컬럼 %d개:" % len(before))
        for c in before:
            print("      " + c)
        after = diff(new, fix_sameday(raw2))
        print("  차단 후 — 반응한 컬럼: %s" % (after if after else "없음 — 누수 차단 확인 OK"))
        if after:
            raise SystemExit("차단이 안 됐다. 중단.")

    print("")
    print("=" * 78)
    print("3) 무결성 — 기존 데이터셋과 같은 분할인가")
    print("=" * 78)
    ship = HERE / "dataset" / "v2" / "model"
    tr = df[df["split"] == "train"].copy()
    va = df[df["split"] == "valid"].copy()
    for nm, cur in (("train", tr), ("valid", va)):
        old = pd.read_parquet(ship / (nm + ".parquet"))
        same = "일치" if set(old["race_id"]) == set(cur["race_id"]) else "★불일치"
        print("  %-6s 기존 %7d행 / 지금 %7d행  race_id 집합 %s"
              % (nm, len(old), len(cur), same))

    print("")
    print("=" * 78)
    print("4) 학습 · 예측")
    print("=" * 78)

    def fit_predict_on(train_df, target_df, cols, label):
        x, y = encode_pair(train_df, target_df, cols)
        groups = train_df.groupby("race_id", sort=False).size().to_numpy()
        ds = lgb.Dataset(x[cols].to_numpy(float), label=x["y_rel"].to_numpy(), group=groups)
        preds = []
        for s in SEEDS:
            p = dict(PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s)
            m = lgb.train(p, ds, num_boost_round=N_ROUND)
            preds.append(m.predict(y[cols].to_numpy(float)))
        print("  %-32s train %7d행 %6d경주 · %d피처 · 시드 %d개"
              % (label, len(train_df), train_df["race_id"].nunique(), len(cols), len(SEEDS)))
        return np.mean(preds, axis=0)

    def fit_predict(train_df, cols, label):
        return fit_predict_on(train_df, new, cols, label)

    trv = df[df["split"].isin(("train", "valid"))].copy()
    new["s_A71_train"] = fit_predict(tr, F71, "A) tier A 71피처 train")
    new["s_A71_trval"] = fit_predict(trv, F71, "B) tier A 71피처 train+valid")
    new["s_F73_train"] = fit_predict(tr, F73, "C) 73피처(날씨·주로 NaN) train")

    # D) H1+H2 — 착차·경쟁강도 4개를 더한다. train 내부 시간분할 CV(28,003경주)에서
    #    logloss 가 유의하게 개선된 구성이다. top-1 은 그 표본에서도 안 움직였다.
    print("")
    tr_x, add = join_extra(tr)
    new_x, _ = join_extra(new)
    new["s_D75_train"] = fit_predict_on(tr_x, new_x, F71 + add, "D) tier A 71 + H1·H2 4개")

    print("")
    print("=" * 78)
    print("5) 예측 봉인")
    print("=" * 78)
    keep = ["race_id", "row_id", "rcDate", "meet", "rcNo", "X_chulNo", "hrNo", "hrName",
            "jkName", "trName", "X_rcDist", "X_grade", "X_dusu",
            "s_A71_train", "s_A71_trval", "s_F73_train", "s_D75_train"]
    keep = [c for c in keep if c in new.columns]
    P = new[keep].copy()
    P["meet_nm"] = pd.to_numeric(P["meet"]).map(MEETNM)
    for c in ("s_A71_train", "s_A71_trval", "s_F73_train", "s_D75_train"):
        P[c.replace("s_", "rank_")] = (P.groupby("race_id")[c]
                                       .rank(ascending=False, method="first").astype(int))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT / ("pred_%d_%s.csv" % (TARGET, stamp))
    P.sort_values(["meet", "rcNo", "rank_A71_train"]).to_csv(path, index=False,
                                                            encoding="utf-8-sig")
    print("  → %s  (%d두 / %d경주)" % (path.name, len(P), P["race_id"].nunique()))
    print("  봉인 시각 %s — 9/11 발주 전" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print("")
    print("=" * 78)
    print("6) 주 모델(A) 1착 후보 — 경주별")
    print("=" * 78)
    for (mt, rn), g in P.sort_values(["meet", "rcNo"]).groupby(["meet_nm", "rcNo"], sort=False):
        g = g.sort_values("rank_A71_train")
        top, alt = g.iloc[0], g.iloc[1]
        b1 = g[g["rank_A71_trval"] == 1]["hrName"].iloc[0]
        agree = "일치" if top["rank_A71_trval"] == 1 else ("B는 " + str(b1))
        print("  %s %2dR %4dm %-6s %2d두 | 1픽 %-9s(%d번 %s) | 2픽 %-9s | %s"
              % (mt, rn, int(top["X_rcDist"]), str(top["X_grade"]), int(top["X_dusu"]),
                 top["hrName"], int(top["X_chulNo"]), top["jkName"], alt["hrName"], agree))

    print("")
    for x, y in (("A71_train", "A71_trval"), ("A71_train", "F73_train"),
                 ("A71_train", "D75_train")):
        n = 0
        for _, g in P.groupby("race_id"):
            a = g.loc[g["rank_" + x] == 1, "row_id"].iloc[0]
            b = g.loc[g["rank_" + y] == 1, "row_id"].iloc[0]
            n += int(a == b)
        print("  1픽 일치: %s vs %s → %d/%d경주" % (x, y, n, P["race_id"].nunique()))

    print("")
    print("=" * 78)
    print("A(기준) vs D(H1+H2) 픽이 갈린 경주 — 내일 볼 지점")
    print("=" * 78)
    n_diff = 0
    for rid, g in P.groupby("race_id"):
        ga = g.loc[g["rank_A71_train"] == 1].iloc[0]
        gd = g.loc[g["rank_D75_train"] == 1].iloc[0]
        if ga["row_id"] != gd["row_id"]:
            n_diff += 1
            print("  %s %2dR  A=%-9s(%d번)  D=%-9s(%d번)"
                  % (ga["meet_nm"], int(ga["rcNo"]), ga["hrName"], int(ga["X_chulNo"]),
                     gd["hrName"], int(gd["X_chulNo"])))
    if not n_diff:
        print("  없음 — 16경주 전부 같은 말을 골랐다")
    print("")
    print("  ★ 16경주는 top-1 표준오차 ±11.8%p 다. H1+H2 효과(logloss -0.008~-0.013)를")
    print("    이 표본으로 잡으려면 |효과|/SE = 0.39 로 불가능하다. 내일 숫자는")
    print("    파이프라인 점검이고 모델 우열 판정이 아니다 — 판정은 train 내부")
    print("    시간분할 CV 28,003경주(arena.cv)에서 이미 했다.")


if __name__ == "__main__":
    main()
