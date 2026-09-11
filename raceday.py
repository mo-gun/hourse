# -*- coding: utf-8 -*-
"""경마 시행일 실시간 예측 — 하나의 스크립트로 하루를 돈다.

2026-09-11 에 실전 검증한 파이프라인(16경주)을 **날짜·OS 에 묶이지 않게** 정리한 것이다.
그날 experiments/{predict,live_predict,score,compare}_20260911.py 로 흩어져 있던 것을
합치고, 하드코딩된 Windows 경로와 고정 날짜를 걷어냈다.

## 하루 순서

    python raceday.py build  --date 20260912    # 프레임 빌드 (D-1 밤 또는 당일 아침, 3~4분)
    python raceday.py train  --date 20260912    # 모델 학습·저장 (1~2분)
    python raceday.py auto   --date 20260912    # 발주 T-10분마다 자동 예측 (하루 종일)
    python raceday.py score  --date 20260912    # 채점 (경주 끝난 뒤 아무 때나)

    python raceday.py status --date 20260912                 # 편성·T-minus 만 보기
    python raceday.py predict --date 20260912 --meet 1 --race 3   # 한 경주만
    python raceday.py predict ... --odds "1:3.2 2:5.8 ..."        # 배당을 직접 넣기

## 이 파이프라인이 쓰는 것 / 못 쓰는 것 (2026-09-11 실측)

| tier | 내용 | 가용 시점 | 쓰나 |
|---|---|---|---|
| A | 71개 — 전적·기수·혈통·거리적성 등 | D-1 확정 | O |
| B | 마체중 2개 | 발주 **T-61~65분** | O |
| C | 날씨·주로 2개 | 제주는 T-12분, **부경은 더 늦다** | 되는 대로 |
| G | 배당 4개 | **발주 -5~+1분** | **X** |

배당은 발주 직전까지 공개 채널 어디에도 없다 — 오픈 API 5종·race.kra.co.kr 73경로·
m.kra.co.kr 65경로·JS 번들 9개를 전수 확인했고, 시행일 1,168회 관측으로 확정했다.
실시간 예상배당은 onbet(발매 계정) 뒤에만 있다. 그래서 **실시간 모델은 75피처**다.

## 왜 프레임을 미리 만드나

as-of 롤링이 원장 44.6만행을 훑어 회당 3~4분이다. 경주마다 다시 하면 못 쓴다.
그래서 프레임은 하루 전에 만들어 두고, T-10분에는 API 한 번 불러
그 경주 행의 **라이브 4컬럼만 갈아끼운다** — 초 단위로 끝난다.
모델도 미리 학습한다(주 1회 재학습이 규약이므로 추론 시 재학습은 불필요).
"""
import argparse
import csv
import io
import json
import os
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent          # ★ 스크립트 위치 기준 — OS 무관
sys.path.insert(0, str(HERE))
os.chdir(HERE)                                   # build_v2 가 상대경로를 쓴다

import schema_v2 as S       # noqa: E402
import build_v2 as B        # noqa: E402
import ebv_join as EJ       # noqa: E402
from kra_client import KRA, KRAError   # noqa: E402

OUT = HERE / "out"
MODELS = OUT / "models"
LEDGER_CSV = HERE / "data/raw/ledger_2010_2026.csv"
MEETNM = {1: "서울", 2: "제주", 3: "부산경남", 4: "영천"}
SEEDS = (20260901, 20260902, 20260903)
N_ROUND = 200
PARAMS = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[3],
              lambdarank_truncation_level=5, learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, verbose=-1, num_threads=6)

# 분할 경계를 못박는다. assign_split() 은 "최신 경주일" 기준 상대 구간이라, 새 경주일을
# 원장에 붙이면 valid/test 창이 밀려 train 구성이 통째로 바뀐다. 팀 파케이가 쓴 값으로
# 고정해야 train/valid 멤버십이 기존 데이터셋과 같아진다(build 가 매번 검증한다).
PIN_LAST = date(2026, 8, 31).toordinal()

WG = ["X_wgHr", "X_wgHr_delta"]                  # tier B — 스키마상 tier=P 라 직접 넣는다
LIVE_COLS = ["X_wgHr", "X_wgHr_delta", "F4_weather", "F4_track_moist"]
SENTINEL_ORD = 91        # ord >= 91 = 미출주·취소·실격 (91~95·98·99, 원장의 2.22%)

# 같은 날 누수 — build_v2.rolling 의 update 는 모든 행에 무조건 돈다. 말은 하루에 한 번만
# 뛰므로 말 단위는 안전하지만 기수·조교사·부마는 여러 경주에 나오므로, 앞 경주의
# (예정 경기에서는 가짜) 착순이 뒤 경주 피처로 샌다. 첫 등장 값을 전파해 막는다.
SAMEDAY_FIX = [("F3_jk_win_rate_life", ("jkNo",)), ("F3_jk_win_rate_365", ("jkNo",)),
               ("F3_jk_win_rate_dist", ("jkNo", "_band")), ("F3_tr_win_rate_365", ("trNo",)),
               ("F2_sire_win_rate", ("F2_sire_id",)), ("F2_sire_prog_n", ("F2_sire_id",))]
RENORM = ["F3_jk_win_rate_365", "F3_tr_win_rate_365"]


# ── 공용 ────────────────────────────────────────────────────────────────
def feats_all():
    """77(배당 포함) / 73(실시간) / 75(73+마체중) / 79(77+마체중)."""
    f77 = list(S.features())
    f73 = list(S.features(exclude_tier=("G",)))
    assert len(f77) == 77 and len(f73) == 73, "%d/%d — 스키마가 바뀌었다" % (len(f77), len(f73))
    return f73, f73 + WG, f77, f77 + WG


def sched(kra, day, meet):
    try:
        plan = kra.fetch("race_plan", meet=meet, rc_date=str(day))
    except KRAError:
        return {}
    return {int(p["rcNo"]): str(p.get("schStTime") or "") for p in plan}


def tminus(day, hhmm, now=None):
    """발주까지 남은 분. **날짜를 포함해서** 센다 (시:분만 쓰면 하루 전엔 전부 '지남')."""
    if len(hhmm) != 4 or not hhmm.isdigit():
        return None
    now = now or datetime.now()
    post = datetime(day // 10000, day // 100 % 100, day % 100, int(hhmm[:2]), int(hhmm[2:]))
    return int(round((post - now).total_seconds() / 60))


def log_path(day):
    return OUT / ("live_pred_%d.csv" % day)


def append_log(day, rec):
    """컬럼이 늘어도 안 깨지게 합집합으로 다시 쓴다 (단순 append 로 한 번 깨뜨렸다)."""
    p = log_path(day)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        old = pd.read_csv(p, encoding="utf-8-sig")
        cols = list(dict.fromkeys(list(old.columns) + list(rec.columns)))
        pd.concat([old.reindex(columns=cols), rec.reindex(columns=cols)],
                  ignore_index=True).to_csv(p, index=False, encoding="utf-8-sig")
    else:
        rec.to_csv(p, index=False, encoding="utf-8-sig")


# ── 프레임 ──────────────────────────────────────────────────────────────
def fetch_upcoming(kra, day):
    """그 날 출전표를 원장 API 에서 받는다 (예정 경기도 행이 있고 결과 필드만 빈다)."""
    cols = list(pd.read_csv(LEDGER_CSV, nrows=0, encoding="utf-8-sig").columns)
    rows = []
    for meet in (1, 2, 3):
        try:
            rows += kra.fetch("race_result", rc_date=str(day), meet=meet)
        except KRAError:
            pass
    if not rows:
        raise SystemExit("%d 에 편성된 경주가 없다." % day)
    return pd.DataFrame(rows), cols


def make_combined_ledger(kra, day, reverse):
    """원장 + 그 날 출전표. 예정 경기는 ord 가 비어 완주 필터에 걸리므로 게이트 번호를
    가짜 착순으로 넣어 통과시킨다. --leakcheck 가 이 가짜가 새는지 검증한다."""
    tag = "rev" if reverse else "fwd"
    out = HERE / ("data/raw/_pred_ledger_%d_%s.csv" % (day, tag))
    if out.exists():
        return out
    up, cols = fetch_upcoming(kra, day)
    gate = pd.to_numeric(up["chulNo"])
    size = up.groupby(["meet", "rcNo"])["chulNo"].transform("size").astype(int)
    up["ord"] = ((size + 1 - gate) if reverse else gate).astype(int).astype(str)
    body = io.open(LEDGER_CSV, encoding="utf-8-sig").read()
    with io.open(out, "w", encoding="utf-8-sig", newline="") as f:
        f.write(body if body.endswith("\n") else body + "\n")
        up.to_csv(f, index=False, header=False, columns=cols)
    return out


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


def build_frame(kra, day, reverse=False):
    cache = OUT / ("_frame_%d_%s.parquet" % (day, "rev" if reverse else "fwd"))
    if cache.exists():
        return pd.read_parquet(cache)
    path = make_combined_ledger(kra, day, reverse)
    orig, B.LEDGER = B.LEDGER, str(path)
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


def fix_sameday(g):
    g = g.sort_values(["meet", "rcNo", "X_chulNo"], kind="stable").copy()
    g["_band"] = pd.to_numeric(g["X_rcDist"], errors="coerce").map(B.dist_band)
    for col, keys in SAMEDAY_FIX:
        if col in g.columns:
            g[col] = g.groupby(list(keys), dropna=False)[col].transform("first")
    for col in RENORM:
        if col in g.columns:
            gr = g.groupby("race_id")[col]
            if col + "_z" in g.columns:
                g[col + "_z"] = (g[col] - gr.transform("mean")) / gr.transform("std")
            if col + "_rk" in g.columns:
                g[col + "_rk"] = gr.rank(pct=True)
    return g.drop(columns=["_band"])


def day_frame(kra, day):
    df = build_frame(kra, day)
    rc = pd.to_numeric(df["rcDate"], errors="coerce")
    return df[df["split"] == "train"].copy(), fix_sameday(df[rc == day].copy())


# ── 당일 갱신 ───────────────────────────────────────────────────────────
def refresh_entries(kra, day, meet, rc_no, g):
    """출전취소를 반영하고 경주 내 값(두수·게이트비·선행마수·_z/_rk 26개)을 재계산한다."""
    try:
        rows = kra.fetch("race_result", rc_date=str(day), meet=meet, rc_no=rc_no)
    except KRAError:
        rows = []
    out = {int(r["chulNo"]) for r in rows
           if str(r.get("ord", "")).strip().isdigit() and int(r["ord"]) >= SENTINEL_ORD}
    try:
        for r in kra.fetch("cancel_info", rc_date=str(day), meet=meet, rc_no=rc_no):
            out.add(int(r["chulNo"]))
    except KRAError:
        pass
    try:
        jc = kra.fetch("jockey_change", rc_date=str(day), meet=meet, rc_no=rc_no)
    except KRAError:
        jc = []
    if jc:
        print("  ⚠ 기수변경 %d건 — F3_jk_* 를 정확히 재계산할 수 없다(as-of 누적 필요). "
              "값은 그대로 두고 표시만 한다." % len(jc))
    if not out:
        return g, 0
    gates = g["X_chulNo"].astype(int)
    dropped = g[gates.isin(out)][["hrName", "X_chulNo"]]
    g = g[~gates.isin(out)].copy()
    n = len(g)
    if n == 0:
        return g, len(dropped)
    g["X_dusu"] = n
    g["X_gate_rel"] = g["X_chulNo"].astype(float) / n
    if {"F5_early_pos", "F5_race_n_front"} <= set(g.columns):
        g["F5_race_n_front"] = float((pd.to_numeric(g["F5_early_pos"], errors="coerce") < 0.25).sum())
    n_re = 0
    for c in S.NORMALIZE_WITHIN_RACE:
        if c not in g.columns:
            continue
        v = pd.to_numeric(g[c], errors="coerce")
        sd = v.std()
        if c + "_z" in g.columns:
            g[c + "_z"] = (v - v.mean()) / (sd if sd and sd == sd else np.nan)
            n_re += 1
        if c + "_rk" in g.columns:
            g[c + "_rk"] = v.rank(pct=True)
            n_re += 1
    print("  ★ 출전취소 %d두 — %s → 두수 %d · 경주내 재계산 %d컬럼"
          % (len(dropped), ", ".join("%s(%d번)" % (r.hrName, int(r.X_chulNo))
                                     for r in dropped.itertuples()), n, n_re))
    return g, len(dropped)


def fetch_live(kra, day, meet, rc_no):
    try:
        rows = kra.fetch("race_result", rc_date=str(day), meet=meet, rc_no=rc_no)
    except KRAError:
        return {}
    out = {}
    for r in rows:
        wg, wgd = B.parse_wg(r.get("wgHr"))
        _, moist = B.parse_track(r.get("track"))
        wx = str(r.get("weather", "")).strip()
        out[int(r["chulNo"])] = dict(X_wgHr=wg, X_wgHr_delta=wgd,
                                     F4_weather=(np.nan if wx in ("", "-") else wx),
                                     F4_track_moist=moist)
    return out


# ── 시장 (배당을 직접 넣을 때) ──────────────────────────────────────────
def parse_odds(text, gates):
    toks = [t for t in re.split(r"[,\s]+", text.strip()) if t]
    if all(":" in t for t in toks):
        return {int(t.split(":")[0]): float(t.split(":")[1]) for t in toks}
    if len(toks) != len(gates):
        raise SystemExit("배당 %d개 vs 출전 %d두 — 'N:배당' 형식을 쓰거나 두수를 맞춰라"
                         % (len(toks), len(gates)))
    return {int(g): float(t) for g, t in zip(sorted(gates), toks)}


def apply_market(g, odds):
    """단승배당 → F6 4개. build_v2 와 같은 식이어야 한다."""
    o = g["X_chulNo"].astype(int).map(odds).astype(float)
    o[(o >= S.ODDS_NONE) | (o <= 0)] = np.nan
    inv = 1.0 / o
    p = inv / inv.sum(skipna=True)
    g["F6_mkt_prob"] = p.to_numpy()
    g["F6_mkt_rank"] = o.rank(method="min").to_numpy()
    sd = p.std()
    g["F6_mkt_prob_z"] = ((p - p.mean()) / (sd if sd and sd == sd else np.nan)).to_numpy()
    pc = p.clip(lower=1e-9)
    g["F6_field_entropy"] = float(-(pc * np.log(pc)).sum())
    print("  시장 피처 — 배당 %d/%d두 · 엔트로피 %.3f"
          % (int(o.notna().sum()), len(g), g["F6_field_entropy"].iloc[0]))
    return g


# ── 모델 ────────────────────────────────────────────────────────────────
def train_one(tr, new, feats, tag):
    x, y = encode_pair(tr, new, feats)
    ds = lgb.Dataset(x[feats].to_numpy(float), label=tr["y_rel"].to_numpy(),
                     group=tr.groupby("race_id", sort=False).size().to_numpy())
    MODELS.mkdir(parents=True, exist_ok=True)
    for s in SEEDS:
        m = lgb.train(dict(PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s),
                      ds, num_boost_round=N_ROUND)
        # LightGBM 의 save_model 은 **경로에 한글이 있으면 실패**한다(Windows C++ 계층).
        # 문자열로 뽑아 파이썬이 쓰고, 읽을 때도 model_str 로 넣는다.
        io.open(MODELS / ("m%s_%d.txt" % (tag, s)), "w",
                encoding="utf-8").write(m.model_to_string())
    lv = {c: sorted(set(tr[c].dropna().astype(str)) | set(new[c].dropna().astype(str)))
          for c in feats if c in tr.columns and not pd.api.types.is_numeric_dtype(tr[c])}
    pd.to_pickle({"feats": feats, "levels": lv}, MODELS / ("m%s_meta.pkl" % tag))
    print("  %s피처 저장 — train %d행 %d경주 · 시드 %d개"
          % (tag, len(tr), tr["race_id"].nunique(), len(SEEDS)))


def load_models(tag):
    meta = pd.read_pickle(MODELS / ("m%s_meta.pkl" % tag))
    ms = [lgb.Booster(model_str=io.open(MODELS / ("m%s_%d.txt" % (tag, s)),
                                        encoding="utf-8").read()) for s in SEEDS]
    return meta, ms


def encode_pair(a, b, cols):
    a, b = a.copy(), b.copy()
    for c in cols:
        if c in a.columns and not pd.api.types.is_numeric_dtype(a[c]):
            lv = sorted(set(a[c].dropna().astype(str)) | set(b[c].dropna().astype(str)))
            for d in (a, b):
                d[c] = pd.Categorical(d[c].astype(str), categories=lv).codes
    return a, b


# ── 명령 ────────────────────────────────────────────────────────────────
def cmd_build(a):
    kra = KRA()
    print("=" * 72)
    print("프레임 빌드 %d (분할 고정 %s)" % (a.date, date.fromordinal(PIN_LAST)))
    print("=" * 72)
    df = build_frame(kra, a.date)
    rc = pd.to_numeric(df["rcDate"], errors="coerce")
    print("  대상 %d두 / %d경주" % ((rc == a.date).sum(),
                                  df[rc == a.date]["race_id"].nunique()))
    ship = HERE / "dataset" / "v2" / "model"
    for nm in ("train", "valid"):
        cur = df[df["split"] == nm]
        if (ship / (nm + ".parquet")).exists():
            old = pd.read_parquet(ship / (nm + ".parquet"))
            ok = "일치" if set(old["race_id"]) == set(cur["race_id"]) else "★불일치"
            print("  %-6s 기존 %7d행 / 지금 %7d행 · race_id %s" % (nm, len(old), len(cur), ok))
    if a.leakcheck:
        print("\n누수 검증 — 가짜 착순을 뒤집어도 피처가 같은가")
        d2 = build_frame(kra, a.date, reverse=True)
        f73 = list(S.features(exclude_tier=("G",)))
        A = fix_sameday(df[rc == a.date].copy()).sort_values("row_id").reset_index(drop=True)
        rc2 = pd.to_numeric(d2["rcDate"], errors="coerce")
        Bf = fix_sameday(d2[rc2 == a.date].copy()).sort_values("row_id").reset_index(drop=True)
        bad = [c for c in f73
               if not (np.allclose(pd.to_numeric(A[c], errors="coerce").astype(float),
                                   pd.to_numeric(Bf[c], errors="coerce").astype(float),
                                   equal_nan=True)
                       if pd.api.types.is_numeric_dtype(A[c])
                       else A[c].astype(str).equals(Bf[c].astype(str)))]
        print("  반응한 컬럼: %s" % (bad if bad else "없음 — 누수 없음 OK"))


def cmd_train(a):
    kra = KRA()
    tr, new = day_frame(kra, a.date)
    _, f75, _, f79 = feats_all()
    f75 = [c for c in f75 if c in tr.columns]
    f79 = [c for c in f79 if c in tr.columns]
    train_one(tr, new, f75, "75")
    train_one(tr, new, f79, "79")


def predict_one(kra, day, meet, rc_no, odds_text=None, quiet=False):
    now = datetime.now()
    sc = sched(kra, day, meet)
    tm = tminus(day, sc.get(rc_no, ""), now)
    print("=== %d %s %dR · 관측 %s · 발주 %s (T%+d분) ==="
          % (day, MEETNM.get(meet, meet), rc_no, now.strftime("%H:%M"),
             sc.get(rc_no, "?"), tm if tm is not None else 0))
    if tm is not None and tm <= 0:
        print("  ⚠ 발주 시각이 지났다 — 사전 예측이 아니다. 기록에 그렇게 남는다.")
    tag = "79" if odds_text else "75"
    meta, models = load_models(tag)
    _, new = day_frame(kra, day)
    g = new[(pd.to_numeric(new["meet"]) == meet) & (pd.to_numeric(new["rcNo"]) == rc_no)].copy()
    if g.empty:
        raise SystemExit("그 경주가 프레임에 없다.")
    g, n_scr = refresh_entries(kra, day, meet, rc_no, g)
    if g.empty:
        raise SystemExit("전원 취소된 경주다.")
    live = fetch_live(kra, day, meet, rc_no)
    n_live = {c: 0 for c in LIVE_COLS}
    for i, row in g.iterrows():
        v = live.get(int(row["X_chulNo"]))
        if not v:
            continue
        for c in LIVE_COLS:
            if v[c] == v[c]:
                g.at[i, c] = v[c]
                n_live[c] += 1
    print("  라이브: " + " · ".join("%s %d/%d" % (c.replace("X_", "").replace("F4_", ""),
                                                 n_live[c], len(g)) for c in LIVE_COLS))
    if odds_text:
        g = apply_market(g, parse_odds(odds_text, g["X_chulNo"].astype(int).tolist()))
    x = g.copy()
    for c, lv in meta["levels"].items():
        x[c] = pd.Categorical(x[c].astype(str), categories=lv).codes
    g["score"] = np.mean([m.predict(x[meta["feats"]].to_numpy(float)) for m in models], axis=0)
    g = g.sort_values("score", ascending=False).reset_index(drop=True)
    g["rank"] = np.arange(1, len(g) + 1)
    print("")
    print("  %-3s %-12s%5s  %-9s%7s%9s" % ("순위", "마명", "게이트", "기수", "마체중", "점수"))
    print("  " + "-" * 52)
    for r in g.itertuples():
        wg = "-" if r.X_wgHr != r.X_wgHr else "%.0f" % r.X_wgHr
        print("  %-3d %-12s%5d  %-9s%7s%9.3f"
              % (r.rank, r.hrName, int(r.X_chulNo), str(r.jkName)[:8], wg, r.score))
    keep = [c for c in ["race_id", "row_id", "meet", "rcNo", "X_chulNo", "hrNo", "hrName",
                        "jkName", "trName", "X_rcDist", "X_grade", "X_dusu", "X_wgHr",
                        "F4_weather", "F4_track_moist", "score", "rank"] if c in g.columns]
    rec = g[keep].copy()
    rec["obs"] = now.strftime("%Y-%m-%d %H:%M:%S")
    rec["tminus"] = tm
    rec["arm"] = tag
    rec["n_live_wgHr"] = n_live["X_wgHr"]
    rec["n_live_weather"] = n_live["F4_weather"]
    rec["n_scratched"] = n_scr
    rec["odds_src"] = "manual" if odds_text else ""
    append_log(day, rec)
    print("\n  → %s" % log_path(day).name)
    return g


def cmd_predict(a):
    predict_one(KRA(), a.date, a.meet, a.race, a.odds)


def cmd_status(a):
    kra = KRA()
    now = datetime.now()
    print("=== %d 현재 %s ===" % (a.date, now.strftime("%H:%M")))
    for meet in (1, 2, 3):
        for rn, hh in sorted(sched(kra, a.date, meet).items()):
            tm = tminus(a.date, hh, now)
            st = "지남" if (tm is not None and tm <= 0) else \
                 ("★T-10 창" if tm is not None and tm <= 12 else "대기")
            print("  %-6s %2dR  %s:%s  T%+6s분  %s"
                  % (MEETNM[meet], rn, hh[:2], hh[2:], tm if tm is not None else "?", st))


def cmd_auto(a):
    kra = KRA()
    plan = {}
    for meet in (1, 2, 3):
        for rn, hh in sched(kra, a.date, meet).items():
            if hh:
                plan[(meet, rn)] = hh
    if not plan:
        raise SystemExit("%d 에 편성된 경주가 없다." % a.date)
    print("편성 %d경주 · T-%d~%d분 창에서 발사 · %d초마다 확인"
          % (len(plan), a.window, a.floor, a.poll))
    while True:
        done = set()
        p = log_path(a.date)
        if p.exists():
            try:
                L = pd.read_csv(p, encoding="utf-8-sig")
                done = {(int(x), int(y)) for x, y in zip(L["meet"], L["rcNo"])}
            except Exception as e:
                print("  ⚠ 로그 읽기 실패(%s) — 계속한다" % type(e).__name__)
        todo = {k: v for k, v in plan.items() if k not in done}
        if not todo:
            print("전 경주 예측 완료.")
            return
        now = datetime.now()
        fired = False
        for (meet, rn), hh in sorted(todo.items(), key=lambda kv: tminus(a.date, kv[1]) or 0):
            tm = tminus(a.date, hh, now)
            if tm is None:
                continue
            if tm <= a.window:
                if tm < a.floor:
                    print("\n  ⚠ %s %dR 은 창을 놓쳤다 (지금 T%+d분). 늦게 기록한다."
                          % (MEETNM[meet], rn, tm))
                print("")
                try:
                    predict_one(kra, a.date, meet, rn)
                except SystemExit as e:
                    print("  건너뜀: %s" % e)
                fired = True
        if not fired:
            nxt = min((tminus(a.date, v, now) or 9999) for v in todo.values())
            print("  %s · 남은 %d경주 · 다음 발사까지 %d분"
                  % (now.strftime("%H:%M"), len(todo), max(nxt - a.window, 0)), flush=True)
        time.sleep(a.poll)


def cmd_score(a):
    kra = KRA()
    p = log_path(a.date)
    if not p.exists():
        raise SystemExit("예측 기록이 없다: %s" % p)
    L = pd.read_csv(p, encoding="utf-8-sig")
    rows = []
    for meet in (1, 2, 3):
        try:
            R = pd.DataFrame(kra.fetch("race_result", rc_date=str(a.date), meet=meet))
        except KRAError:
            continue
        if R.empty:
            continue
        R["r"] = pd.to_numeric(R["rcNo"]); R["c"] = pd.to_numeric(R["chulNo"])
        R["o"] = pd.to_numeric(R["ord"], errors="coerce")
        R["od"] = pd.to_numeric(R["winOdds"], errors="coerce")
        R.loc[(R["od"] <= 0) | (R["od"] >= S.ODDS_NONE), "od"] = np.nan
        sc = sched(kra, a.date, meet)
        for rn, g in R.groupby("r"):
            fin = g[g["o"].between(1, 16)]
            P = L[(L["meet"] == meet) & (L["rcNo"] == rn)]
            if not len(P):
                continue
            tm = P[P["tminus"] > 0]["tminus"].max() if (P["tminus"] > 0).any() else P["tminus"].max()
            P = P[P["tminus"] == tm].sort_values("rank")
            t3 = list(P.head(3)["hrName"].astype(str).str.strip())
            if len(fin) < 1 or not (fin["o"] == 1).any() or not g["od"].notna().any():
                rows.append(dict(장=MEETNM[meet], R=int(rn), 발주=sc.get(rn, ""), 두수=len(P),
                                 픽1=t3[0], 일착="집계중", 배당="", 적중="", 연승="", 삼픽="",
                                 순위=0, 시장="", 갈림=""))
                continue
            w = fin[fin["o"] == 1]
            wn = str(w["hrName"].iloc[0]).strip()
            plc = set(fin[fin["o"] <= 3]["hrName"].astype(str).str.strip())
            hit = P["hrName"].astype(str).str.strip() == wn
            mk = g[g["od"].notna()].nsmallest(1, "od")
            mkn = str(mk["hrName"].iloc[0]).strip() if len(mk) else "?"
            rows.append(dict(
                장=MEETNM[meet], R=int(rn), 발주=sc.get(rn, ""), 두수=len(fin), 픽1=t3[0],
                일착=wn, 배당=("%.1f" % w["od"].iloc[0]) if pd.notna(w["od"].iloc[0]) else "-",
                적중=("O" if t3[0] == wn else "X"), 연승=("O" if t3[0] in plc else "X"),
                삼픽=("O" if wn in t3 else "X"),
                순위=int(P[hit]["rank"].iloc[0]) if hit.any() else 0,
                시장=mkn, 갈림=("★" if t3[0] != mkn else "")))
    if not rows:
        raise SystemExit("채점할 경주가 없다.")
    T = pd.DataFrame(rows).sort_values(["발주", "장"])
    print(T.to_string(index=False))
    D = T[T["적중"] != ""]
    n = len(D)
    if not n:
        return
    print("\n== 채점 %d경주 ==" % n)
    for c, lab, e in (("적중", "1착 적중", 34), ("연승", "1픽 연승권", 63), ("삼픽", "3픽중 1착", 66)):
        h = int((D[c] == "O").sum())
        print("  %-12s%2d/%d = %5.1f%%   기대 %d%%" % (lab, h, n, h / n * 100, e))
    print("  표준오차 ±%.1f%%p — 이 표본으로 우열을 판정하지 말 것"
          % (np.sqrt(.34 * .66 / n) * 100))
    print("  시장과 갈림 %d/%d" % (int((D["갈림"] == "★").sum()), n))
    out = OUT / ("scored_%d.csv" % a.date)
    T.to_csv(out, index=False, encoding="utf-8-sig")
    print("  → %s" % out.name)


def main():
    ap = argparse.ArgumentParser(description="경마 시행일 실시간 예측")
    sub = ap.add_subparsers(dest="cmd", required=True)
    today = int(datetime.now().strftime("%Y%m%d"))
    for name, fn, helptext in (("build", cmd_build, "프레임 빌드 (3~4분)"),
                               ("train", cmd_train, "모델 학습·저장"),
                               ("status", cmd_status, "편성·T-minus"),
                               ("auto", cmd_auto, "T-10분마다 자동 예측"),
                               ("predict", cmd_predict, "한 경주 예측"),
                               ("score", cmd_score, "채점")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--date", type=int, default=today, help="YYYYMMDD (기본: 오늘)")
        p.set_defaults(fn=fn)
        if name == "build":
            p.add_argument("--leakcheck", action="store_true", help="가짜 착순 누수 검증(2회 빌드)")
        if name == "predict":
            p.add_argument("--meet", type=int, required=True, choices=(1, 2, 3))
            p.add_argument("--race", type=int, required=True)
            p.add_argument("--odds", default=None,
                           help='단승배당 직접 입력 "1:3.2 2:5.8 ..." — 주면 79피처 모델')
        if name == "auto":
            p.add_argument("--window", type=int, default=12)
            p.add_argument("--floor", type=int, default=6)
            p.add_argument("--poll", type=int, default=60)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
