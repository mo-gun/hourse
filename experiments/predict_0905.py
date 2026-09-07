# -*- coding: utf-8 -*-
"""
2026-09-05 경주 예측 + 채점.

9/5 는 test 종료(2026-08-30) 이후라 어느 분할에도 속하지 않은 완전 미개봉 구간이다.
경주가 이미 끝났으므로 예측하고 바로 채점할 수 있다.

핵심 주의 — 분할 경계를 고정한다.
  build_v2.assign_split() 은 `last = 최신 경주일` 기준 상대 구간이다. 9월 행을 그냥 붙이면
  valid/test 창이 6일 밀려 train 구성이 바뀐다. 그래서 last 를 2026-08-31 로 **고정**한다.
  그러면 train/valid/test 멤버십이 기존 데이터셋과 동일하고, 9월 행만 test 뒤에 덧붙는다.

    python experiments/predict_0905.py
"""
import sys, io, json, time
from datetime import date
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(r"c:/Users/SSAFY/Desktop/주제선정")
sys.path.insert(0, str(HERE))
import schema_v2 as S                                        # noqa: E402
import build_v2 as B                                         # noqa: E402
import ebv_join as EJ                                        # noqa: E402

TARGET = 20260905
DAYS3 = (20260904, 20260905, 20260906)        # 표본 보강용 — 9/5 전후 개최일
PIN_LAST = date(2026, 8, 31).toordinal()      # 기존 데이터셋이 쓴 최신일
CACHE = HERE / "experiments" / "out" / "_frame_0905.parquet"
MEETNM = {1: "서울", 2: "제주", 3: "부산경남", 4: "영천"}

# 팀 표준 피처 목록 — 로컬 schema_v2 는 구버전이라 v2.2.0 비활성 5개가 남아 있다.
# docs/dataset/schema_v2.py 가 단일 원천이므로 거기서 가져온다.
TEAMDS = Path(r"c:/Users/SSAFY/Desktop/말고리즘/S15P21A304/docs/dataset")
import importlib.util as _ilu
_sp = _ilu.spec_from_file_location("team_schema", TEAMDS / "schema_v2.py")
TS = _ilu.module_from_spec(_sp); sys.modules["team_schema"] = TS; _sp.loader.exec_module(TS)
SEEDS = (20260901, 20260902, 20260903)
PARAMS = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[3],
              lambdarank_truncation_level=5, learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, verbose=-1, num_threads=4)
N_ROUND = 200


def assign_split_pinned(df):
    """build_v2.assign_split 과 같지만 last 를 고정한다."""
    ho = json.load(io.open(S.HOLDOUT_FILE, encoding="utf-8"))
    days = set(ho["days"])
    df["split"] = "train"
    is_game = df["day"].isin(days)
    df.loc[is_game, "split"] = "game"
    t0 = PIN_LAST - S.TEST_WEEKS * 7
    v0 = t0 - S.VALID_WEEKS * 7
    df.loc[~is_game & (df["_ordinal"] > t0), "split"] = "test"
    df.loc[~is_game & (df["_ordinal"] > v0) & (df["_ordinal"] <= t0), "split"] = "valid"
    print(f"  분할 고정: train ≤ {date.fromordinal(v0)} · valid ~ {date.fromordinal(t0)} · test 이후")
    return df


print("=" * 78)
print("1) 원장 로드 + 피처 생성 (분할 경계 고정)")
print("=" * 78)
t0 = time.time()
if CACHE.exists():
    print(f"  캐시 사용: {CACHE.name}")
    df = pd.read_parquet(CACHE)
else:
    df = B.load_ledger(None)
    df = assign_split_pinned(df)
    df = B.add_targets(df)
    df = B.add_speed_fig(df)
    df = B.add_sections(df)
    df = B.rolling(df)
    df, added = EJ.attach(df)
    df = B.add_base_and_norm(df)
    df.to_parquet(CACHE, index=False)
print(f"  완료 {time.time()-t0:.0f}s")

feats = [c for c in S.features() if c in df.columns]
idx = [c["name"] for c in S.INDEX if c["name"] in df.columns]
tgt = [c["name"] for c in S.TARGETS if c["name"] in df.columns]
keep = list(dict.fromkeys(idx + tgt + feats + ["split", "rcDate", "meet", "rcNo", "hrName",
                                               "jkName", "winOdds", "y_ord"]))
keep = [c for c in keep if c in df.columns]
D = df[keep].copy()
for c in S.CATEGORICAL:
    if c in D.columns:
        D[c] = D[c].astype("category")

rc = pd.to_numeric(D["rcDate"], errors="coerce")
tr = D[D["split"] == "train"].copy()
va = D[D["split"] == "valid"].copy()
new = D[rc == TARGET].copy()

print("\n" + "=" * 78)
print("2) 무결성 확인 — 기존 데이터셋과 같은 분할인가")
print("=" * 78)
ship = HERE / "dataset" / "v2" / "model"
for nm, cur in (("train", tr), ("valid", va)):
    old = pd.read_parquet(ship / f"{nm}.parquet")
    same_n = len(old) == len(cur)
    same_r = set(old.race_id) == set(cur.race_id)
    print(f"  {nm:6s} 기존 {len(old):>7,}행 / 지금 {len(cur):>7,}행  "
          f"행수 {'일치' if same_n else '불일치'} · race_id 집합 {'일치' if same_r else '불일치'}")
print(f"  9/5    {len(new):>7,}행 {new.race_id.nunique()}경주  ← 예측 대상 (어느 분할에도 없음)")

print("\n" + "=" * 78)
print(f"3) 학습 후 {TARGET} 예측")
print("=" * 78)


def encode_pair(a, b, cols):
    a, b = a.copy(), b.copy()
    for c in cols:
        if c in a.columns and not pd.api.types.is_numeric_dtype(a[c]):
            lv = sorted(set(a[c].dropna().astype(str)) | set(b[c].dropna().astype(str)))
            for d in (a, b):
                d[c] = pd.Categorical(d[c].astype(str), categories=lv).codes
    return a, b


def groups(d):
    return d.groupby("race_id", sort=False).size().to_numpy()


def fit_predict_on(target, cols):
    a, b = encode_pair(tr, target, cols)
    ds = lgb.Dataset(a[cols].to_numpy(float), label=a["y_rel"].to_numpy(), group=groups(a))
    ps = []
    for s in SEEDS:
        p = dict(PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s)
        m = lgb.train(p, ds, num_boost_round=N_ROUND)
        ps.append(m.predict(b[cols].to_numpy(float)))
    return np.mean(ps, axis=0)


def fit_predict(cols):
    return fit_predict_on(new, cols)


F77 = [c for c in TS.features() if c in tr.columns]
F73 = [c for c in TS.features(exclude_tier=("G",)) if c in tr.columns]
assert len(F77) == 77 and len(F73) == 73, f"팀 기준과 다르다: {len(F77)}/{len(F73)}"
print(f"  train {len(tr):,}행 {tr.race_id.nunique():,}경주 · 피처 {len(F77)} / {len(F73)}")
new["s77"] = fit_predict(F77)
new["s73"] = fit_predict(F73)
new["smkt"] = -pd.to_numeric(new["F6_mkt_rank"], errors="coerce").to_numpy(float)


def score_df(frame, col):
    t = frame[["race_id", col, "y_win", "y_plc"]].copy()
    t = t.sort_values(["race_id", col], ascending=[True, False], kind="stable")
    g = t.groupby("race_id", sort=False)
    p1 = g.head(1)
    return dict(win=p1.y_win.mean() * 100, plc=p1.y_plc.mean() * 100,
                top3=g.head(3).groupby("race_id").y_win.max().mean() * 100,
                n=int(t.race_id.nunique()))


def score(col):
    return score_df(new, col)


print("\n" + "=" * 78)
print(f"4) 채점 — {TARGET} 실제 결과 대조")
print("=" * 78)
print(f"\n{'모델':<26}{'1착 적중':>9}{'연승권':>9}{'3두중 1착':>10}")
print("-" * 56)
res = {}
for col, nm in (("s77", "LightGBM 77피처"), ("s73", "LightGBM 73피처 (실시간)"),
                ("smkt", "시장 (인기 1위마)")):
    m = score(col); res[nm] = m
    print(f"{nm:<26}{m['win']:>8.1f}%{m['plc']:>8.1f}%{m['top3']:>9.1f}%")
print("-" * 56)
print(f"경주 {res['시장 (인기 1위마)']['n']}개 — 표본이 작아 ±12%p 수준의 흔들림이 정상이다\n")

print("=" * 78)
print("5) 경주별 예측 (73피처 실시간 조건)")
print("=" * 78)
out_rows = []
for rid, g in new.groupby("race_id", sort=False):
    g = g.sort_values("s73", ascending=False)
    pick = g.iloc[0]
    winner = g[g.y_win == 1]
    wname = winner.iloc[0]["hrName"] if len(winner) else "?"
    wodds = float(winner.iloc[0]["winOdds"]) if len(winner) else float("nan")
    hit = "◎" if int(pick.y_win) == 1 else ("○" if int(pick.y_plc) == 1 else "×")
    out_rows.append(dict(race=str(rid), meet=MEETNM.get(int(pick["meet"]), str(pick["meet"])), rcNo=int(pick["rcNo"]),
                         pick=str(pick["hrName"]), pick_odds=float(pick["winOdds"]),
                         pick_ord=int(pick["y_ord"]), mkt_rank=int(pick["F6_mkt_rank"]),
                         winner=wname, win_odds=wodds, hit=hit))
O = pd.DataFrame(out_rows)
print(f"\n{'경마장':<6}{'R':>3}  {'예측 1순위':<12}{'배당':>6}{'착순':>5}{'인기':>5}  "
      f"{'실제 1착':<12}{'배당':>6}  판정")
print("-" * 78)
for _, r in O.iterrows():
    print(f"{r['meet']:<6}{r['rcNo']:>3}  {r['pick']:<12}{r['pick_odds']:>6.1f}"
          f"{r['pick_ord']:>5}{r['mkt_rank']:>5}  {r['winner']:<12}{r['win_odds']:>6.1f}  {r['hit']}")
print("-" * 78)
print(f"◎ 1착 적중 {(O.hit=='◎').sum()} · ○ 연승권 {(O.hit=='○').sum()} · × 실패 {(O.hit=='×').sum()}")

print("\n" + "=" * 78)
print("6) 표본 보강 — 9/4~9/6 세 개최일 합산")
print("=" * 78)
big = D[rc.isin(DAYS3)].copy()
big["s77"] = fit_predict_on(big, F77)
big["s73"] = fit_predict_on(big, F73)
big["smkt"] = -pd.to_numeric(big["F6_mkt_rank"], errors="coerce").to_numpy(float)
by_day = big.assign(_d=pd.to_numeric(big["rcDate"], errors="coerce"))
print(f"\n{'모델':<26}{'1착 적중':>9}{'연승권':>9}{'3두중 1착':>10}")
print("-" * 56)
res3 = {}
for col, nm in (("s77", "LightGBM 77피처"), ("s73", "LightGBM 73피처 (실시간)"),
                ("smkt", "시장 (인기 1위마)")):
    m = score_df(big, col); res3[nm] = m
    print(f"{nm:<26}{m['win']:>8.1f}%{m['plc']:>8.1f}%{m['top3']:>9.1f}%")
print("-" * 56)
n3 = res3["시장 (인기 1위마)"]["n"]
se3 = (0.35 * 0.65 / n3) ** 0.5 * 100
print(f"경주 {n3}개 · 1착 적중률 표준오차 ±{se3:.1f}%p\n")
print(f"{'개최일':<12}{'경주':>5}{'77피처':>9}{'73피처':>9}{'시장':>9}")
print("-" * 46)
for d, g in by_day.groupby("_d"):
    a, b, c = score_df(g, "s77"), score_df(g, "s73"), score_df(g, "smkt")
    print(f"{int(d):<12}{a['n']:>5}{a['win']:>8.1f}%{b['win']:>8.1f}%{c['win']:>8.1f}%")

io.open(HERE / "experiments" / "out" / "predict_0905.json", "w", encoding="utf-8").write(
    json.dumps({"scores_0905": res, "scores_3days": res3, "races": out_rows}, ensure_ascii=False, indent=1, default=str))
print("\n저장 experiments/out/predict_0905.json")
