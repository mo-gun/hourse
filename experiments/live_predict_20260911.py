# -*- coding: utf-8 -*-
"""T-10분 실시간 예측 — 경주 하나씩, 발주 직전에 돌린다.

D-1 예측(predict_20260911.py)과 다른 점은 **그 시점에 새로 차 있는 값을 쓴다는 것**이다.

  tier A 71개   D-1 확정                          ← D-1 모델도 쓴다
  tier B  2개   마체중. 발주 **T-61~65분** (2026-09-10 제주 실측)
  tier C  2개   날씨·주로. 발주 T-10분 (팀 기록, 이번에 재확인한다)
  tier G  4개   배당. **못 쓴다** — 오픈 API 전수 0행이고 공개 웹에도 예상배당이 없다.
                m.kra.co.kr 경주성적표는 이미 뛴 경주의 확정배당만 보인다(실측).
                발매 시스템(로그인)과 경마장 전광판에만 있다.

→ **75피처** (73 + 마체중 2). 배당을 못 쓰니 시장(39.3%)과는 여전히 격차가 있다.

## 왜 미리 만들어 두는가 — 속도

as-of 프레임을 경주마다 다시 빌드하면 회당 180~230초다(원장 44.6만행 롤링).
그래서 **프레임은 하루 전에 만들어 두고**, T-10분에는 API 한 번 호출해
그 경주 행의 라이브 5컬럼만 갈아끼운 뒤 예측한다 — 초 단위로 끝난다.
모델도 미리 학습해 디스크에 저장한다(주 1회 재학습이 규약이다).

    # 하루 전 한 번 — 모델 학습·저장
    python experiments/live_predict_20260911.py --train

    # 발주 10분 전, 경주마다
    python experiments/live_predict_20260911.py --meet 3 --race 1

    # 상태만 보기 (어느 경주가 T-10분 창에 들어왔나)
    python experiments/live_predict_20260911.py --status
"""
import argparse
import io
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(r"c:/Users/SSAFY/Desktop/주제선정")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "experiments"))
import build_v2 as B                                    # noqa: E402
from kra_client import KRA                              # noqa: E402
from predict_20260911 import (TARGET, TS, SEEDS, PARAMS, N_ROUND, MEETNM,   # noqa: E402
                              build_frame, fix_sameday, join_extra, encode_pair)

OUT = HERE / "experiments" / "out"
MODELS = OUT / "live_models"
LOG = OUT / ("live_pred_%d.csv" % TARGET)
WG = ["X_wgHr", "X_wgHr_delta"]                 # tier B — 로컬 스키마는 tier=P 라 직접 넣는다
LIVE_COLS = ["X_wgHr", "X_wgHr_delta", "F4_weather", "F4_track_moist"]


def feature_sets(df):
    f73 = [c for c in TS.features(exclude_tier=("G",)) if c in df.columns]
    f75 = f73 + [c for c in WG if c in df.columns]
    f77 = [c for c in TS.features() if c in df.columns]          # 73 + F6 4개
    f79 = f77 + [c for c in WG if c in df.columns]
    assert len(f73) == 73 and len(f75) == 75, "%d/%d" % (len(f73), len(f75))
    assert len(f77) == 77 and len(f79) == 79, "%d/%d" % (len(f77), len(f79))
    return f73, f75, f77, f79


def prepared_frame():
    """D-1 에 만든 프레임 + 같은 날 누수 차단. 9/11 행만 돌려준다."""
    df = build_frame(False)
    rc = pd.to_numeric(df["rcDate"], errors="coerce")
    tr = df[df["split"] == "train"].copy()
    new = fix_sameday(df[rc == TARGET].copy())
    return tr, new


def do_train():
    tr, new = prepared_frame()
    _, f75, _, f79 = feature_sets(tr)
    MODELS.mkdir(parents=True, exist_ok=True)
    for tag, feats in (("75", f75), ("79", f79)):
        _train_one(tr, new, feats, tag)


def _train_one(tr, new, feats, tag):
    f75 = feats
    x, _ = encode_pair(tr, new, f75)
    groups = tr.groupby("race_id", sort=False).size().to_numpy()
    ds = lgb.Dataset(x[f75].to_numpy(float), label=tr["y_rel"].to_numpy(), group=groups)
    for s in SEEDS:
        p = dict(PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s)
        m = lgb.train(p, ds, num_boost_round=N_ROUND)
        # ★ LightGBM 의 save_model/Booster(model_file=) 은 **경로에 한글이 있으면 실패**한다
        #   (Windows C++ 계층 한계). 이 폴더가 "주제선정" 이라 그대로 쓰면
        #   "Model file ... is not available for writes" 가 난다.
        #   → 문자열로 뽑아 파이썬이 쓰고, 읽을 때도 model_str 로 넣는다.
        io.open(MODELS / ("t10_%s_%d.txt" % (tag, s)), "w",
                encoding="utf-8").write(m.model_to_string())
        print("  저장 t10_%s_%d.txt" % (tag, s))
    # 범주형 인코딩 사전을 고정해 둔다 — 추론 때 학습과 같은 코드를 써야 한다
    lv = {}
    for c in f75:
        if c in tr.columns and not pd.api.types.is_numeric_dtype(tr[c]):
            lv[c] = sorted(set(tr[c].dropna().astype(str)) | set(new[c].dropna().astype(str)))
    pd.to_pickle({"feats": f75, "levels": lv}, MODELS / ("t10_%s_meta.pkl" % tag))
    print("  학습 완료 — train %d행 %d경주 · %s피처 · 시드 %d개"
          % (len(tr), tr["race_id"].nunique(), tag, len(SEEDS)))


def load_models(tag="75"):
    meta = pd.read_pickle(MODELS / ("t10_%s_meta.pkl" % tag))
    ms = [lgb.Booster(model_str=io.open(MODELS / ("t10_%s_%d.txt" % (tag, s)),
                                        encoding="utf-8").read()) for s in SEEDS]
    return meta, ms


def fetch_live(kra, meet, rc_no):
    """그 경주의 라이브 값. 원장 API 는 당일 값(마체중·날씨·주로)을 채워 준다."""
    rows = kra.fetch("race_result", rc_date=str(TARGET), meet=meet, rc_no=rc_no)
    out = {}
    for r in rows:
        wg, wgd = B.parse_wg(r.get("wgHr"))
        ts, moist = B.parse_track(r.get("track"))
        wx = str(r.get("weather", "")).strip()
        out[int(r["chulNo"])] = dict(X_wgHr=wg, X_wgHr_delta=wgd,
                                     F4_weather=(np.nan if wx in ("", "-") else wx),
                                     F4_track_moist=moist)
    return out


def sched(kra, meet):
    plan = kra.fetch("race_plan", meet=meet, rc_date=str(TARGET))
    return {int(p["rcNo"]): str(p.get("schStTime") or "") for p in plan}


def tminus(hhmm, now=None):
    """발주까지 남은 분. **날짜를 포함해서** 센다.

    이걸 시:분만으로 계산하면 하루 전에 돌렸을 때 전 경주가 "지남" 으로 나온다
    (9/11 10:50 발주를 9/10 17:30 과 비교하게 된다). 실제로 한 번 그랬다.
    """
    if len(hhmm) != 4 or not hhmm.isdigit():
        return None
    now = now or datetime.now()
    post = datetime(TARGET // 10000, TARGET // 100 % 100, TARGET % 100,
                    int(hhmm[:2]), int(hhmm[2:]))
    return int(round((post - now).total_seconds() / 60))


def do_status():
    kra = KRA()
    now = datetime.now()
    print("=== %s 현재 %s ===" % (TARGET, now.strftime("%H:%M")))
    print("  %-6s %3s %7s %9s %8s" % ("장", "R", "발주", "T-minus", "상태"))
    for meet, nm in ((1, "서울"), (2, "제주"), (3, "부산경남")):
        sc = sched(kra, meet)
        for rn in sorted(sc):
            tm = tminus(sc[rn], now)
            st = "지남" if (tm is not None and tm <= 0) else \
                 ("★T-10 창" if tm is not None and tm <= 12 else "대기")
            print("  %-6s %3d   %s:%s %+8s분 %8s"
                  % (nm, rn, sc[rn][:2], sc[rn][2:], tm if tm is not None else "?", st))


SENTINEL_ORD = 91          # 원장 ord >= 91 = 미출주·취소·실격 (91·92·93·94·95·98·99, 전체 2.22%)


def refresh_entries(kra, meet, rc_no, g):
    """출전취소를 반영하고 **경주 내 값을 다시 계산**한다.

    원장은 취소마를 지우지 않고 ord 센티널로 표시한다(실측: 아신 94 · 아리온태양 95).
    D-1 프레임은 취소 전에 굳혀졌으므로 그대로 예측하면 두수가 틀린 전제로 돈다.

    취소마가 빠지면 다음이 전부 바뀐다 — 그래서 재계산한다:
      X_dusu            출주두수
      X_gate_rel        게이트 / 두수
      F5_race_n_front   이 경주 선행마 수
      _z / _rk  24개    경주 내 z-score · 순위 백분위 (NORMALIZE_WITHIN_RACE)

    ※ 기수변경은 F3_jk_* 를 정확히 다시 만들 수 없다(as-of 누적이 필요). 발생하면
      경고만 하고 값은 그대로 둔다 — 조용히 틀린 값을 쓰는 것보다 낫다.
    """
    from kra_client import KRAError
    rows = kra.fetch("race_result", rc_date=str(TARGET), meet=meet, rc_no=rc_no)
    out = set()
    for r in rows:
        o = str(r.get("ord", "")).strip()
        if o.isdigit() and int(o) >= SENTINEL_ORD:
            out.add(int(r["chulNo"]))
    try:                                   # cancel_info 로 교차확인
        for r in kra.fetch("cancel_info", rc_date=str(TARGET), meet=meet, rc_no=rc_no):
            out.add(int(r["chulNo"]))
    except KRAError:
        pass
    try:
        jc = kra.fetch("jockey_change", rc_date=str(TARGET), meet=meet, rc_no=rc_no)
    except KRAError:
        jc = []
    if jc:
        print("  ⚠ 기수변경 %d건 — F3_jk_* 를 정확히 재계산할 수 없다(as-of 누적 필요). "
              "값을 그대로 두고 표시만 한다:" % len(jc))
        for r in jc[:4]:
            print("      ", {k: r.get(k) for k in ("chulNo", "jkBef", "jkAft",
                                                   "befBudam", "aftBudam") if k in r})
    if not out:
        return g, []
    gates = g["X_chulNo"].astype(int)
    dropped = g[gates.isin(out)][["hrName", "X_chulNo"]].copy()
    g = g[~gates.isin(out)].copy()
    n = len(g)
    g["X_dusu"] = n
    g["X_gate_rel"] = g["X_chulNo"].astype(float) / n
    if "F5_early_pos" in g.columns and "F5_race_n_front" in g.columns:
        g["F5_race_n_front"] = float((pd.to_numeric(g["F5_early_pos"],
                                                    errors="coerce") < 0.25).sum())
    n_re = 0
    for c in TS.NORMALIZE_WITHIN_RACE:
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
    print("  ★ 출전취소 %d두 반영 — %s → 두수 %d · 경주내 재계산 %d컬럼"
          % (len(dropped), ", ".join("%s(%d번)" % (r.hrName, int(r.X_chulNo))
                                     for r in dropped.itertuples()), n, n_re))
    return g, dropped


def parse_odds(text, gates):
    """'1:3.2 2:5.8 ...' 또는 게이트 순서대로 '3.2 5.8 ...' 를 {게이트: 단승배당} 으로."""
    toks = [t for t in re.split(r"[,\s]+", text.strip()) if t]
    out = {}
    if all(":" in t for t in toks):
        for t in toks:
            g, v = t.split(":", 1)
            out[int(g)] = float(v)
    else:
        if len(toks) != len(gates):
            raise SystemExit("배당 %d개 vs 출전 %d두 — 게이트를 'N:배당' 으로 주거나 두수를 맞춰라"
                             % (len(toks), len(gates)))
        for g, t in zip(sorted(gates), toks):
            out[int(g)] = float(t)
    return out


def apply_market(g, odds):
    """단승배당 -> F6 4개. build_v2 와 **같은 식**이어야 한다.

      F6_mkt_prob      (1/배당) 을 경주 내 합=1 로 정규화 (오버라운드 제거)
      F6_mkt_rank      배당 오름차순 순위 (method='min')
      F6_mkt_prob_z    경주 내 z-score
      F6_field_entropy -(p·log p) 합 — 경주 내 상수
    배당 9999.9 계열(>= ODDS_NONE=900)은 '배당 없음' 특수값이라 제외한다.
    """
    import schema_v2 as S
    o = g["X_chulNo"].astype(int).map(odds).astype(float)
    o[o >= S.ODDS_NONE] = np.nan
    o[o <= 0] = np.nan
    inv = 1.0 / o
    p = inv / inv.sum(skipna=True)
    g["F6_mkt_prob"] = p.to_numpy()
    g["F6_mkt_rank"] = o.rank(method="min").to_numpy()
    sd = p.std()
    g["F6_mkt_prob_z"] = ((p - p.mean()) / (sd if sd and sd == sd else np.nan)).to_numpy()
    pc = p.clip(lower=1e-9)
    g["F6_field_entropy"] = float(-(pc * np.log(pc)).sum())
    n = int(o.notna().sum())
    print("  시장 피처 계산 — 배당 %d/%d두 · 1인기 %s · 엔트로피 %.3f"
          % (n, len(g),
             (g.loc[g["F6_mkt_rank"] == 1, "hrName"].iloc[0]
              if (g["F6_mkt_rank"] == 1).any() else "?"),
             g["F6_field_entropy"].iloc[0]))
    return g


def do_predict(meet, rc_no, odds_text=None):
    kra = KRA()
    now = datetime.now()
    sc = sched(kra, meet)
    tm = tminus(sc.get(rc_no, ""), now)
    print("=== %s %s %dR · 관측 %s · 발주 %s (T%+d분) ==="
          % (TARGET, MEETNM[meet], rc_no, now.strftime("%H:%M"),
             sc.get(rc_no, "?"), tm if tm is not None else 0))
    if tm is not None and tm <= 0:
        print("  ⚠ 이미 발주 시각이 지났다. 사전 예측이 아니다 — 기록은 남기지만 그렇게 표시한다.")

    tag = "79" if odds_text else "75"
    meta, models = load_models(tag)
    _, new = prepared_frame()
    g = new[(pd.to_numeric(new["meet"]) == meet) & (pd.to_numeric(new["rcNo"]) == rc_no)].copy()
    if g.empty:
        raise SystemExit("그 경주가 프레임에 없다. --meet/--race 확인.")

    g, dropped = refresh_entries(kra, meet, rc_no, g)
    if g.empty:
        raise SystemExit("전원 취소된 경주다.")

    live = fetch_live(kra, meet, rc_no)
    n_live = {c: 0 for c in LIVE_COLS}
    for i, row in g.iterrows():
        v = live.get(int(row["X_chulNo"]))
        if not v:
            continue
        for c in LIVE_COLS:
            if v[c] == v[c]:                       # NaN 아님
                g.at[i, c] = v[c]
                n_live[c] += 1
    print("  라이브 충전: " + " · ".join("%s %d/%d" % (c, n_live[c], len(g)) for c in LIVE_COLS))
    if n_live["X_wgHr"] == 0:
        print("  ⚠ 마체중이 아직 안 찼다 (발주 T-61~65분에 찬다). 지금 예측하면 tier B 가 결측이다.")
    if odds_text:
        g = apply_market(g, parse_odds(odds_text, g["X_chulNo"].astype(int).tolist()))
        print("  → 79피처 모델(73 + 마체중 2 + 시장 4) 사용")

    feats = meta["feats"]
    x = g.copy()
    for c, lv in meta["levels"].items():
        x[c] = pd.Categorical(x[c].astype(str), categories=lv).codes
    X = x[feats].to_numpy(float)
    score = np.mean([m.predict(X) for m in models], axis=0)
    g["score"] = score
    g = g.sort_values("score", ascending=False).reset_index(drop=True)
    g["rank"] = np.arange(1, len(g) + 1)

    print("")
    print("  %-3s %-10s %-7s %-7s %6s %8s" % ("순위", "마명", "기수", "조교사", "게이트", "마체중"))
    print("  " + "-" * 52)
    for r in g.itertuples():
        wgt = "-" if r.X_wgHr != r.X_wgHr else "%.0f" % r.X_wgHr
        print("  %-3d %-10s %-7s %-7s %6d %8s"
              % (r.rank, r.hrName, str(r.jkName)[:6], str(r.trName)[:6], int(r.X_chulNo), wgt))

    rec = g[["race_id", "row_id", "meet", "rcNo", "X_chulNo", "hrNo", "hrName", "jkName",
             "X_wgHr", "F4_weather", "F4_track_moist", "score", "rank"]].copy()
    rec["obs"] = now.strftime("%Y-%m-%d %H:%M:%S")
    rec["tminus"] = tm
    rec["n_live_wgHr"] = n_live["X_wgHr"]
    rec["n_live_weather"] = n_live["F4_weather"]
    rec["arm"] = tag
    rec["odds_src"] = "manual" if odds_text else ""
    rec["n_scratched"] = len(dropped)
    # ★ 단순 append 는 컬럼이 늘면 파일이 깨진다. 실제로 그랬다 — --odds 를 붙이며
    #   arm/odds_src 2개를 늘렸는데 헤더는 이미 17컬럼으로 쓰여 있어 뒤 행이 19컬럼이 되고
    #   다음 read_csv 가 "Expected 17 fields, saw 19" 로 죽었다(그래서 --auto 가 멈췄다).
    #   컬럼 합집합으로 다시 쓴다 — 스키마가 늘어도 자기 복구된다.
    if LOG.exists():
        old = pd.read_csv(LOG, encoding="utf-8-sig")
        cols = list(dict.fromkeys(list(old.columns) + list(rec.columns)))
        pd.concat([old.reindex(columns=cols), rec.reindex(columns=cols)],
                  ignore_index=True).to_csv(LOG, index=False, encoding="utf-8-sig")
    else:
        rec.to_csv(LOG, index=False, encoding="utf-8-sig")
    print("")
    print("  → %s 에 추가 (%d행)" % (LOG.name, len(rec)))
    print("  ★ 16경주 top-1 표준오차 ±11.8%p. 이건 파이프라인 점검이고 우열 판정이 아니다.")


def do_auto(window=12, floor=6, poll=60):
    """대기하다가 각 경주 T-window~T-floor 분 창에서 한 번씩 예측한다.

    경주가 두 경마장에 교차 편성돼 10:40~17:45 사이에 16번을 돌려야 한다.
    수동으로 지키는 건 비현실적이라 여기서 대기·발사한다. 이미 예측한 경주는
    로그를 보고 건너뛰므로 중간에 끊고 다시 켜도 중복되지 않는다.
    """
    import time
    kra = KRA()
    plan = {}
    for meet in (1, 2, 3):
        for rn, hhmm in sched(kra, meet).items():
            if hhmm:
                plan[(meet, rn)] = hhmm
    print("편성 %d경주 · T-%d~%d분 창에서 발사 · %d초마다 확인" % (len(plan), window, floor, poll))
    while True:
        done = set()
        if LOG.exists():
            try:
                L = pd.read_csv(LOG, encoding="utf-8-sig")
                done = {(int(a), int(b)) for a, b in zip(L["meet"], L["rcNo"])}
            except Exception as e:
                # 로그가 깨져도 루프를 죽이지 않는다 — 최악이라도 중복 예측이지
                # 발사를 놓치는 것보다 낫다
                print("  ⚠ 로그 읽기 실패(%s) — 중복 위험을 안고 계속한다" % type(e).__name__)
        todo = {k: v for k, v in plan.items() if k not in done}
        if not todo:
            print("전 경주 예측 완료.")
            return
        now = datetime.now()
        fired = False
        for (meet, rn), hhmm in sorted(todo.items(), key=lambda kv: tminus(kv[1]) or 0):
            tm = tminus(hhmm, now)
            if tm is None:
                continue
            if floor <= tm <= window:
                print("")
                do_predict(meet, rn)
                fired = True
            elif tm < floor:
                # 창을 놓쳤다 — 그냥 지나가면 기록이 비니 늦게라도 남기고 표시한다
                print("")
                print("  ⚠ %s %dR 은 T-%d분 창을 놓쳤다 (지금 T%+d분). 늦게 기록한다."
                      % (MEETNM[meet], rn, window, tm))
                do_predict(meet, rn)
                fired = True
        if not fired:
            nxt = min((tminus(v, now) or 9999) for v in todo.values())
            print("  %s · 남은 %d경주 · 다음 발사까지 %d분" % (now.strftime("%H:%M"), len(todo), max(nxt - window, 0)),
                  flush=True)
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="모델 학습·저장 (하루 전 한 번)")
    ap.add_argument("--status", action="store_true", help="어느 경주가 T-10 창인가")
    ap.add_argument("--auto", action="store_true",
                    help="대기하다가 경주마다 T-10분에 자동으로 한 번씩 예측")
    ap.add_argument("--meet", type=int, choices=(1, 2, 3))
    ap.add_argument("--race", type=int)
    ap.add_argument("--odds", default=None,
                    help="단승배당 직접 입력. \"1:3.2 2:5.8 ...\" 또는 게이트 순서 \"3.2 5.8 ...\". "
                         "주면 79피처(시장 포함) 모델을 쓴다")
    a = ap.parse_args()
    if a.train:
        do_train()
    elif a.status:
        do_status()
    elif a.auto:
        do_auto()
    elif a.meet and a.race:
        do_predict(a.meet, a.race, a.odds)
    else:
        ap.error("--train / --status / --auto / (--meet 과 --race) 중 하나가 필요하다")


if __name__ == "__main__":
    main()
