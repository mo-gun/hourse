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
    assert len(f73) == 73 and len(f75) == 75, "%d/%d" % (len(f73), len(f75))
    return f73, f75


def prepared_frame():
    """D-1 에 만든 프레임 + 같은 날 누수 차단. 9/11 행만 돌려준다."""
    df = build_frame(False)
    rc = pd.to_numeric(df["rcDate"], errors="coerce")
    tr = df[df["split"] == "train"].copy()
    new = fix_sameday(df[rc == TARGET].copy())
    return tr, new


def do_train():
    tr, new = prepared_frame()
    _, f75 = feature_sets(tr)
    MODELS.mkdir(parents=True, exist_ok=True)
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
        io.open(MODELS / ("t10_75_%d.txt" % s), "w", encoding="utf-8").write(m.model_to_string())
        print("  저장 t10_75_%d.txt" % s)
    # 범주형 인코딩 사전을 고정해 둔다 — 추론 때 학습과 같은 코드를 써야 한다
    lv = {}
    for c in f75:
        if c in tr.columns and not pd.api.types.is_numeric_dtype(tr[c]):
            lv[c] = sorted(set(tr[c].dropna().astype(str)) | set(new[c].dropna().astype(str)))
    pd.to_pickle({"feats": f75, "levels": lv}, MODELS / "t10_75_meta.pkl")
    print("  학습 완료 — train %d행 %d경주 · 75피처 · 시드 %d개"
          % (len(tr), tr["race_id"].nunique(), len(SEEDS)))


def load_models():
    meta = pd.read_pickle(MODELS / "t10_75_meta.pkl")
    ms = [lgb.Booster(model_str=io.open(MODELS / ("t10_75_%d.txt" % s),
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


def do_predict(meet, rc_no):
    kra = KRA()
    now = datetime.now()
    sc = sched(kra, meet)
    tm = tminus(sc.get(rc_no, ""), now)
    print("=== %s %s %dR · 관측 %s · 발주 %s (T%+d분) ==="
          % (TARGET, MEETNM[meet], rc_no, now.strftime("%H:%M"),
             sc.get(rc_no, "?"), tm if tm is not None else 0))
    if tm is not None and tm <= 0:
        print("  ⚠ 이미 발주 시각이 지났다. 사전 예측이 아니다 — 기록은 남기지만 그렇게 표시한다.")

    meta, models = load_models()
    _, new = prepared_frame()
    g = new[(pd.to_numeric(new["meet"]) == meet) & (pd.to_numeric(new["rcNo"]) == rc_no)].copy()
    if g.empty:
        raise SystemExit("그 경주가 프레임에 없다. --meet/--race 확인.")

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
    hdr = not LOG.exists()
    rec.to_csv(LOG, mode="a", header=hdr, index=False, encoding="utf-8-sig")
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
            L = pd.read_csv(LOG, encoding="utf-8-sig")
            done = {(int(a), int(b)) for a, b in zip(L["meet"], L["rcNo"])}
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
    a = ap.parse_args()
    if a.train:
        do_train()
    elif a.status:
        do_status()
    elif a.auto:
        do_auto()
    elif a.meet and a.race:
        do_predict(a.meet, a.race)
    else:
        ap.error("--train / --status / --auto / (--meet 과 --race) 중 하나가 필요하다")


if __name__ == "__main__":
    main()
