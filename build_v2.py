# -*- coding: utf-8 -*-
"""
v2 빌더 — model / game / sim 세 갈래.

    python build_v2.py model     학습셋
    python build_v2.py game      게임 구동 데이터 (배당 API 호출 있음)
    python build_v2.py sim       경기 흐름 재생 데이터
    python build_v2.py all       전부

누수 방지 3원칙 (v1 에서 확립, v2 도 동일)
  1. as-of 집계는 시간순 단일 패스. 행마다 emit(현재까지 누적) → update(이 경주 반영).
  2. 기준기록(par)은 train 구간에서만 산출.
  3. 육종가는 경주일 **직전** 스냅샷만 (ebv_join 이 강제).

게임풀
  dataset/v2/game_holdout.json 의 개최일은 학습·검증·평가에서 완전히 뺀다.
  이 파일을 읽지 않고 빌드하면 안 된다.
"""
import os, sys, json, io, time, argparse
from collections import defaultdict, deque
from datetime import date

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd

import schema_v2 as S
import aux_join as AX
import ebv_join as EJ

LEDGER = "data/raw/ledger_2010_2026.csv"
MEET_CODE = {"서울": 1, "제주": 2, "부산경남": 3, "1": 1, "2": 2, "3": 3}
MEET_NAME = {1: "서울", 2: "제주", 3: "부산경남"}
ORD_MAX = 16
W365, W28 = 365, 28

SECT_ORD = {   # 경마장별 구간 순위 컬럼 (초·중·종반)
    1: {"early": ["sjS1fOrd"], "mid": ["sj_3cOrd", "sj_2cOrd"], "late": ["sjG3fOrd", "sj_4cOrd"]},
    3: {"early": ["buS1fOrd"], "mid": ["buG8fOrd", "buG6fOrd"], "late": ["buG3fOrd", "buG2fOrd"]},
    2: {"early": ["jeS1fTime"], "mid": ["je_2cTime", "je_1cTime"], "late": ["jeG3fTime"], "_t": True},
}
SECT_TIME = {1: {"s1f": "seS1fAccTime", "g3f": "seG3fAccTime"},
             2: {"s1f": "jeS1fTime", "g3f": "jeG3fTime"},
             3: {"s1f": "buS1fAccTime", "g3f": "buG3fAccTime"}}


# ══════════════════════════════════════════════════════════════════════
# 공통 — 로드 · 파싱 · 분할
# ══════════════════════════════════════════════════════════════════════
def _f(x):
    try:
        return float(str(x).strip())
    except Exception:
        return np.nan


def parse_wg(s):
    s = str(s).strip()
    if not s or s in ("-", "0()"):
        return np.nan, np.nan
    w, d = np.nan, np.nan
    if "(" in s:
        h, _, t = s.partition("(")
        w = _f(h)
        t = t.rstrip(")").strip()
        if t:
            d = _f(t.replace("+", ""))
    else:
        w = _f(s)
    return (np.nan if (w != w or w <= 0) else w), d


def parse_track(s):
    s = str(s).strip()
    if not s or s.startswith("(0%)") or s == "-":
        return np.nan, np.nan
    st, _, rest = s.partition("(")
    st = st.strip()
    return (st or np.nan), _f(rest.replace("%", "").replace(")", "").strip())


def dist_band(d):
    if d != d:
        return "unk"
    d = int(d)
    return "S" if d <= 1200 else "SM" if d <= 1400 else "M" if d <= 1700 else "ML" if d <= 1900 else "L"


def load_ledger(limit=None):
    print("원장 로드...", flush=True)
    df = pd.read_csv(LEDGER, dtype=str, encoding="utf-8-sig", nrows=limit, low_memory=False)
    df["rcDate"] = pd.to_numeric(df["rcDate"], errors="coerce")
    df["meet"] = df["meet"].astype(str).str.strip().map(MEET_CODE)
    for c in ("rcNo", "chulNo", "age", "rcDist"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("ord", "rcTime", "wgBudam", "chaksun1"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").replace(0, np.nan)
    df["winOdds"] = pd.to_numeric(df["winOdds"], errors="coerce").replace(0, np.nan)
    df["plcOdds"] = pd.to_numeric(df["plcOdds"], errors="coerce").replace(0, np.nan)
    # 9999.9 는 "배당 없음"을 뜻하는 특수값이다. 진짜 배당으로 계산하면 숫자가 터진다 —
    # 안 거르고 수익률을 재면 +250% 가 나온다(실제 -21.4%). 여기서 한 번 막으면
    # F6_mkt_prob · F6_mkt_rank · game/entries 까지 전부 정리된다.  [EDA 2026-09 ②]
    for _oc in ("winOdds", "plcOdds"):
        df.loc[df[_oc] >= S.ODDS_NONE, _oc] = np.nan
    # 등급 표기 통합 (`국5등급`/`국5` → `국5`). 안 하면 42개 표기가 서로 다른
    # 등급으로 학습된다.  [EDA 2026-09 ⑤]
    df["rank"] = df["rank"].map(S.normalize_grade)

    wg = df["wgHr"].map(parse_wg)
    df["X_wgHr"] = [a for a, _ in wg]
    df["X_wgHr_delta"] = [b for _, b in wg]
    tk = df["track"].map(parse_track)
    df["track_state"] = [a for a, _ in tk]
    df["F4_track_moist"] = [b for _, b in tk]
    df["F4_weather"] = df["weather"].replace({"-": np.nan, "": np.nan})

    n0 = len(df)
    df = df[df["ord"].between(1, ORD_MAX) & df["rcDate"].notna() & df["hrNo"].notna()].copy()
    print(f"  유효 완주 {len(df):,}행 (제외 {n0-len(df):,} — 미출주·취소·실격 센티널)")

    df["day"] = df["rcDate"].astype(int).astype(str) + "_" + df["meet"].map(MEET_NAME)
    df["race_id"] = df["rcDate"].astype(int).astype(str) + "_" + \
        df["meet"].astype("Int64").astype(str) + "_" + df["rcNo"].astype("Int64").astype(str)
    df["row_id"] = df["race_id"] + "_" + df["chulNo"].astype("Int64").astype(str)
    df["dusu"] = df.groupby("race_id")["hrNo"].transform("size")
    df["_ordinal"] = [date(int(d) // 10000, int(d) // 100 % 100, int(d) % 100).toordinal()
                      for d in df["rcDate"]]
    df["_band"] = df["rcDist"].map(dist_band)
    df["_wet"] = (df["F4_track_moist"] >= 10).astype(float)
    df["_birth_yr"] = EJ.recover_birth_year(df)
    return df.sort_values(["_ordinal", "meet", "rcNo", "chulNo"]).reset_index(drop=True)


def assign_split(df):
    """game 예약을 먼저 빼고, 나머지를 최신 기준 상대 구간으로 나눈다."""
    ho = json.load(io.open(S.HOLDOUT_FILE, encoding="utf-8"))
    days = set(ho["days"])
    df["split"] = "train"
    is_game = df["day"].isin(days)
    df.loc[is_game, "split"] = "game"

    rest = df[~is_game]
    last = int(rest["_ordinal"].max())
    t0 = last - S.TEST_WEEKS * 7
    v0 = t0 - S.VALID_WEEKS * 7
    df.loc[~is_game & (df["_ordinal"] > t0), "split"] = "test"
    df.loc[~is_game & (df["_ordinal"] > v0) & (df["_ordinal"] <= t0), "split"] = "valid"
    print(f"  게임 예약 {ho['n_days']}일 / {int(is_game.sum()):,}행 (seed={ho['seed']})")
    return df


def add_targets(df):
    df["y_ord"] = df["ord"].astype(int)
    df["y_win"] = (df["y_ord"] == 1).astype(int)
    df["y_plc"] = (df["y_ord"] <= np.where(df["dusu"] >= 7, 3, 2)).astype(int)
    df["y_rel"] = np.maximum(0, df["dusu"] - df["y_ord"]).astype(int)
    inv = 1.0 / df["winOdds"]
    tot = inv.groupby(df["race_id"]).transform("sum")
    df["F6_mkt_prob"] = inv / tot
    df["F6_mkt_rank"] = df.groupby("race_id")["winOdds"].rank(method="min")
    df["_overround"] = tot - 1.0
    return df


def add_speed_fig(df):
    tr = df["split"] == "train"
    key = ["meet", "rcDist", "track_state"]
    par = df[tr].groupby(key, dropna=False)["rcTime"].agg(["median", "std", "size"])
    par = par[par["size"] >= 20]
    par2 = df[tr].groupby(["meet", "rcDist"], dropna=False)["rcTime"].agg(["median", "std", "size"])
    par2 = par2[par2["size"] >= 20]
    k1 = pd.MultiIndex.from_arrays([df[c] for c in key])
    k2 = pd.MultiIndex.from_arrays([df["meet"], df["rcDist"]])
    med = pd.Series(par["median"].reindex(k1).to_numpy(), index=df.index)
    sd = pd.Series(par["std"].reindex(k1).to_numpy(), index=df.index)
    med = med.fillna(pd.Series(par2["median"].reindex(k2).to_numpy(), index=df.index))
    sd = sd.fillna(pd.Series(par2["std"].reindex(k2).to_numpy(), index=df.index)).replace(0, np.nan)
    df["y_speed_fig"] = (100 + 10 * (med - df["rcTime"]) / sd).clip(40, 160)
    print(f"  par {len(par):,}조합(+폴백 {len(par2):,}) · speed_fig {df['y_speed_fig'].notna().mean()*100:.0f}%")
    return df


def add_sections(df):
    for slot in ("early", "mid", "late"):
        df[f"_pos_{slot}"] = np.nan
    for meet, cfg in SECT_ORD.items():
        m = df["meet"] == meet
        if not m.any():
            continue
        for slot in ("early", "mid", "late"):
            val = pd.Series(np.nan, index=df.index[m])
            for c in cfg.get(slot, []):
                if c in df.columns:
                    v = pd.to_numeric(df.loc[m, c], errors="coerce")
                    val = val.fillna(v.where(v > 0))
            if cfg.get("_t"):
                val = pd.DataFrame({"r": df.loc[m, "race_id"], "v": val}) \
                    .groupby("r")["v"].rank(method="min")
            df.loc[m, f"_pos_{slot}"] = ((val - 1) / (df.loc[m, "dusu"] - 1)).to_numpy()
    for slot in ("s1f", "g3f"):
        raw = pd.Series(np.nan, index=df.index)
        for meet, cfg in SECT_TIME.items():
            c = cfg[slot]
            if c in df.columns:
                m = df["meet"] == meet
                v = pd.to_numeric(df.loc[m, c], errors="coerce")
                raw.loc[m] = v.where(v > 0).to_numpy()
        tr = df["split"] == "train"
        par = raw[tr].groupby([df.loc[tr, "meet"], df.loc[tr, "rcDist"]]).median()
        k = pd.MultiIndex.from_arrays([df["meet"], df["rcDist"]])
        df[f"_t_{slot}"] = par.reindex(k).to_numpy() - raw.to_numpy()
    return df


# ══════════════════════════════════════════════════════════════════════
# model — as-of 롤링
# ══════════════════════════════════════════════════════════════════════
class Win365:
    __slots__ = ("q", "n", "w")

    def __init__(self):
        self.q, self.n, self.w = deque(), 0, 0

    def tick(self, day):
        while self.q and day - self.q[0][0] > W365:
            _, w = self.q.popleft()
            self.n -= 1; self.w -= w

    def add(self, day, w):
        self.q.append((day, w)); self.n += 1; self.w += w


def _r(w, n):
    return w / n if n else np.nan


ROLL_NUM = ["F1_starts_life", "F1_win_rate_life", "F1_ord_avg3", "F1_ordpct_avg5",
            "F1_speed_avg3", "F1_prize_life", "F1_prize_365", "F1_grade_move",
            "F1_layoff_days", "F1_first_start",
            "F1_tr_sessions_28d", "F1_tr_jk_ridden_28d",
            "F2_sire_win_rate", "F2_sire_prog_n",
            "F3_jk_win_rate_365", "F3_jk_win_rate_life", "F3_jk_win_rate_dist",
            "F3_tr_win_rate_365", "F3_jkhr_starts", "F3_jkhr_win_rate",
            "F4_hr_dist_win_rate", "F4_dist_gap", "F4_hr_meet_win_rate",
            "F4_hr_wet_win_rate",
            "F5_early_pos", "F5_pos_gain", "F5_g3f_time", "F5_sect_n"]
ROLL_STR = ["F5_style", "F2_sire_id"]


def _grade_num(g):
    s = str(g)
    for ch in s:
        if ch.isdigit():
            return int(ch)
    return None


def rolling(df):
    n = len(df)
    print("보조 인덱스 로드...", flush=True)
    tr_idx = AX.load_train_index()
    ped = AX.load_pedigree()

    hist = defaultdict(list)
    hr_life = defaultdict(lambda: [0, 0])
    hr_prize = defaultdict(float)
    hr_p365 = defaultdict(lambda: deque())
    hr_dist = defaultdict(lambda: [0, 0])
    hr_meet = defaultdict(lambda: [0, 0])
    hr_wet = defaultdict(lambda: [0, 0])
    jk_life = defaultdict(lambda: [0, 0])
    jk_365 = defaultdict(Win365)
    jk_dist = defaultdict(lambda: [0, 0])
    tr_365 = defaultdict(Win365)
    jkhr = defaultdict(lambda: [0, 0])
    sire = defaultdict(lambda: [0, 0, set()])

    out = {c: np.full(n, np.nan) for c in ROLL_NUM}
    outs = {c: np.array([None] * n, dtype=object) for c in ROLL_STR}

    A = {c: df[c].to_numpy() for c in
         ("_ordinal", "y_ord", "dusu", "y_win", "rcDist", "_band", "meet", "_wet",
          "y_speed_fig", "rank", "chaksun1", "_pos_early", "_pos_late", "_t_g3f",
          "hrNo", "jkNo", "trNo")}
    t0 = time.time()
    for i in range(n):
        day, hr, jk, tn = A["_ordinal"][i], A["hrNo"][i], A["jkNo"][i], A["trNo"][i]
        band, mt = A["_band"][i], A["meet"][i]

        # ── emit ──
        h = hr_life[hr]
        out["F1_starts_life"][i] = h[0]
        out["F1_win_rate_life"][i] = _r(h[1], h[0])
        out["F1_first_start"][i] = 1.0 if h[0] == 0 else 0.0
        out["F1_prize_life"][i] = hr_prize[hr]
        q = hr_p365[hr]
        while q and day - q[0][0] > W365:
            q.popleft()
        out["F1_prize_365"][i] = sum(x[1] for x in q)

        H = hist[hr]
        if H:
            last = H[-1]
            out["F1_layoff_days"][i] = day - last[0]
            gp, gn = _grade_num(last[5]), _grade_num(A["rank"][i])
            if gp is not None and gn is not None:
                out["F1_grade_move"][i] = gp - gn
            L3 = H[-3:]
            out["F1_ord_avg3"][i] = np.mean([x[1] for x in L3])
            s3 = [x[3] for x in L3 if x[3] == x[3]]
            if s3:
                out["F1_speed_avg3"][i] = np.mean(s3)
            L5 = H[-5:]
            out["F1_ordpct_avg5"][i] = np.mean(
                [(x[1] - 1) / (x[2] - 1) if x[2] > 1 else np.nan for x in L5])
            bd = [x for x in H if x[4] == x[4] and dist_band(x[4]) == band]
            best = min(H, key=lambda x: x[1])
            if best[4] == best[4] and A["rcDist"][i] == A["rcDist"][i]:
                out["F4_dist_gap"][i] = abs(A["rcDist"][i] - best[4])
            pe = [x[6] for x in H if x[6] == x[6]]
            if pe:
                mE = float(np.mean(pe))
                out["F5_early_pos"][i] = mE
                outs["F5_style"][i] = ("선행" if mE < .20 else "선입" if mE < .45
                                       else "중단" if mE < .70 else "추입")
                fin = [(x[1] - 1) / (x[2] - 1) if x[2] > 1 else np.nan
                       for x in H if x[6] == x[6]]
                g = [a - b for a, b in zip(pe, fin) if b == b]
                if g:
                    out["F5_pos_gain"][i] = float(np.mean(g))
            out["F5_sect_n"][i] = len(pe)
            tg = [x[7] for x in H if x[7] == x[7]]
            if tg:
                out["F5_g3f_time"][i] = float(np.mean(tg))

        d_ = hr_dist[(hr, band)]; out["F4_hr_dist_win_rate"][i] = _r(d_[1], d_[0])
        m_ = hr_meet[(hr, mt)]; out["F4_hr_meet_win_rate"][i] = _r(m_[1], m_[0])
        w_ = hr_wet[hr]; out["F4_hr_wet_win_rate"][i] = _r(w_[1], w_[0])
        j = jk_life[jk]; out["F3_jk_win_rate_life"][i] = _r(j[1], j[0])
        jw = jk_365[jk]; jw.tick(day); out["F3_jk_win_rate_365"][i] = _r(jw.w, jw.n)
        jd = jk_dist[(jk, band)]; out["F3_jk_win_rate_dist"][i] = _r(jd[1], jd[0])
        tw = tr_365[tn]; tw.tick(day); out["F3_tr_win_rate_365"][i] = _r(tw.w, tw.n)
        p = jkhr[(jk, hr)]
        out["F3_jkhr_starts"][i] = p[0]; out["F3_jkhr_win_rate"][i] = _r(p[1], p[0])
        t5 = AX.train_features(tr_idx, hr, day)
        out["F1_tr_sessions_28d"][i] = t5[0]
        out["F1_tr_jk_ridden_28d"][i] = t5[3]
        fa = (ped.get(hr) or (None, None, None))[0]
        outs["F2_sire_id"][i] = fa
        if fa:
            s = sire[fa]
            out["F2_sire_win_rate"][i] = _r(s[1], s[0]) if s[0] else np.nan
            out["F2_sire_prog_n"][i] = len(s[2])

        # ── update ──
        wv = int(A["y_win"][i]); pz = A["chaksun1"][i] if wv else 0.0
        h[0] += 1; h[1] += wv
        if pz == pz:
            hr_prize[hr] += pz; q.append((day, pz))
        d_[0] += 1; d_[1] += wv
        m_[0] += 1; m_[1] += wv
        if A["_wet"][i] == 1:
            w_[0] += 1; w_[1] += wv
        j[0] += 1; j[1] += wv; jw.add(day, wv)
        jd[0] += 1; jd[1] += wv; tw.add(day, wv)
        p[0] += 1; p[1] += wv
        if fa:
            s = sire[fa]; s[0] += 1; s[1] += wv; s[2].add(hr)
        H.append((day, A["y_ord"][i], A["dusu"][i], A["y_speed_fig"][i], A["rcDist"][i],
                  A["rank"][i], A["_pos_early"][i], A["_t_g3f"][i]))
        if len(H) > 40:
            del H[:-40]
        if i and i % 100000 == 0:
            print(f"    {i:,}/{n:,}  {time.time()-t0:.0f}s", flush=True)

    for c, v in out.items():
        df[c] = v
    for c, v in outs.items():
        df[c] = v
    print(f"  as-of 롤링 완료 {time.time()-t0:.0f}s")
    return df


def add_base_and_norm(df):
    base = {"X_age": df["age"], "X_sex": df["sex"], "X_prd_cty": df["name"],
            "X_rcDist": df["rcDist"], "X_grade": df["rank"], "X_dusu": df["dusu"],
            "X_chulNo": df["chulNo"], "X_gate_rel": df["chulNo"] / df["dusu"],
            "X_wgBudam": df["wgBudam"], "X_rating": df["rating"],
            "F4_month": (df["rcDate"].astype(int) // 100) % 100,
            "F5_race_n_front": np.nan}
    df = pd.concat([df, pd.DataFrame(base, index=df.index)], axis=1)
    df["F5_race_n_front"] = (df["F5_early_pos"] < 0.25).astype(float) \
        .groupby(df["race_id"]).transform("sum")
    df["F6_field_entropy"] = df.groupby("race_id")["F6_mkt_prob"].transform(
        lambda p: -(p * np.log(p.clip(1e-9))).sum())
    # 인기도는 경주 내 상대값이 본질이라 z 를 따로 만든다 (rank 는 F6_mkt_rank 가 이미 있음)
    _mp = df["F6_mkt_prob"]
    _mu = _mp.groupby(df["race_id"]).transform("mean")
    _sd = _mp.groupby(df["race_id"]).transform("std").replace(0, np.nan)
    df["F6_mkt_prob_z"] = (_mp - _mu) / _sd
    # 이번 거리에 맞춘 육종가
    if "F2_ebv_sprint" in df.columns:
        short = df["rcDist"] <= 1400
        df["F2_ebv_dist_fit"] = np.where(short, df["F2_ebv_sprint"], df["F2_ebv_route"])

    new = {}
    for c in S.NORMALIZE_WITHIN_RACE:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        mu = v.groupby(df["race_id"]).transform("mean")
        sd = v.groupby(df["race_id"]).transform("std").replace(0, np.nan)
        new[c + "_z"] = (v - mu) / sd          # ★ v1 과 달리 fillna(0) 안 한다
        new[c + "_rk"] = v.groupby(df["race_id"]).rank(pct=True)
    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    print(f"  경주내 정규화 {len(new)}컬럼 (결측은 NaN 유지)")
    return df


def build_model(limit=None):
    os.makedirs(S.MODEL_DIR, exist_ok=True)
    df = load_ledger(limit)
    df = assign_split(df)
    df = add_targets(df)
    df = add_speed_fig(df)
    df = add_sections(df)
    df = rolling(df)
    df, added = EJ.attach(df)
    EJ.report(df, added)
    df = add_base_and_norm(df)

    cols = [c["name"] for c in S.INDEX] + [c["name"] for c in S.TARGETS]
    feats = [c for c in S.features() if c in df.columns]
    keep = [c for c in cols + feats if c in df.columns]
    out = df[keep].copy()
    for c in S.CATEGORICAL + ["split"]:
        if c in out.columns:
            out[c] = out[c].astype("category")
    for c in out.columns:
        if str(out[c].dtype) == "float64":
            out[c] = out[c].astype("float32")

    bad = [c for c in out.columns if c in S.FORBIDDEN]
    assert not bad, f"누수 컬럼: {bad}"

    for sp in ("train", "valid", "test", "game"):
        sub = out[out["split"] == sp]
        if not len(sub):
            continue
        p = f"{S.MODEL_DIR}/{sp}.parquet"
        sub.to_parquet(p, index=False)
        print(f"  {sp:6s} {len(sub):>7,}행 {sub['race_id'].nunique():>6,}경주 "
              f"{int(sub['rcDate'].min())}~{int(sub['rcDate'].max())}  "
              f"{os.path.getsize(p)/1e6:.0f}MB")
    miss = [c for c in S.features() if c not in out.columns]
    if miss:
        print(f"  미구현 {len(miss)}: {', '.join(miss[:6])}")
    return out


# ══════════════════════════════════════════════════════════════════════
# game / sim
# ══════════════════════════════════════════════════════════════════════
def build_sim(limit=None):
    """게임풀 경주의 구간별 통과 기록. ★ 학습에 안 쓴다."""
    os.makedirs(S.SIM_DIR, exist_ok=True)
    df = load_ledger(limit)
    df = assign_split(df)
    g = df[df["split"] == "game"].copy()
    print(f"게임풀 {g['race_id'].nunique():,}경주 / {len(g):,}두")
    rows = []
    for meet, cfg in SECT_ORD.items():
        m = g["meet"] == meet
        if not m.any():
            continue
        sub = g[m]
        for slot, pt in (("early", "S1F"), ("mid", "C2"), ("late", "G3F")):
            val = pd.Series(np.nan, index=sub.index)
            for c in cfg.get(slot, []):
                if c in sub.columns:
                    v = pd.to_numeric(sub[c], errors="coerce")
                    val = val.fillna(v.where(v > 0))
            if cfg.get("_t"):
                val = pd.DataFrame({"r": sub["race_id"], "v": val}) \
                    .groupby("r")["v"].rank(method="min")
            rows.append(pd.DataFrame({
                "race_id": sub["race_id"], "chulNo": sub["chulNo"], "hrNo": sub["hrNo"],
                "point": pt, "pass_ord": val.to_numpy(),
                "rel_pos": ((val - 1) / (sub["dusu"] - 1)).to_numpy()}))
        rows.append(pd.DataFrame({
            "race_id": sub["race_id"], "chulNo": sub["chulNo"], "hrNo": sub["hrNo"],
            "point": "FIN", "pass_ord": sub["ord"].to_numpy(),
            "rel_pos": ((sub["ord"] - 1) / (sub["dusu"] - 1)).to_numpy()}))
    sim = pd.concat(rows, ignore_index=True).dropna(subset=["pass_ord"])
    sim["point"] = pd.Categorical(sim["point"], S.SIM_POINTS, ordered=True)
    sim = sim.sort_values(["race_id", "point", "pass_ord"])
    p = f"{S.SIM_DIR}/passing.parquet"
    sim.to_parquet(p, index=False)
    print(f"  저장 {p}  {len(sim):,}행  {os.path.getsize(p)/1e6:.0f}MB")
    print(f"  지점별 충전: {sim.groupby('point', observed=True).size().to_dict()}")
    return sim


def build_game(limit=None, fetch_odds=True):
    """경주 카드 + 출주마 + 7승식 적중배당."""
    os.makedirs(S.GAME_DIR, exist_ok=True)
    df = load_ledger(limit)
    df = assign_split(df)
    g = df[df["split"] == "game"].copy()
    print(f"게임풀 {g['race_id'].nunique():,}경주 / {len(g):,}두")

    card = g.groupby("race_id").agg(
        rcDate=("rcDate", "first"), meet=("meet", "first"), rcNo=("rcNo", "first"),
        rcDist=("rcDist", "first"), grade=("rank", "first"),
        weather=("F4_weather", "first"), track_state=("track_state", "first"),
        track_moist=("F4_track_moist", "first"), dusu=("dusu", "first"),
        prize1=("chaksun1", "first")).reset_index()
    card["meet_nm"] = card["meet"].map(MEET_NAME)
    card.to_parquet(f"{S.GAME_DIR}/race_card.parquet", index=False)
    print(f"  race_card {len(card):,}경주")

    ent = g[["race_id", "chulNo", "hrNo", "hrName", "age", "sex", "name",
             "wgBudam", "X_wgHr", "X_wgHr_delta", "jkNo", "jkName", "trNo", "trName",
             "rating", "winOdds", "plcOdds", "ord", "rcTime"]].copy()
    ent = ent.rename(columns={"name": "prd_cty", "X_wgHr": "wgHr",
                              "X_wgHr_delta": "wgHr_delta", "ord": "y_ord"})
    inv = 1.0 / ent["winOdds"]
    ent["mkt_prob"] = inv / inv.groupby(ent["race_id"]).transform("sum")
    ent["mkt_rank"] = ent.groupby("race_id")["winOdds"].rank(method="min")
    ent.to_parquet(f"{S.GAME_DIR}/entries.parquet", index=False)
    print(f"  entries {len(ent):,}두")

    if fetch_odds:
        fetch_payouts(g)
    return card, ent


def fetch_payouts(g):
    """7승식 적중 조합의 배당만 저장. 경주당 전 조합은 1,040행이라 과하다."""
    import urllib.request, urllib.parse, ssl
    CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE
    KEY = [l.split("=", 1)[1].strip() for l in open(".env", encoding="utf-8")
           if l.startswith("KRA_API_KEY_ENCODED")][0]
    out_p = f"{S.GAME_DIR}/payouts.csv"
    done = set()
    if os.path.exists(out_p):
        done = set(pd.read_csv(out_p, usecols=["race_id"])["race_id"])
    # 적중 조합 = 착순 1,2,3위의 chulNo
    fin = g[g["ord"] <= 3].sort_values(["race_id", "ord"])
    top = fin.groupby("race_id")["chulNo"].apply(list).to_dict()

    keys = g[["rcDate", "meet", "race_id"]].drop_duplicates()
    daymeet = keys.groupby([keys["rcDate"].astype(int), keys["meet"].astype(int)]).size()
    todo = [(d, m) for (d, m) in daymeet.index
            if not set(keys[(keys.rcDate == d) & (keys.meet == m)]["race_id"]) <= done]
    print(f"  배당 수집 대상 {len(todo)}개 (개최일×경마장). 완료분 {len(done):,}경주 스킵")

    f = io.open(out_p, "a" if done else "w", encoding="utf-8-sig", newline="")
    import csv as _csv
    w = _csv.writer(f)
    if not done:
        w.writerow(["race_id", "pool", "combo", "odds"])
    got = 0
    for i, (d, m) in enumerate(todo, 1):
        q = {"serviceKey": KEY, "_type": "json", "numOfRows": "20000", "pageNo": "1",
             "meet": m, "rc_date": d}
        url = "https://apis.data.go.kr/B551015/API160_1/integratedInfo_1?" + "&".join(
            f"{k}={v if k=='serviceKey' else urllib.parse.quote(str(v))}" for k, v in q.items())
        try:
            raw = urllib.request.urlopen(urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"}), timeout=120, context=CTX
            ).read().decode("utf-8", "replace")
            it = (json.loads(raw).get("response", {}).get("body", {}) or {}).get("items") or {}
            it = it.get("item", []) if isinstance(it, dict) else []
        except Exception as e:
            print(f"    [실패] {d} meet={m}: {type(e).__name__}"); continue
        by_race = defaultdict(list)
        for r in it:
            by_race[f"{d}_{m}_{r.get('rcNo')}"].append(r)
        for rid, rs in by_race.items():
            if rid not in top:
                continue
            t = [int(x) for x in top[rid]]
            for r in rs:
                pool = str(r.get("pool", ""))
                c = [r.get("chulNo"), r.get("chulNo2"), r.get("chulNo3")]
                c = [int(x) for x in c if x not in (None, 0, "0")]
                hit = ((pool == "단승식" and c == t[:1]) or
                       (pool == "연승식" and c and c[0] in t[:3]) or
                       (pool == "복승식" and sorted(c) == sorted(t[:2])) or
                       (pool == "쌍승식" and c == t[:2]) or
                       (pool == "복연승식" and len(c) == 2 and set(c) <= set(t[:3])) or
                       (pool == "삼복승식" and sorted(c) == sorted(t[:3])) or
                       (pool == "삼쌍승식" and c == t[:3]))
                if hit:
                    w.writerow([rid, pool, "-".join(map(str, c)), r.get("odds")])
                    got += 1
        f.flush()
        if i % 40 == 0:
            print(f"    {i}/{len(todo)}  적중배당 {got:,}행", flush=True)
    f.close()
    print(f"  저장 {out_p}  이번 실행 {got:,}행")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["model", "game", "sim", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-odds", action="store_true")
    a = ap.parse_args()
    t = time.time()
    if a.what in ("model", "all"):
        print("═" * 60 + "\nmodel\n" + "═" * 60); build_model(a.limit)
    if a.what in ("sim", "all"):
        print("═" * 60 + "\nsim\n" + "═" * 60); build_sim(a.limit)
    if a.what in ("game", "all"):
        print("═" * 60 + "\ngame\n" + "═" * 60); build_game(a.limit, not a.no_odds)
    print(f"\n총 {time.time()-t:.0f}s")
