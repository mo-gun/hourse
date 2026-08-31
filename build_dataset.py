# -*- coding: utf-8 -*-
"""
2단계 — 학습 데이터셋 빌더.

data/ledger_2010_2026.csv (원장) → dataset/kra_ml_v1.csv (학습용)

핵심 원칙 — 누수 방지
  as-of 집계는 전부 **시간순 단일 패스**로 계산한다.
  각 행에서 "먼저 현재까지의 누적값을 피처로 기록(emit) → 그 다음 이 경주 결과로 갱신(update)".
  이 순서 때문에 구조적으로 미래를 볼 수 없다. groupby.shift 류보다 검증하기 쉽다.

기준기록(par)은 **train 구간(2010~2024)에서만** 산출해 전 구간에 적용한다.
valid/test 기록으로 par 를 만들면 그 자체가 누수다.

실행
  python build_dataset.py
  python build_dataset.py --limit 50000     # 빠른 검증용
"""
import os, sys, json, math, argparse, time
from collections import defaultdict, deque
from datetime import date

sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import numpy as np

import schema as S

LEDGER = "data/ledger_2010_2026.csv"
OUTDIR = "dataset"
OUT_CSV = f"{OUTDIR}/kra_ml_v1.csv"
OUT_PARQUET = f"{OUTDIR}/kra_ml_v1.parquet"

W365 = 365
MEET_CODE = {"서울": 1, "제주": 2, "부산경남": 3, "1": 1, "2": 2, "3": 3}
ORD_MAX = 16          # 실측 최대 출주두수. 91~99 는 취소·실격 센티널이다

# 구간 시간 컬럼 — 경마장별. S1F(초반 1F) 와 G3F(상 3F).
# 컬럼명이 AccTime 이지만 누적/구간 의미가 경마장마다 섞여 있어 절대값 비교는 하지 않는다.
# (경마장 × 거리) 기준기록 대비 편차로만 쓰므로 계열 내 일관성만 있으면 된다.
SECT_TIME = {
    1: {"s1f": "seS1fAccTime", "g3f": "seG3fAccTime"},
    2: {"s1f": "jeS1fTime",    "g3f": "jeG3fTime"},
    3: {"s1f": "buS1fAccTime", "g3f": "buG3fAccTime"},
}


# ══════════════════════════════════════════════════════════════════════
# 파싱 헬퍼
# ══════════════════════════════════════════════════════════════════════
def f(x, default=np.nan):
    try:
        v = float(str(x).strip())
        return v
    except Exception:
        return default


def parse_wg(s):
    """'465(+9)' → (465.0, 9.0) / '470()' → (470.0, nan) / '0()' → (nan, nan)"""
    s = str(s).strip()
    if not s or s in ("-", "0()"):
        return np.nan, np.nan
    w, d = np.nan, np.nan
    if "(" in s:
        head, _, tail = s.partition("(")
        w = f(head)
        tail = tail.rstrip(")").strip()
        if tail:
            d = f(tail.replace("+", ""))
    else:
        w = f(s)
    if w is not None and not np.isnan(w) and w <= 0:
        w = np.nan
    return w, d


def parse_track(s):
    """'포화 (16%)' → ('포화', 16.0) / ' (0%)' → (nan, nan)"""
    s = str(s).strip()
    if not s or s.startswith("(0%)") or s in ("-",):
        return np.nan, np.nan
    state, _, rest = s.partition("(")
    state = state.strip()
    moist = f(rest.replace("%", "").replace(")", "").strip())
    if state == "":
        return np.nan, np.nan
    if moist == 0 and state == "":
        return np.nan, np.nan
    return state, moist


_MARGIN = {"-": 0.0, "동착": 0.0, "코": 0.05, "머리": 0.1, "머": 0.1,
           "목": 0.25, "간": 0.05, "½": 0.5, "¾": 0.75, "¼": 0.25,
           "대차": 15.0, "완승": 10.0}


def parse_margin(s):
    """착차 '2½' → 2.5 / '코' → 0.05 / '-' → 0.0"""
    s = str(s).strip()
    if not s:
        return np.nan
    if s in _MARGIN:
        return _MARGIN[s]
    num, frac = "", 0.0
    for ch in s:
        if ch.isdigit() or ch == ".":
            num += ch
        elif ch in _MARGIN:
            frac += _MARGIN[ch]
    if num:
        return float(num) + frac
    return frac if frac else np.nan


def dist_band(d):
    if d != d:
        return "unk"
    d = int(d)
    if d <= 1200:
        return "S"          # 단거리
    if d <= 1400:
        return "SM"
    if d <= 1700:
        return "M"
    if d <= 1900:
        return "ML"
    return "L"              # 장거리


# ══════════════════════════════════════════════════════════════════════
# 구간기록 → 초/중/종반 순위
#   경마장마다 컬럼이 완전히 다르고 상호 배타적이다.
#   시간 컬럼은 의미가 모호한 사례가 있어(1000m 경주에서 se_1cAccTime=0)
#   **순위 컬럼**을 1순위로 쓰고, 순위가 없는 제주는 시간으로 경주 내 순위를 만든다.
# ══════════════════════════════════════════════════════════════════════
SECT = {
    1: {"early": ["sjS1fOrd"], "mid": ["sj_3cOrd", "sj_2cOrd", "sj_1cOrd"],
        "late": ["sjG3fOrd", "sj_4cOrd"]},
    3: {"early": ["buS1fOrd"], "mid": ["buG8fOrd", "buG6fOrd", "buG4fOrd"],
        "late": ["buG3fOrd", "buG2fOrd"]},
    2: {"early": ["jeS1fTime"], "mid": ["je_2cTime", "je_1cTime"],
        "late": ["jeG3fTime"], "_time": True},
}


def add_section_positions(df):
    """early/mid/late 순위를 통합 컬럼으로. 없으면 NaN."""
    for slot in ("early", "mid", "late"):
        df[f"_sect_{slot}"] = np.nan
    for meet, cfg in SECT.items():
        m = df["meet"] == meet
        if not m.any():
            continue
        is_time = cfg.get("_time", False)
        for slot in ("early", "mid", "late"):
            val = pd.Series(np.nan, index=df.index[m])
            for col in cfg[slot]:
                if col not in df.columns:
                    continue
                v = pd.to_numeric(df.loc[m, col], errors="coerce")
                v = v.where(v > 0)                       # 0 = 해당 없음
                val = val.fillna(v)
            if is_time:
                # 제주는 시간만 → 경주 내 시간 순위로 환산
                tmp = pd.DataFrame({"race_id": df.loc[m, "race_id"], "v": val})
                val = tmp.groupby("race_id")["v"].rank(method="min")
            df.loc[m, f"_sect_{slot}"] = val.values
    # 상대위치 0(선두)~1(최후미)
    for slot in ("early", "mid", "late"):
        o = df[f"_sect_{slot}"]
        df[f"_pos_{slot}"] = (o - 1) / (df["dusu"] - 1).replace(0, np.nan)

    # 구간 시간 → (경마장×거리) 기준 대비 편차. 양수 = 기준보다 빠름.
    for slot in ("s1f", "g3f"):
        raw = pd.Series(np.nan, index=df.index)
        for meet, cfg in SECT_TIME.items():
            col = cfg[slot]
            if col not in df.columns:
                continue
            m = df["meet"] == meet
            v = pd.to_numeric(df.loc[m, col], errors="coerce")
            raw.loc[m] = v.where(v > 0).values
        # 기준기록은 train 구간에서만 산출한다 (valid/test 로 만들면 그 자체가 누수)
        tr = df["split"] == "train"
        par = raw[tr].groupby([df.loc[tr, "meet"], df.loc[tr, "rcDist"]]).median()
        key = pd.MultiIndex.from_arrays([df["meet"], df["rcDist"]])
        df[f"_t_{slot}"] = par.reindex(key).to_numpy() - raw.to_numpy()
    return df


# ══════════════════════════════════════════════════════════════════════
def load_ledger(limit=None):
    print("원장 로드...", flush=True)
    df = pd.read_csv(LEDGER, dtype=str, encoding="utf-8-sig",
                     nrows=limit, low_memory=False)
    print(f"  {len(df):,}행 × {len(df.columns)}열")

    df["rcDate"] = pd.to_numeric(df["rcDate"], errors="coerce").astype("Int64")
    # 원장의 meet 은 숫자가 아니라 경마장 이름이다 ('서울'/'제주'/'부산경남').
    # API 파라미터 코드와 같은 번호로 정규화한다.
    df["meet"] = df["meet"].astype(str).str.strip().map(MEET_CODE)
    for c in ("rcNo", "chulNo", "age", "ilsu", "rcDist"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["ord"] = pd.to_numeric(df["ord"], errors="coerce")
    df["rcTime"] = pd.to_numeric(df["rcTime"], errors="coerce")
    df["wgBudam"] = pd.to_numeric(df["wgBudam"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").replace(0, np.nan)
    df["chaksun1"] = pd.to_numeric(df["chaksun1"], errors="coerce")
    df["winOdds"] = pd.to_numeric(df["winOdds"], errors="coerce").replace(0, np.nan)
    df["plcOdds"] = pd.to_numeric(df["plcOdds"], errors="coerce").replace(0, np.nan)

    wg = df["wgHr"].map(parse_wg)
    df["X_wgHr"] = [a for a, _ in wg]
    df["X_wgHr_delta"] = [b for _, b in wg]
    tr = df["track"].map(parse_track)
    df["F4_track_state"] = [a for a, _ in tr]
    df["F4_track_moist"] = [b for _, b in tr]
    df["F4_weather"] = df["weather"].replace({"-": np.nan, "": np.nan})
    df["_margin"] = df["diffUnit"].map(parse_margin)

    # 유효 완주행만.
    #   ord 0      = 미출주
    #   ord 91~99  = 센티널(출전취소·경주중지·실격·낙마 등). 실측 분포상 정상 착순은 1~16 뿐이고
    #                91,92,93,94,95,98,99 만 별도로 존재한다. 이걸 남기면 dusu 와 착순이 전부 깨진다.
    before = len(df)
    df = df[df["ord"].between(1, ORD_MAX)].copy()
    df = df[df["rcDate"].notna() & df["hrNo"].notna()].copy()
    print(f"  유효 완주 {len(df):,}행 (제외 {before-len(df):,} — 미출주·취소·실격)")

    df["race_id"] = (df["rcDate"].astype(str) + "_" +
                     df["meet"].astype("Int64").astype(str) + "_" +
                     df["rcNo"].astype("Int64").astype(str))
    df["dusu"] = df.groupby("race_id")["hrNo"].transform("size")
    df["row_id"] = df["race_id"] + "_" + df["chulNo"].astype("Int64").astype(str)
    df["_ord_ordinal"] = [date(int(d) // 10000, int(d) // 100 % 100, int(d) % 100).toordinal()
                          for d in df["rcDate"]]
    df["F4_dist_band"] = df["rcDist"].map(dist_band)
    df["_wet"] = (df["F4_track_moist"] >= 10).astype(float)
    df = df.sort_values(["_ord_ordinal", "meet", "rcNo", "chulNo"]).reset_index(drop=True)
    return df


def add_targets_and_market(df):
    df["y_ord"] = df["ord"].astype(int)
    df["y_win"] = (df["y_ord"] == 1).astype(int)
    plc_cut = np.where(df["dusu"] >= 7, 3, 2)
    df["y_plc"] = (df["y_ord"] <= plc_cut).astype(int)
    df["y_rel"] = np.maximum(0, df["dusu"] - df["y_ord"]).astype(int)
    df["y_rcTime"] = df["rcTime"]

    # 시장 — 오버라운드 제거 후 경주 내 합=1 로 정규화
    df["F6X_win_odds"] = df["winOdds"]
    df["F6X_plc_odds"] = df["plcOdds"]
    inv = 1.0 / df["winOdds"]
    s = inv.groupby(df["race_id"]).transform("sum")
    df["F6X_mkt_prob"] = inv / s
    df["F6X_overround"] = s - 1.0
    df["F6X_mkt_rank"] = df.groupby("race_id")["winOdds"].rank(method="min").astype("Int64")
    return df


def add_par_and_speed(df):
    """기준기록(par)은 train 구간에서만 산출한다. valid/test 로 par 를 만들면 누수."""
    tr = df[df["split"] == "train"]
    key = ["meet", "rcDist", "F4_track_state"]
    par = tr.groupby(key, dropna=False)["rcTime"].agg(["median", "std", "size"])
    par = par[par["size"] >= 20]
    # 주로상태별 표본이 없으면 (meet, dist) 로 폴백
    par2 = tr.groupby(["meet", "rcDist"], dropna=False)["rcTime"].agg(["median", "std", "size"])
    par2 = par2[par2["size"] >= 20]

    df = df.merge(par[["median", "std"]].rename(columns={"median": "_par", "std": "_parsd"}),
                  left_on=key, right_index=True, how="left")
    df = df.merge(par2[["median", "std"]].rename(columns={"median": "_par2", "std": "_parsd2"}),
                  left_on=["meet", "rcDist"], right_index=True, how="left")
    df["_par"] = df["_par"].fillna(df["_par2"])
    df["_parsd"] = df["_parsd"].fillna(df["_parsd2"]).replace(0, np.nan)
    # 빠를수록 높은 점수. 100 기준 ±10 스케일
    df["y_speed_fig"] = 100 + 10 * (df["_par"] - df["rcTime"]) / df["_parsd"]
    df["y_speed_fig"] = df["y_speed_fig"].clip(40, 160)
    print(f"  par 테이블 {len(par):,}조합 (+폴백 {len(par2):,})  "
          f"speed_fig 산출 {df['y_speed_fig'].notna().mean()*100:.0f}%")
    return df


def assign_split(df):
    d = df["rcDate"].astype(int)
    df["split"] = "train"
    for name, (a, b) in S.SPLIT.items():
        df.loc[(d >= a) & (d <= b), "split"] = name
    return df


# ══════════════════════════════════════════════════════════════════════
# as-of 롤링 — 시간순 단일 패스. emit → update.
# ══════════════════════════════════════════════════════════════════════
class Win:
    """365일 이동창 승/복승 카운터."""
    __slots__ = ("q", "n", "w", "p")

    def __init__(self):
        self.q, self.n, self.w, self.p = deque(), 0, 0, 0

    def expire(self, today):
        while self.q and today - self.q[0][0] > W365:
            _, w, p = self.q.popleft()
            self.n -= 1; self.w -= w; self.p -= p

    def add(self, day, w, p):
        self.q.append((day, w, p)); self.n += 1; self.w += w; self.p += p


def _rate(w, n):
    return w / n if n else np.nan


def rolling_features(df):
    n = len(df)
    print(f"as-of 롤링 단일 패스 ({n:,}행)...", flush=True)

    # ── 상태 저장소 ──
    hr_hist = defaultdict(list)          # hrNo -> [(day, ord, dusu, dist, speed, wet, grade, margin, prize, posE, posL)]
    hr_life = defaultdict(lambda: [0, 0, 0])          # n, win, plc
    hr_dist = defaultdict(lambda: [0, 0])             # (hr, band) -> n, win
    hr_meet = defaultdict(lambda: [0, 0])
    hr_wet = defaultdict(lambda: [0, 0])
    hr_w365 = defaultdict(Win)

    jk_life = defaultdict(lambda: [0, 0, 0])
    jk_w365 = defaultdict(Win)
    jk_meet = defaultdict(lambda: [0, 0])
    jk_dist = defaultdict(lambda: [0, 0])
    jk_ordpct = defaultdict(lambda: [0.0, 0])
    tr_life = defaultdict(lambda: [0, 0, 0])
    tr_w365 = defaultdict(Win)
    jkhr = defaultdict(lambda: [0, 0])
    jktr = defaultdict(lambda: [0, 0])

    out = {c: np.full(n, np.nan) for c in ROLL_NUM}
    out_s = {c: np.array([None] * n, dtype=object) for c in ROLL_STR}

    day_a = df["_ord_ordinal"].to_numpy()
    ord_a = df["y_ord"].to_numpy()
    dusu_a = df["dusu"].to_numpy()
    win_a = df["y_win"].to_numpy()
    plc_a = df["y_plc"].to_numpy()
    dist_a = df["rcDist"].to_numpy()
    band_a = df["F4_dist_band"].to_numpy()
    meet_a = df["meet"].to_numpy()
    wet_a = df["_wet"].to_numpy()
    spd_a = df["y_speed_fig"].to_numpy()
    grade_a = df["rank"].to_numpy()
    marg_a = df["_margin"].to_numpy()
    prize_a = df["chaksun1"].to_numpy()
    posE_a = df["_pos_early"].to_numpy()
    posM_a = df["_pos_mid"].to_numpy()
    posL_a = df["_pos_late"].to_numpy()
    ts1f_a = df["_t_s1f"].to_numpy()
    tg3f_a = df["_t_g3f"].to_numpy()
    hr_a = df["hrNo"].to_numpy()
    jk_a = df["jkNo"].to_numpy()
    trn_a = df["trNo"].to_numpy()

    t0 = time.time()
    for i in range(n):
        day = day_a[i]; hr = hr_a[i]; jk = jk_a[i]; tn = trn_a[i]
        band = band_a[i]; mt = meet_a[i]

        # ───────── emit : 지금까지의 누적값만 사용 ─────────
        h = hr_life[hr]
        out["F1_starts_life"][i] = h[0]
        out["F1_win_rate_life"][i] = _rate(h[1], h[0])
        out["F1_plc_rate_life"][i] = _rate(h[2], h[0])
        out["F1_first_start"][i] = 1.0 if h[0] == 0 else 0.0
        w = hr_w365[hr]; w.expire(day)
        out["F1_starts_365"][i] = w.n
        out["F1_win_rate_365"][i] = _rate(w.w, w.n)

        hist = hr_hist[hr]
        if hist:
            last = hist[-1]
            out["F1_ord_last1"][i] = last[1]
            out["F1_layoff_days"][i] = day - last[0]
            out_s["F1_grade_last"][i] = last[6]
            g_prev, g_now = _grade_num(last[6]), _grade_num(grade_a[i])
            if g_prev is not None and g_now is not None:
                out["F1_grade_move"][i] = g_prev - g_now      # 등급 숫자↓ = 승급
            if len(hist) >= 2:
                out["F1_ord_last2"][i] = hist[-2][1]
            if len(hist) >= 3:
                out["F1_ord_last3"][i] = hist[-3][1]
            last3 = hist[-3:]
            out["F1_ord_avg3"][i] = np.mean([x[1] for x in last3])
            sp3 = [x[4] for x in last3 if x[4] == x[4]]
            if sp3:
                out["F1_speed_avg3"][i] = np.mean(sp3)
            mg3 = [x[7] for x in last3 if x[7] == x[7]]
            if mg3:
                out["F1_margin_avg3"][i] = np.mean(mg3)
            last5 = hist[-5:]
            out["F1_ordpct_avg5"][i] = np.mean(
                [(x[1] - 1) / (x[2] - 1) if x[2] > 1 else np.nan for x in last5])
            recent = [x for x in hist if day - x[0] <= W365]
            sp = [x[4] for x in recent if x[4] == x[4]]
            if sp:
                out["F1_speed_best365"][i] = max(sp)
            out["F1_prize_365"][i] = sum(x[8] for x in recent if x[8] == x[8])
            # 거리 적성
            bd = [x for x in hist if x[3] == x[3] and dist_band(x[3]) == band]
            if bd:
                out["F4_hr_dist_speed"][i] = np.nanmean([x[4] for x in bd]) if any(
                    x[4] == x[4] for x in bd) else np.nan
            best = min(hist, key=lambda x: (x[1], -(x[4] if x[4] == x[4] else -9e9)))
            out["F4_hr_best_dist"][i] = best[3]
            if best[3] == best[3] and dist_a[i] == dist_a[i]:
                out["F4_dist_gap"][i] = abs(dist_a[i] - best[3])
            if last[3] == last[3] and dist_a[i] == dist_a[i]:
                out["F4_dist_change"][i] = dist_a[i] - last[3]
            # 주행 스타일 — 과거 구간위치만
            pe = [x[9] for x in hist if x[9] == x[9]]
            pm = [x[10] for x in hist if x[10] == x[10]]
            pl = [x[11] for x in hist if x[11] == x[11]]
            if pe:
                mE = float(np.mean(pe))
                out["F5_early_pos"][i] = mE
                out["F5_style_consistency"][i] = float(np.std(pe))
                out_s["F5_style"][i] = ("선행" if mE < 0.20 else "선입" if mE < 0.45
                                        else "중단" if mE < 0.70 else "추입")
                fin = [(x[1] - 1) / (x[2] - 1) if x[2] > 1 else np.nan for x in hist if x[9] == x[9]]
                gains = [a - b for a, b in zip(pe, fin) if b == b]
                if gains:
                    out["F5_pos_gain"][i] = float(np.mean(gains))
            if pm:
                out["F5_mid_pos"][i] = float(np.mean(pm))
            if pl:
                out["F5_late_pos"][i] = float(np.mean(pl))
            out["F5_sect_n"][i] = len(pe)
            ts = [x[12] for x in hist if x[12] == x[12]]
            if ts:
                out["F5_s1f_time"][i] = float(np.mean(ts))
            tg = [x[13] for x in hist if x[13] == x[13]]
            if tg:
                out["F5_g3f_time"][i] = float(np.mean(tg))
                out["F5_g3f_best"][i] = float(np.max(tg))

        d = hr_dist[(hr, band)]
        out["F4_hr_dist_starts"][i] = d[0]
        out["F4_hr_dist_win_rate"][i] = _rate(d[1], d[0])
        m_ = hr_meet[(hr, mt)]
        out["F4_hr_meet_starts"][i] = m_[0]
        out["F4_hr_meet_win_rate"][i] = _rate(m_[1], m_[0])
        wt = hr_wet[hr]
        out["F4_hr_wet_starts"][i] = wt[0]
        out["F4_hr_wet_win_rate"][i] = _rate(wt[1], wt[0])

        j = jk_life[jk]
        out["F3_jk_starts_life"][i] = j[0]
        out["F3_jk_win_rate_life"][i] = _rate(j[1], j[0])
        out["F3_jk_plc_rate_life"][i] = _rate(j[2], j[0])
        jw = jk_w365[jk]; jw.expire(day)
        out["F3_jk_starts_365"][i] = jw.n
        out["F3_jk_win_rate_365"][i] = _rate(jw.w, jw.n)
        jm = jk_meet[(jk, mt)]
        out["F3_jk_win_rate_meet"][i] = _rate(jm[1], jm[0])
        jd = jk_dist[(jk, band)]
        out["F3_jk_win_rate_dist"][i] = _rate(jd[1], jd[0])
        jo = jk_ordpct[jk]
        out["F3_jk_ordpct_avg"][i] = jo[0] / jo[1] if jo[1] else np.nan
        t_ = tr_life[tn]
        out["F3_tr_win_rate_life"][i] = _rate(t_[1], t_[0])
        tw = tr_w365[tn]; tw.expire(day)
        out["F3_tr_starts_365"][i] = tw.n
        out["F3_tr_win_rate_365"][i] = _rate(tw.w, tw.n)
        p = jkhr[(jk, hr)]
        out["F3_jkhr_starts"][i] = p[0]
        out["F3_jkhr_win_rate"][i] = _rate(p[1], p[0])
        out["F3_jkhr_first"][i] = 1.0 if p[0] == 0 else 0.0
        q = jktr[(jk, tn)]
        out["F3_jktr_win_rate"][i] = _rate(q[1], q[0])

        # ───────── update : 이 경주 결과를 반영 ─────────
        wv, pv = int(win_a[i]), int(plc_a[i])
        h[0] += 1; h[1] += wv; h[2] += pv
        w.add(day, wv, pv)
        d[0] += 1; d[1] += wv
        m_[0] += 1; m_[1] += wv
        if wet_a[i] == 1:
            wt[0] += 1; wt[1] += wv
        j[0] += 1; j[1] += wv; j[2] += pv
        jw.add(day, wv, pv)
        jm[0] += 1; jm[1] += wv
        jd[0] += 1; jd[1] += wv
        if dusu_a[i] > 1:
            jo[0] += (ord_a[i] - 1) / (dusu_a[i] - 1); jo[1] += 1
        t_[0] += 1; t_[1] += wv; t_[2] += pv
        tw.add(day, wv, pv)
        p[0] += 1; p[1] += wv
        q[0] += 1; q[1] += wv
        hist.append((day, ord_a[i], dusu_a[i], dist_a[i], spd_a[i], wet_a[i],
                     grade_a[i], marg_a[i], prize_a[i] if wv else 0.0,
                     posE_a[i], posM_a[i], posL_a[i], ts1f_a[i], tg3f_a[i]))
        if len(hist) > 40:                 # 메모리 상한 — 40경주면 모든 창을 덮는다
            del hist[:-40]

        if i and i % 100000 == 0:
            print(f"    {i:,}/{n:,}  {time.time()-t0:.0f}s", flush=True)

    for c, v in out.items():
        df[c] = v
    for c, v in out_s.items():
        df[c] = v
    print(f"  완료 {time.time()-t0:.0f}s")
    return df


def _grade_num(g):
    """'국6' '혼4' '국6등급' → 6, 4, 6.  숫자가 곧 등급 수준(작을수록 상위).

    국(국산)·혼(혼합)·오픈이 섞여 있어 계열 간 정확한 서열은 아니지만,
    같은 말의 **연속 두 경주 사이 이동 방향**을 보는 용도라 숫자 비교로 충분하다.
    """
    s = str(g).strip()
    if not s or s == "nan":
        return None
    for ch in s:
        if ch.isdigit():
            return int(ch)
    return None


ROLL_NUM = [
    "F1_starts_life", "F1_win_rate_life", "F1_plc_rate_life", "F1_starts_365",
    "F1_win_rate_365", "F1_ord_last1", "F1_ord_last2", "F1_ord_last3", "F1_ord_avg3",
    "F1_ordpct_avg5", "F1_speed_avg3", "F1_speed_best365", "F1_margin_avg3",
    "F1_prize_365", "F1_grade_move", "F1_layoff_days", "F1_first_start",
    "F3_jk_starts_life", "F3_jk_win_rate_life", "F3_jk_plc_rate_life",
    "F3_jk_starts_365", "F3_jk_win_rate_365", "F3_jk_win_rate_meet",
    "F3_jk_win_rate_dist", "F3_jk_ordpct_avg", "F3_tr_starts_365",
    "F3_tr_win_rate_365", "F3_tr_win_rate_life", "F3_jkhr_starts",
    "F3_jkhr_win_rate", "F3_jkhr_first", "F3_jktr_win_rate",
    "F4_hr_dist_starts", "F4_hr_dist_win_rate", "F4_hr_dist_speed",
    "F4_hr_best_dist", "F4_dist_gap", "F4_dist_change", "F4_hr_meet_starts",
    "F4_hr_meet_win_rate", "F4_hr_wet_starts", "F4_hr_wet_win_rate",
    "F5_early_pos", "F5_mid_pos", "F5_late_pos", "F5_pos_gain",
    "F5_style_consistency", "F5_sect_n", "F5_s1f_time", "F5_g3f_time", "F5_g3f_best",
]
ROLL_STR = ["F1_grade_last", "F5_style"]


def add_race_level_pace(df):
    """경주 레벨 F5 — 선행마 두수와 페이스 압력. 전부 as-of 피처로만 계산."""
    isf = (df["F5_early_pos"] < 0.25).astype(float)
    df["F5_race_n_front"] = isf.groupby(df["race_id"]).transform("sum")
    press = (1.0 - df["F5_early_pos"]).clip(lower=0)
    df["F5_race_pace_press"] = press.groupby(df["race_id"]).transform("mean")
    # 내 각질이 이 구도에서 유리한가: 페이스 빠르면(선행 많으면) 추입 유리
    df["F5_style_fit"] = (df["F5_race_n_front"] - 1).clip(lower=0) * df["F5_early_pos"].fillna(0.5)
    gate_rel = df["chulNo"] / df["dusu"]
    df["X_gate_rel"] = gate_rel
    df["F5_gate_style_fit"] = -(gate_rel * (1 - df["F5_early_pos"].fillna(0.5)))
    return df


def finalize(df):
    df["X_age"] = df["age"]
    df["X_sex"] = df["sex"]
    df["X_prd_cty"] = df["name"]                 # 산지는 prd 가 아니라 name 컬럼
    df["X_rcDist"] = df["rcDist"]
    df["X_grade"] = df["rank"]
    df["X_chulNo"] = df["chulNo"]
    df["X_dusu"] = df["dusu"]
    df["X_wgBudam"] = df["wgBudam"]
    df["X_ilsu"] = df["ilsu"]
    df["X_rating"] = df["rating"]
    df["X_prize_cond"] = df["chaksun1"]
    df["F4_month"] = (df["rcDate"].astype(int) // 100) % 100

    # 아직 소스가 없는 컬럼은 스키마 유지를 위해 빈 값으로 만든다
    pending = [c for c in S.features() if c not in df.columns]
    if pending:
        df = pd.concat(
            [df, pd.DataFrame(np.nan, index=df.index, columns=pending)], axis=1)

    cols = ([c["name"] for c in S.INDEX] + [c["name"] for c in S.TARGETS] +
            S.features() + [c["name"] for c in S.F6X])
    cols = [c for c in cols if c in df.columns]
    return df[cols], pending


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-parquet", action="store_true")
    a = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()
    df = load_ledger(a.limit)
    df = assign_split(df)
    df = add_targets_and_market(df)
    df = add_par_and_speed(df)
    df = add_section_positions(df)
    df = rolling_features(df)
    df = add_race_level_pace(df)
    out, pending = finalize(df)

    # 누수 검사 — 금지 필드가 결과에 남아있으면 즉시 실패
    bad = [c for c in out.columns if c in S.FORBIDDEN_SOURCE_FIELDS]
    assert not bad, f"누수 컬럼 발견: {bad}"

    # float_format 은 파일 크기를 위한 것. 유효숫자 6자리면 승률·시간 피처에 충분하고
    # 405MB → 247MB, gzip 후 76MB 로 GitHub 100MB 제한 안에 들어온다.
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig", float_format="%.6g")
    print(f"\n저장 {OUT_CSV}  {len(out):,}행 × {len(out.columns)}열  "
          f"({os.path.getsize(OUT_CSV)/1e6:.0f} MB)")
    import gzip, shutil
    with open(OUT_CSV, "rb") as a, gzip.open(OUT_CSV + ".gz", "wb", compresslevel=9) as b:
        shutil.copyfileobj(a, b)
    print(f"저장 {OUT_CSV}.gz  ({os.path.getsize(OUT_CSV + '.gz')/1e6:.0f} MB)  "
          f"— 팀 공유본. pd.read_csv 가 .gz 를 그대로 읽는다")
    if not a.no_parquet:
        try:
            out.to_parquet(OUT_PARQUET, index=False)
            print(f"저장 {OUT_PARQUET}  ({os.path.getsize(OUT_PARQUET)/1e6:.0f} MB)")
        except Exception as e:
            print(f"parquet 실패(무시): {type(e).__name__}")

    print("\n분할")
    for k, v in out["split"].value_counts().items():
        rr = out[out["split"] == k]
        print(f"  {k:6s} {v:>8,}행  {rr['race_id'].nunique():>6,}경주  "
              f"{int(rr['rcDate'].min())}~{int(rr['rcDate'].max())}")
    print(f"\n미구현 컬럼 {len(pending)}개 (조교·장제·혈통·인기도서브모델): "
          f"{', '.join(pending[:6])}...")
    print(f"총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
