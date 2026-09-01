# -*- coding: utf-8 -*-
"""
육종가(EBV) as-of 조인.

★ 누수 주의 — 이 파일의 존재 이유
  육종가는 **그 말 자신의 경주성적을 포함**해 계산된다(설명 시트: "출주 시 개체의
  경주성적이 ... 합산되어 산출됨"). 2026-06-30 스냅샷을 2015년 경주에 붙이면
  모델이 그 말의 미래 성적 요약을 보게 된다. 지금까지 잡은 어떤 누수보다 크다.
  → 반드시 **경주일 직전 스냅샷**만 붙인다. 스냅샷 20개(2016-07 ~ 2026-06).

조인 키
  육종가 파일에는 마번(hrNo)이 없고 마명만 있다. 마명 단독은 위험하다 —
  마명 하나에 생년이 둘 이상인 동명이마가 18,981두 중 1,775두(9.3%)다.
  그래서 **(마명, 생년)** 을 쓴다. 생년은 원장에서 (경주년도 − 연령) 으로 복원하며,
  말별 중앙값을 쓰면 97.9% 가 단일값이고 육종가 파일의 생년과 98.7% 일치한다.

시트 우선순위
  같은 스냅샷에 경주마·육성마 양쪽에 있는 말이 3,800건. 경주마를 우선한다
  (실제 출주 성적이 반영된 값).
"""
import os
import numpy as np
import pandas as pd

EBV_LONG = "data/raw/ebv_long.csv"

# 말 본인 육종가 (경주마 > 육성마 우선)
OWN_SHEETS = ["경주마", "육성마"]
SIRE_SHEET = "씨수말"
DAM_SHEET = "씨암말"

EBV_COLS = ["ebv_prize", "ebv_prize_acc", "ebv_prize_sprint", "ebv_prize_route",
            "ebv_time", "ebv_time_acc", "ebv_time_sprint", "ebv_time_route",
            "ebv_ssgblup", "ebv_ssgblup_acc", "inbreeding"]


def recover_birth_year(df, name_col="hrName", id_col="hrNo",
                       date_col="rcDate", age_col="age"):
    """원장에서 생년 복원 — (경주년도 − 연령) 의 말별 중앙값.

    연말/연초 경계에서 ±1 흔들리는 말이 641두(2.1%) 있어 중앙값으로 굳힌다.
    """
    y = pd.to_numeric(df[date_col], errors="coerce") // 10000
    age = pd.to_numeric(df[age_col], errors="coerce")
    by = y - age
    return by.groupby(df[id_col]).transform("median")


def load_ebv():
    if not os.path.exists(EBV_LONG):
        print("  ebv_long.csv 없음 — tools/parse_ebv.py 먼저 실행. EBV 피처는 결측 유지")
        return None
    e = pd.read_csv(EBV_LONG, low_memory=False)
    e["hrName"] = e["hrName"].astype(str).str.strip()
    e["birth_yr"] = pd.to_numeric(e["birth_yr"], errors="coerce")
    return e


def _own_table(e):
    """말 본인 육종가 — (asof, 마명, 생년) 유일하게 정리."""
    own = e[e["sheet"].isin(OWN_SHEETS)].copy()
    own["_pri"] = own["sheet"].map({s: i for i, s in enumerate(OWN_SHEETS)})
    own = own.sort_values("_pri").drop_duplicates(["asof", "hrName", "birth_yr"])
    keep = ["asof", "hrName", "birth_yr"] + [c for c in EBV_COLS if c in own.columns]
    own = own[keep].copy()
    own["asof"] = own["asof"].astype("Float64")
    own["birth_yr"] = own["birth_yr"].astype("Float64")
    return own.sort_values("asof")


def _sire_table(e):
    """부마 육종가 — 씨수말 시트. 마명으로만 조인(생년 없음이 흔함)."""
    s = e[e["sheet"] == SIRE_SHEET].copy()
    s = s.drop_duplicates(["asof", "hrName"]).copy()
    s["asof"] = s["asof"].astype("Float64")
    cols = [c for c in ("ebv_prize", "ebv_prize_acc", "ebv_prize_sprint",
                        "ebv_prize_route", "ebv_ssgblup") if c in s.columns]
    out = s[["asof", "hrName"] + cols].rename(
        columns={c: "sire_" + c for c in cols})
    return out.sort_values("asof")


def attach(df, name_col="hrName", date_col="rcDate", birth_col="_birth_yr",
           sire_name_col=None):
    """df 에 as-of EBV 컬럼을 붙인다. df 는 rcDate 오름차순일 필요 없음.

    반환 컬럼: F2_ebv_* (말 본인), F2_sire_ebv_* (부마), F2_inbreeding
    """
    e = load_ebv()
    if e is None:
        return df, []

    own = _own_table(e)
    snaps = np.array(sorted(own["asof"].unique()))

    # 각 행에 '경주일 직전 스냅샷' 배정. searchsorted(left) → asof < rcDate 인 마지막 것
    rc = pd.to_numeric(df[date_col], errors="coerce").to_numpy()
    idx = np.searchsorted(snaps, rc, side="left") - 1
    snap = np.where(idx >= 0, snaps[np.clip(idx, 0, len(snaps) - 1)], np.nan)
    left = df[[name_col, birth_col]].copy()
    # merge 양쪽 dtype 을 맞춘다 (int/float 혼합 경고 방지)
    left[birth_col] = pd.to_numeric(left[birth_col], errors="coerce").astype("Float64")
    left["asof"] = pd.Series(snap).astype("Float64").to_numpy()
    left["_row"] = np.arange(len(left))

    m = left.merge(own, left_on=["asof", name_col, birth_col],
                   right_on=["asof", "hrName", "birth_yr"], how="left")
    m = m.sort_values("_row")

    added = []
    ren = {"ebv_prize": "F2_ebv_prize", "ebv_prize_acc": "F2_ebv_acc",
           "ebv_prize_sprint": "F2_ebv_sprint", "ebv_prize_route": "F2_ebv_route",
           "ebv_time": "F2_ebv_time", "ebv_time_acc": "F2_ebv_time_acc",
           "ebv_ssgblup": "F2_ebv_ssgblup", "inbreeding": "F2_inbreeding"}
    for src, dst in ren.items():
        if src in m.columns:
            df[dst] = m[src].to_numpy()
            added.append(dst)

    # 부마 육종가 — 원장엔 부마명이 없으므로 aux_pedigree 의 faHrName 을 넘겨받아야 한다
    if sire_name_col and sire_name_col in df.columns:
        sire = _sire_table(e)
        ls = pd.DataFrame({"asof": pd.Series(snap).astype("Float64").to_numpy(),
                           "hrName": df[sire_name_col].astype(str).str.strip().to_numpy()})
        ls["_row"] = np.arange(len(ls))
        ms = ls.merge(sire, on=["asof", "hrName"], how="left").sort_values("_row")
        for c in ms.columns:
            if c.startswith("sire_ebv_"):
                dst = "F2_" + c
                df[dst] = ms[c].to_numpy()
                added.append(dst)
    return df, added


def report(df, added, split_col="split"):
    if not added:
        print("  EBV 컬럼 없음")
        return
    print("  EBV as-of 조인 결과")
    base = added[0]
    hit = df[base].notna()
    print(f"    전체 매칭 {hit.mean()*100:.1f}%")
    y = pd.to_numeric(df["rcDate"], errors="coerce") // 10000
    for yr in sorted(y.dropna().unique()):
        if yr < 2016:
            continue
        g = hit[y == yr]
        if len(g):
            print(f"      {int(yr)}: {g.mean()*100:5.1f}%")
    print(f"    붙은 컬럼 {len(added)}개: {', '.join(added)}")
