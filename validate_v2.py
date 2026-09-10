# -*- coding: utf-8 -*-
"""
v2 검증 — 6명이 이 데이터를 믿으려면 통과해야 하는 관문.

v1 검증에 더해 v2 고유 항목 두 가지를 본다.
  · 게임풀 격리 — game 의 경주가 train/valid/test 에 단 한 건도 없어야 한다
  · 육종가 as-of — 붙은 스냅샷이 항상 경주일 **이전**이어야 한다

실행: python validate_v2.py
"""
import sys, json, io, glob
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd

import schema_v2 as S
from build_v2 import LEDGER, MEET_CODE, MEET_NAME, ORD_MAX

FAIL, WARN = [], []


def ck(ok, label, detail=""):
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAIL.append(label)


def wn(ok, label, detail=""):
    print(f"  [{'OK  ' if ok else 'WARN'}] {label}" + (f"  — {detail}" if detail else ""))
    if not ok:
        WARN.append(label)


def main():
    parts = {}
    for sp in ("train", "valid", "test", "game"):
        p = f"{S.MODEL_DIR}/{sp}.parquet"
        parts[sp] = pd.read_parquet(p)
    df = pd.concat(parts.values(), ignore_index=True)
    print(f"model 합계 {len(df):,}행 × {len(df.columns)}열\n")

    print("1. 스키마")
    want = ([c["name"] for c in S.INDEX] + [c["name"] for c in S.TARGETS] +
            [c for c in S.features()])
    miss = [c for c in want if c not in df.columns]
    extra = [c for c in df.columns if c not in want]
    ck(not miss, "스키마 컬럼 전부 존재", f"누락 {miss[:5]}")
    ck(not extra, "스키마 외 컬럼 없음", f"초과 {extra[:5]}")
    ck(not [c for c in df.columns if c in S.FORBIDDEN], "누수 금지 필드 미포함")
    ck("X_ilsu" not in df.columns, "X_ilsu 제거됨 (개최일 순번이라 폐기)")

    print("\n2. 게임풀 격리 — v2 핵심")
    ho = json.load(io.open(S.HOLDOUT_FILE, encoding="utf-8"))
    hd = set(ho["days"])
    gm = set(parts["game"]["race_id"])
    other = set(parts["train"]["race_id"]) | set(parts["valid"]["race_id"]) | set(parts["test"]["race_id"])
    ck(not (gm & other), "game 경주가 train/valid/test 에 없음",
       f"겹침 {len(gm & other)}건")
    d = parts["game"]["rcDate"].astype(int).astype(str) + "_" + \
        parts["game"]["meet"].astype(int).map(MEET_NAME)
    ck(set(d) <= hd, "game 이 예약 개최일 안에만 있음")
    od = parts["train"]["rcDate"].astype(int).astype(str) + "_" + \
        parts["train"]["meet"].astype(int).map(MEET_NAME)
    ck(not (set(od) & hd), "train 에 예약 개최일이 섞이지 않음",
       f"침범 {len(set(od) & hd)}일")
    ck(len(gm) == ho["n_races"], "게임풀 경주 수가 예약 파일과 일치",
       f"{len(gm):,} vs {ho['n_races']:,}")

    print("\n3. 분할 · 키")
    ck(df["row_id"].is_unique, "row_id 고유")
    g = df.groupby("race_id")
    ck((g["split"].nunique() == 1).all(), "한 경주가 두 split 에 안 걸침")
    ck((g["y_ord"].min() == 1).all(), "모든 경주에 1착 존재")
    ck((df["X_dusu"] == g["row_id"].transform("size")).all(), "X_dusu = 경주별 행 수")
    for sp in ("train", "valid", "test"):
        s = parts[sp]
        print(f"       {sp:6s} {len(s):>7,}행 {s['race_id'].nunique():>6,}경주 "
              f"{int(s['rcDate'].min())}~{int(s['rcDate'].max())}")
    order = (parts["train"]["rcDate"].max() <= parts["valid"]["rcDate"].min()
             and parts["valid"]["rcDate"].max() <= parts["test"]["rcDate"].min())
    ck(order, "train < valid < test 시간 순서")
    wn(parts["test"]["race_id"].nunique() >= 500,
       "test 경주 수 500 이상 (6명 비교 검정력)",
       f"{parts['test']['race_id'].nunique()}경주 — 차이 5%p 미만은 구분 어려움")

    print("\n4. 누수 재현 — 원장에서 독립 재계산")
    led = pd.read_csv(LEDGER, dtype=str, encoding="utf-8-sig",
                      usecols=["rcDate", "meet", "rcNo", "hrNo", "ord"], low_memory=False)
    led["ord"] = pd.to_numeric(led["ord"], errors="coerce")
    led["rcDate"] = pd.to_numeric(led["rcDate"], errors="coerce")
    led["meet"] = led["meet"].astype(str).str.strip().map(MEET_CODE)
    led = led[led["ord"].between(1, ORD_MAX) & led["rcDate"].notna()]
    led = led.sort_values(["rcDate", "meet", "rcNo"])
    led["race_id"] = (led["rcDate"].astype(int).astype(str) + "_" +
                      led["meet"].astype("Int64").astype(str) + "_" +
                      led["rcNo"].astype(str).str.strip())
    led["cum"] = led.groupby("hrNo").cumcount()
    ref = led.set_index(led["race_id"] + "|" + led["hrNo"])["cum"]
    j = df.assign(_k=df["race_id"] + "|" + df["hrNo"].astype(str)).join(ref, on="_k", how="inner")
    r = (j["F1_starts_life"] == j["cum"]).mean()
    ck(r > 0.999, "F1_starts_life = 이전 출주수 (독립 재계산)", f"일치 {r*100:.3f}% / n={len(j):,}")
    first = j[j["cum"] == 0]
    ck(bool((first["F1_first_start"] == 1).all()), "첫 출전 플래그 정확", f"n={len(first):,}")

    print("\n5. 육종가 as-of")
    e = pd.read_csv("data/raw/ebv_long.csv", usecols=["asof"])
    snaps = np.array(sorted(e["asof"].unique()))
    rc = df["rcDate"].astype(int).to_numpy()
    idx = np.searchsorted(snaps, rc, side="left") - 1
    used = np.where(idx >= 0, snaps[np.clip(idx, 0, len(snaps) - 1)], np.nan)
    ok = (used < rc) | np.isnan(used)
    ck(ok.all(), "붙은 스냅샷이 항상 경주일 이전", f"위반 {int((~ok).sum())}건")
    hit = df["F2_ebv_prize"].notna()
    y = df["rcDate"].astype(int) // 10000
    print(f"       EBV 매칭 전체 {hit.mean()*100:.1f}%  "
          f"2017+ {hit[y >= 2017].mean()*100:.1f}%")

    print("\n6. 미래 상관 탐지")
    sus = []
    for c in S.features():
        if c not in df.columns or df[c].dtype.kind not in "if":
            continue
        v = df[c]
        if v.notna().sum() < 1000 or v.nunique() < 3:
            continue
        rr = np.corrcoef(v.fillna(v.median()), df["y_ord"])[0, 1]
        if abs(rr) > 0.55:
            sus.append((c, round(float(rr), 3)))
    ck(not sus, "이번 경주 착순과 과도한 상관(|r|>0.55) 없음", str(sus))

    print("\n7. game / sim 정합")
    card = pd.read_parquet(f"{S.GAME_DIR}/race_card.parquet")
    ent = pd.read_parquet(f"{S.GAME_DIR}/entries.parquet")
    pay = pd.read_csv(f"{S.GAME_DIR}/payouts.csv")
    sim = pd.read_parquet(f"{S.SIM_DIR}/passing.parquet")
    ck(set(card["race_id"]) == gm, "race_card 가 게임풀과 정확히 일치")
    ck(set(ent["race_id"]) == gm, "entries 가 게임풀과 정확히 일치")
    covered = pay["race_id"].nunique()
    wn(covered / len(gm) > 0.95, "배당 커버리지 95% 이상",
       f"{covered:,}/{len(gm):,} = {covered/len(gm)*100:.1f}%")
    pools = set(pay["pool"].unique())
    ck(pools >= set(S.POOLS), "7승식 전부 존재", f"누락 {set(S.POOLS)-pools}")
    print(f"       승식별 적중배당: "
          f"{ {p: int((pay['pool']==p).sum()) for p in S.POOLS} }")
    ck(set(sim["race_id"]) <= gm, "sim 이 게임풀 밖 경주를 안 담음")
    mid = (sim["point"] == "C2").sum()
    wn(mid / len(gm) > 5, "중간지점(C2) 충전 충분",
       f"{mid:,}행 — 코너 없는 단거리 경주엔 없음")

    print("\n8. 그룹별 충전율")
    for gname, blk in S.FEATURE_BLOCKS.items():
        cols = [c["name"] for c in blk if c["name"] in df.columns]
        if cols:
            print(f"  {gname:3s} {len(cols):>2}컬럼  {df[cols].notna().mean().mean()*100:5.1f}%")

    print("\n" + "=" * 62)
    if FAIL:
        print(f"실패 {len(FAIL)}: {FAIL}")
        sys.exit(1)
    print(f"전 항목 통과. 경고 {len(WARN)}건." + (f" {WARN}" if WARN else ""))


if __name__ == "__main__":
    main()
