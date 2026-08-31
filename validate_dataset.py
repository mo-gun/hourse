# -*- coding: utf-8 -*-
"""
3단계 — 데이터셋 검증. 6명이 이 데이터를 믿으려면 통과해야 하는 관문.

가장 중요한 검사는 **누수 재현 검증**이다.
빌더와 무관하게, 원장에서 직접 "그 경주 이전 행만"으로 값을 다시 계산해서
데이터셋의 값과 일치하는지 대조한다. 빌더 로직을 그대로 베끼면 검증이 아니므로
여기서는 pandas 로 독립 구현한다.

실행
  python validate_dataset.py
"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import numpy as np

import schema as S
from build_dataset import LEDGER, OUT_CSV, MEET_CODE, ORD_MAX

FAIL = []
WARN = []


def check(ok, label, detail=""):
    mark = "OK  " if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAIL.append(label)


def warn(ok, label, detail=""):
    if not ok:
        print(f"  [WARN] {label}" + (f"  — {detail}" if detail else ""))
        WARN.append(label)
    else:
        print(f"  [OK  ] {label}" + (f"  — {detail}" if detail else ""))


def main():
    print("데이터셋 로드...")
    df = pd.read_csv(OUT_CSV, encoding="utf-8-sig", low_memory=False)
    print(f"  {len(df):,}행 × {len(df.columns)}열\n")

    # ── 1. 스키마 정합 ────────────────────────────────────────────────
    print("1. 스키마 정합")
    expected = ([c["name"] for c in S.INDEX] + [c["name"] for c in S.TARGETS] +
                S.features() + [c["name"] for c in S.F6X])
    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]
    check(not missing, "스키마 컬럼 전부 존재", f"누락 {len(missing)}: {missing[:5]}")
    check(not extra, "스키마 외 컬럼 없음", f"초과 {len(extra)}: {extra[:5]}")
    bad = [c for c in df.columns if c in S.FORBIDDEN_SOURCE_FIELDS]
    check(not bad, "누수 금지 원본필드 미포함", str(bad))

    # ── 2. 키 무결성 ─────────────────────────────────────────────────
    print("\n2. 키 무결성")
    check(df["row_id"].is_unique, "row_id 고유")
    check(df["race_id"].notna().all(), "race_id 결측 없음")
    g = df.groupby("race_id")
    check((g["y_ord"].min() == 1).all(), "모든 경주에 1착 존재")
    dup = g["y_ord"].apply(lambda s: s.duplicated().sum()).sum()
    warn(dup == 0, "경주 내 착순 중복 없음", f"동착 포함 {dup}건")
    check((df["X_dusu"] == g["row_id"].transform("size")).all(),
          "X_dusu == 경주별 행 수")

    # ── 3. 분할 무결성 ───────────────────────────────────────────────
    print("\n3. 분할 무결성")
    spans = g["split"].nunique()
    check((spans == 1).all(), "한 경주가 두 split 에 걸치지 않음",
          f"위반 {int((spans>1).sum())}경주")
    for name, (a, b) in S.SPLIT.items():
        sub = df[df["split"] == name]
        if len(sub):
            check(sub["rcDate"].between(a, b).all(), f"{name} 기간 준수",
                  f"{int(sub['rcDate'].min())}~{int(sub['rcDate'].max())}")
    order_ok = df.groupby("split")["rcDate"].max().reindex(
        ["train", "valid", "test"]).dropna().is_monotonic_increasing
    check(order_ok, "train < valid < test 시간 순서")

    # ── 4. 누수 재현 검증 (핵심) ──────────────────────────────────────
    print("\n4. 누수 재현 검증 — 원장에서 독립 재계산")
    led = pd.read_csv(LEDGER, dtype=str, encoding="utf-8-sig", low_memory=False)
    led["meet"] = led["meet"].astype(str).str.strip().map(MEET_CODE)
    led["rcDate"] = pd.to_numeric(led["rcDate"], errors="coerce")
    led["ord"] = pd.to_numeric(led["ord"], errors="coerce")
    led["rcNo"] = pd.to_numeric(led["rcNo"], errors="coerce")
    # 행 포함 규칙은 빌더와 동일해야 한다. 여기서 센티널(91~99)을 남기면
    # 누적 카운트가 어긋나서, 누수가 없는데도 불일치로 보인다.
    led = led[led["ord"].between(1, ORD_MAX) & led["rcDate"].notna()]
    led["race_id"] = (led["rcDate"].astype(int).astype(str) + "_" +
                      led["meet"].astype("Int64").astype(str) + "_" +
                      led["rcNo"].astype("Int64").astype(str))
    led["win"] = (led["ord"] == 1).astype(int)
    led = led.sort_values(["rcDate", "meet", "rcNo"])

    # 말 기준: 통산 출주수/승수는 '그 경주 이전' 행만
    led["cum_starts"] = led.groupby("hrNo").cumcount()
    led["cum_wins"] = led.groupby("hrNo")["win"].cumsum() - led["win"]
    ref = led.set_index(led["race_id"] + "|" + led["hrNo"])[["cum_starts", "cum_wins"]]

    d = df.copy()
    d["_k"] = d["race_id"] + "|" + d["hrNo"].astype(str)
    j = d.join(ref, on="_k", how="inner")
    print(f"  대조 가능 {len(j):,}행")

    s_ok = (j["F1_starts_life"] == j["cum_starts"]).mean()
    check(s_ok > 0.999, "F1_starts_life = 이전 출주수 (독립 재계산 일치)",
          f"일치율 {s_ok*100:.3f}%")

    exp_wr = np.where(j["cum_starts"] > 0, j["cum_wins"] / j["cum_starts"].replace(0, np.nan), np.nan)
    got = j["F1_win_rate_life"].to_numpy()
    both = ~(np.isnan(exp_wr) | np.isnan(got))
    wr_ok = np.isclose(exp_wr[both], got[both], atol=1e-9).mean() if both.sum() else 0
    check(wr_ok > 0.999, "F1_win_rate_life = 이전 승률 (독립 재계산 일치)",
          f"일치율 {wr_ok*100:.3f}% / n={both.sum():,}")

    first = j[j["cum_starts"] == 0]
    check(bool((first["F1_starts_life"] == 0).all()), "첫 출전 말의 통산 출주수 = 0",
          f"n={len(first):,}")
    check(bool(first["F1_win_rate_life"].isna().all()), "첫 출전 말의 통산 승률 = 결측")
    check(bool((first["F1_first_start"] == 1).all()), "첫 출전 플래그 정확")

    # 미래 상관 검사 — as-of 피처가 이번 경주 결과를 알고 있으면 이상하게 높은 상관이 나온다
    print("\n5. 미래 정보 유입 탐지 (as-of 피처 vs 이번 경주 착순)")
    suspicious = []
    for c in S.features():
        if c not in df.columns or df[c].dtype.kind not in "if":
            continue
        v = df[c]
        if v.notna().sum() < 1000 or v.nunique() < 3:
            continue
        r = np.corrcoef(v.fillna(v.median()), df["y_ord"])[0, 1]
        if abs(r) > 0.55:
            suspicious.append((c, round(float(r), 3)))
    check(not suspicious, "이번 경주 착순과 과도한 상관(|r|>0.55) 없음", str(suspicious))

    # ── 6. 시장 베이스라인 ───────────────────────────────────────────
    print("\n6. 시장 베이스라인 (모델이 넘어야 할 기준)")
    for name in ("train", "valid", "test"):
        sub = df[(df["split"] == name) & df["F6X_mkt_rank"].notna()]
        if not len(sub):
            continue
        fav = sub[sub["F6X_mkt_rank"] == 1]
        if not len(fav):
            continue
        top1 = fav["y_win"].mean()
        top3 = fav["y_plc"].mean()
        print(f"  {name:6s} 인기1위 승률 {top1*100:5.1f}%  복승률 {top3*100:5.1f}%  "
              f"(n={len(fav):,}경주)")
    ov = df["F6X_overround"].dropna()
    if len(ov):
        print(f"  평균 공제율(overround) {ov.mean()*100:.1f}%  "
              f"— 한국 경마 단승 공제율과 대조해볼 것")

    # ── 7. 충전율 ────────────────────────────────────────────────────
    print("\n7. 그룹별 충전율")
    for gname in ("X", "F1", "F2", "F3", "F4", "F5", "F6"):
        cols = [c["name"] for c in S.FEATURE_BLOCKS[gname] if c["name"] in df.columns]
        if not cols:
            continue
        fill = df[cols].notna().mean().mean()
        empty = [c for c in cols if df[c].notna().sum() == 0]
        print(f"  {gname:3s} {len(cols):>2}컬럼  평균 충전율 {fill*100:5.1f}%"
              + (f"   빈 컬럼 {len(empty)}개" if empty else ""))

    print("\n" + "=" * 62)
    if FAIL:
        print(f"실패 {len(FAIL)}건: {FAIL}")
        sys.exit(1)
    print(f"전 항목 통과. 경고 {len(WARN)}건.")


if __name__ == "__main__":
    main()
