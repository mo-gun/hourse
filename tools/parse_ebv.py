# -*- coding: utf-8 -*-
"""
육종가(EBV) 반기 스냅샷 20개 → 하나의 as-of 테이블.

왜 이름으로 매핑하면 안 되나
  20개 파일에 **컬럼 구성이 7종** 있고, 이름이 서로 어긋난다.
    · 2018·2019 파일은 상금육종가 계열에 '(초)' 가 붙어 있다 (중앙값 107·112 — 명백히 상금)
    · 2016-12 는 주파기록 계열이 상금 계열보다 **먼저** 나온다
    · pandas 가 동명 컬럼에 '.1' 을 붙이는데, 어느 계열의 정확도인지는 순서로만 알 수 있다
  이름만 믿으면 상금육종가와 주파기록육종가가 조용히 뒤바뀐다.

그래서 값 범위 + 컬럼 순서로 판별한다
    중앙값 0~1        → 정확도 (직전 계열에 귀속)
    중앙값 < 0        → 주파기록 계열 (초, 음수일수록 빠름)
    중앙값 1~20       → 근교계수 (%)
    중앙값 > 20       → 상금 계열 (평균100·표준편차20 표준화)

⚠ 누수 주의
  육종가는 **그 말 자신의 경주성적을 포함**해 계산된다("출주 시 개체의 경주성적이 합산").
  따라서 기준일 이후 경주에만 붙여야 한다. 이 스크립트는 asof 컬럼을 남기고,
  조인은 build 단계에서 '경주일 직전 스냅샷'으로 한다.

산출물
  data/raw/ebv_long.csv   (asof, sheet, 마명, 생년, ...) 롱 포맷

실행: python tools/parse_ebv.py
"""
import os, re, sys, glob
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

SRC = "data/raw/ebv/*.xlsx"
OUT = "data/raw/ebv_long.csv"
SHEETS = ["씨수말", "씨암말", "경주마", "육성마"]

ID_MAP = {"마명": "hrName", "생년": "birth_yr", "성별": "sex", "생산국": "prd_cty",
          "부마": "sire_nm", "모마": "dam_nm", "부계통": "sire_line",
          "외조부마": "damsire_nm", "경마공원": "meet_nm", "군": "grp"}


def _n(c):
    return re.sub(r"\s+", "", str(c))


def asof_of(path):
    m = re.search(r"\((\d{4})-(\d{2})-(\d{2})기준\)", os.path.basename(path))
    return int(m.group(1) + m.group(2) + m.group(3)) if m else None


def classify(name, med):
    """(계열, 역할) 판정. 계열은 prize/time/ssgblup/inbreed/acc."""
    n = _n(name)
    if "근교" in n:
        return "inbreed", "val"
    if "정확도" in n:
        fam = "ssgblup" if "SSGBLUP" in n else None
        return "acc", fam
    if med is None or med != med:
        return None, None
    if "SSGBLUP" in n:
        return "ssgblup", "val"
    if 0 <= med <= 1:
        return "acc", None
    fam = "time" if med < 0 else ("inbreed" if med <= 20 else "prize")
    role = "sprint" if "단거리" in n else ("route" if "중장거리" in n else "val")
    return fam, role


def parse_sheet(path, sheet):
    try:
        d = pd.read_excel(path, sheet_name=sheet, header=0)
    except Exception:
        return None
    if "마명" not in [_n(c) for c in d.columns]:
        return None
    d.columns = [_n(c) for c in d.columns]

    out = pd.DataFrame(index=d.index)
    for ko, en in ID_MAP.items():
        if ko in d.columns:
            out[en] = d[ko]

    cur_fam = None
    for c in d.columns:
        if _n(c) in ID_MAP or not any(k in _n(c) for k in ("육종가", "정확도", "계수")):
            continue
        v = pd.to_numeric(d[c], errors="coerce")
        if v.notna().sum() == 0:
            continue
        med = float(v.median())
        fam, role = classify(c, med)
        if fam is None:
            continue
        if fam == "acc":
            # 정확도는 직전에 나온 계열에 귀속. 이름에 SSGBLUP 이 있으면 그쪽.
            tgt = role or cur_fam
            if tgt:
                out[f"ebv_{tgt}_acc"] = v
            continue
        if fam == "inbreed":
            out["inbreeding"] = v
            continue
        cur_fam = fam
        col = f"ebv_{fam}" + ("" if role == "val" else f"_{role}")
        out[col] = v

    out.insert(0, "sheet", sheet)
    out.insert(0, "asof", asof_of(path))
    return out[out["hrName"].notna()]


def main():
    files = sorted(glob.glob(SRC), key=asof_of)
    if not files:
        print(f"{SRC} 에 파일이 없다."); return
    frames, report = [], []
    for f in files:
        got = []
        for sh in SHEETS:
            r = parse_sheet(f, sh)
            if r is not None and len(r):
                frames.append(r)
                got.append(f"{sh}{len(r):,}")
        report.append((asof_of(f), ", ".join(got)))

    df = pd.concat(frames, ignore_index=True)
    df["hrName"] = df["hrName"].astype(str).str.strip()
    df["birth_yr"] = pd.to_numeric(df.get("birth_yr"), errors="coerce").astype("Int64")
    df = df.drop_duplicates(["asof", "sheet", "hrName", "birth_yr"])
    df = df.sort_values(["asof", "sheet", "hrName"])
    df.to_csv(OUT, index=False, encoding="utf-8-sig", float_format="%.4f")

    print(f"저장 {OUT}: {len(df):,}행 × {len(df.columns)}열\n")
    print(f"{'기준일':<10} 시트별 행수")
    for a, g in report:
        print(f"  {a:<10} {g}")
    print(f"\n스냅샷 {df['asof'].nunique()}개  {df['asof'].min()} ~ {df['asof'].max()}")
    print(f"고유 마명 {df['hrName'].nunique():,}두\n")
    print("컬럼별 충전율 / 값 범위 — 계열이 섞이지 않았는지 검산")
    for c in sorted(df.columns):
        if not c.startswith(("ebv_", "inbreeding")):
            continue
        v = pd.to_numeric(df[c], errors="coerce").dropna()
        if not len(v):
            print(f"  {c:22s} (비어있음)"); continue
        print(f"  {c:22s} {v.notna().sum():>7,}행  중앙 {v.median():8.2f}  "
              f"[{v.min():8.2f}, {v.max():8.2f}]")


if __name__ == "__main__":
    main()
