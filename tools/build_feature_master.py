# -*- coding: utf-8 -*-
"""지표 마스터(feature 테이블 시드)를 만든다 — 팀에 명세가 없어서 우리가 기준을 낸다.

## 왜 우리가 만드나

백엔드가 "deck 담당의 실제 지표(55개)" 를 세 군데서 기다리고 있는데
(ReferenceDataSeeder.java:138 · Feature.java:16 · GetRaceCardService.java:34)
**그 55개를 적어 놓은 문서가 레포 어디에도 없다.** API명세서 §11-1 도
`featureKeys: ["horse_place_rate", "..."]` 로 줄임표만 있다.

그래서 AI 쪽에서 기준을 만들어 백/프런트에 넘긴다. 근거는 셋 다 이미 있는 것이다:
    · 축 배치   basemodel/config.py::AXIS_FEATURES (6축 × 지표)
    · 설명 문구 docs/dataset/schema_v2.py 의 각 컬럼 desc
    · 가용 시점 schema_v2 의 tier (A=전일 · B=T-61분 · C=당일 · G=발주직전 · P=비활성)

## 측정해서 채우는 두 칸

    direction        높을수록 좋은가 — **train** 에서 정한다 (valid 에 적합시키지 않는다)
    ref_win_rate_bp  이 지표 하나로만 1등을 뽑았을 때 실제 적중률 (3800 = 38.0%)
                     schema.sql:107 의 정의이자 POST /ai/features/reference 의 산출물.
                     명세대로 valid 1,254경주에서 잰다

## 규약
    test 는 열지 않는다 · game 은 쓰지 않는다 · 정렬을 바꾸지 않는다

    python tools/build_feature_master.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_temperature import AXES, AXIS_FEATURES, pct_within_race, C, TEAM   # noqa: E402

sys.path.insert(0, str(TEAM / "docs" / "dataset"))
import schema_v2 as S                                                        # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"
TIER_KO = {"A": "전일 확정", "B": "발주 T-61분", "C": "당일 경마장별",
           "G": "발주 T-5분", "P": "비활성"}
# 실시간(73피처) 예측에 쓸 수 있는 tier — G(배당)는 발주 5분 전이라 못 쓴다
REALTIME_OK = {"A", "B", "C"}

# ─────────────────────────────────────────────────────────────────────────
# 역할 — 같은 정보를 세 벌로 담은 파생 컬럼을 유저 화면에서 뺀다
#
#   PRIMARY  화면에 이름·참고승률이 보이는 지표. feature 테이블에 넣는다
#   DERIVED  `_z`(경주 내 z-score) · `_rk`(경주 내 순위 백분위) — **모델 입력으로만** 쓴다
#
# 왜 빼나: win_rate_life / _z / _rk 는 같은 값을 세 방식으로 적은 것이라
# 단독 적중률이 28.9% 로 똑같이 나온다. 모델에는 서로 다른 인코딩이라 쓸모가 있지만
# 화면에 "통산 승률", "통산 승률 z", "통산 승률 순위" 가 나란히 뜨면 지표가 아니라 잡음이다.
# 축 점수(그룹 평균)에서도 같은 정보가 세 번 세어져 그 지표에만 3배 무게가 붙는다.
# ─────────────────────────────────────────────────────────────────────────
def role_of(internal: str) -> str:
    return "DERIVED" if internal.endswith(("_z", "_rk")) else "PRIMARY"


# ─────────────────────────────────────────────────────────────────────────
# 프런트가 이미 하드코딩한 지표 코드에 맞춘다 — 우리가 이름을 새로 지으면 화면이 깨진다
#
#   front/src/room/BettingBoard.jsx:11 STAT_AXES 가 트레이딩 카드 스탯 막대에
#   speed_index · early_pace · late_pace · jockey_win_rate · wet_track_rate 다섯을
#   문자열로 박아 두고 있다. API명세서 §11-1 예시도 같은 어휘를 쓴다.
#   뜻이 일치하는 것만 그 이름을 그대로 받는다.
#
#   받지 않는 둘 — 데이터에 **복승률이 없다**. 우리한테는 승률뿐이라
#   horse_place_rate / trainer_place_rate 로 부르면 이름이 거짓말이 된다.
#   이 둘은 레거시 픽스처(제거 예정)에만 남아 있어 화면을 깨지 않는다.
# ─────────────────────────────────────────────────────────────────────────
FE_ALIAS = {
    "F1_speed_avg3":      "speed_index",        # 최근 3경주 평균 스피드지수
    "F5_early_pos":       "early_pace",         # 과거 평균 초반 상대위치
    "F5_g3f_time":        "late_pace",          # 과거 평균 상3F — 막판 각력
    "F3_jk_win_rate_365": "jockey_win_rate",    # 기수 최근 1년 승률
    "F4_hr_wet_win_rate": "wet_track_rate",     # 이 말의 불량주로 승률
    "F1_ordpct_avg5":     "recent5_avg_rank",   # 최근 5경주 상대착순
}


def product_code(internal: str) -> str:
    """F1_win_rate_life → win_rate_life. 프런트가 쓰는 이름이 있으면 그걸 우선한다."""
    if internal in FE_ALIAS:
        return FE_ALIAS[internal]
    return internal.split("_", 1)[1] if internal[:1] in "FX" and "_" in internal else internal


def main():
    meta = {c['name']: c for c in S.ALL}
    tr, va = C.load("train"), C.load("valid")
    tr_win = tr["y_win"].to_numpy(bool)
    va_win = va["y_win"].to_numpy(float)
    rid = va["race_id"]

    # 코드 충돌 검사 — 같은 이름이 두 축에 생기면 featureKeys 가 망가진다
    seen, dup = {}, set()
    for ax in AXES:
        for f in AXIS_FEATURES[ax]:
            c = product_code(f)
            if c in seen:
                dup.add(c)
            seen[c] = f

    rows, order = [], 0
    for ax in AXES:
        for f in AXIS_FEATURES[ax]:
            if f not in tr.columns:
                continue
            m = meta.get(f)
            if not pd.api.types.is_numeric_dtype(tr[f]):
                continue                       # F5_style · F2_sire_id — 범주형은 축 점수에서 뺀다
            p_tr = pct_within_race(tr, f)
            if np.nanstd(p_tr) < 1e-6:
                continue
            d = 1 if p_tr[tr_win].mean() >= 50.0 else -1

            # 단독 적중률 — 이 지표만으로 경주별 1등을 뽑는다 (동점은 경주에서 제외)
            s = pct_within_race(va, f) * d
            g = pd.Series(s).groupby(rid.values)
            top = g.transform("max").to_numpy()
            is_top = s >= top - 1e-9
            n_top = pd.Series(is_top.astype(int)).groupby(rid.values).transform("sum").to_numpy()
            uniq = is_top & (n_top == 1)
            hit = va_win[uniq].mean() if uniq.sum() else np.nan

            order += 1
            # `F1_ordpct_avg5_z` 는 베이스 `F1_ordpct_avg5` 의 별칭을 물려받아
            # `recent5_avg_rank_z` 가 된다 — 같은 지표의 변형임이 이름에 드러나야 한다
            if f.endswith(("_z", "_rk")):
                stem, suf = f.rsplit("_", 1)
                code = product_code(stem) + "_" + suf
            else:
                code = product_code(f)
            if code in dup:
                code = f.lower()
            rows.append(dict(
                display_order=order, group_code=ax, code=code, source_column=f,
                name_ko=(m["desc"].split("—")[0].strip() if m else code),
                desc=(m["desc"] if m else ""), tier=(m["tier"] if m else "?"),
                tier_ko=TIER_KO.get(m["tier"] if m else "?", "?"),
                realtime=(m["tier"] in REALTIME_OK) if m else False,
                direction=d, role=role_of(f),
                ref_win_rate_bp=(int(round(hit * 10000)) if hit == hit else None),
                ref_races=int(uniq.sum())))

    OUT.mkdir(exist_ok=True)
    with open(OUT / "feature_master.csv", "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    (OUT / "feature_master.json").write_text(
        json.dumps({"version": "2026-09-14", "groups": AXES,
                    "note": "축 점수 = 그룹 안 지표 점수의 산술 평균 후 경주 내 순위 환산(0~100)",
                    "features": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    prim = [r for r in rows if r["role"] == "PRIMARY"]
    print("지표 %d개 — 화면용 PRIMARY %d개 + 모델 전용 DERIVED %d개"
          % (len(rows), len(prim), len(rows) - len(prim)))
    print("")
    print("%-13s%-24s%-22s%-12s%6s%10s%8s"
          % ("그룹", "code", "이름", "가용시점", "방향", "단독적중", "경주"))
    print("-" * 98)
    cur = None
    for r in prim:
        g = r["group_code"] if r["group_code"] != cur else ""
        cur = r["group_code"]
        print("%-13s%-24s%-22s%-12s%6s%9s%8d"
              % (g, r["code"], r["name_ko"][:20], r["tier_ko"],
                 "↑" if r["direction"] > 0 else "↓",
                 ("%.1f%%" % (r["ref_win_rate_bp"] / 100)) if r["ref_win_rate_bp"] else "-",
                 r["ref_races"]))
    print("-" * 98)
    n_rt = sum(1 for r in prim if r["realtime"])
    print("실시간 가용 %d / 화면용 %d개 (나머지는 리플레이 전용)" % (n_rt, len(prim)))
    bp = pd.Series([r["group_code"] for r in prim]).value_counts()
    ba = pd.Series([r["group_code"] for r in rows]).value_counts()
    print("축별  " + "  ".join("%s %d(+%d)" % (k, bp.get(k, 0), ba.get(k, 0) - bp.get(k, 0))
                              for k in AXES))
    weak = [r for r in prim if r["ref_win_rate_bp"] and r["ref_win_rate_bp"] < 1200]
    if weak:
        print("")
        print("단독 적중이 무작위(약 9.4%%) 수준인 지표 %d개 — 축 평균을 희석한다:" % len(weak))
        for r in weak:
            print("    %-12s %-22s %.1f%%"
                  % (r["group_code"], r["code"], r["ref_win_rate_bp"] / 100))
    print("")
    print("→ out/feature_master.csv · out/feature_master.json")


if __name__ == "__main__":
    main()
