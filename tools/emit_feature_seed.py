# -*- coding: utf-8 -*-
"""지표 마스터 → 백엔드가 그대로 넣을 수 있는 시드 SQL.

out/feature_master.json 을 읽어 feature_group(6) · feature(57) INSERT 를 만든다.
백엔드 ReferenceDataSeeder 는 "feature 테이블에 행이 하나라도 있으면 픽스처를
건너뛴다"(ReferenceDataSeeder.java:179). 이 SQL 을 한 번 넣으면 임시 픽스처 5개
대신 이 57개가 쓰인다.

## 구조 — 41행 (축 6 + 표시 지표 35)

축 점수를 **basemodel 6축 타워가 직접** 준다(tools/roundtrip_tower.py 실측 32.9%).
그러면 백엔드가 그룹 안 지표를 평균할 필요가 없고, 파생 컬럼(`_z`·`_rk`) 22개는
AI 서비스 내부 입력일 뿐이라 **DB 에 넣을 이유가 없다.**

    ax_*      6개   is_displayed=FALSE   슬라이더가 곱하는 축 점수. 계산의 전부
    표시 지표  35개   is_displayed=TRUE    트레이딩 카드 스탯 막대 + 출전표 원본값

`feature.is_displayed` 는 스키마에 이미 있는 칸이다 (schema.sql:103).

## 백엔드가 같이 고쳐야 하는 것

EntryFeatureScoreAxisProvider.fromFeatureScores 는 지금 **그룹 안 지표 점수를 산술
평균**한다. 그대로 두면 축 점수가 순위환산 평균본이 되어 top-1 이 32.9% → 28.3% 로
떨어진다. `ax_*` 여섯 행을 직접 읽어야 한다 — 그 클래스 주석이 예고한 변경이다
("축 점수 테이블 적재(AI-05)가 끝나면 이 클래스만 바꾸면 된다").

## 비워 두는 칸

    ref_profit_bp   단독 수익률. 확정배당이 있어야 재는데 valid 파케이에 winOdds 가 없다
    min/max_weight  항목 단위 슬라이더는 MP-03 으로 제외됐다. 0~100 으로 열어 둔다

    python tools/emit_feature_seed.py
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).resolve().parent.parent / "out"

GROUP_KO = {
    "CONDITION": ("컨디션", "요즘 기세가 좋은지"),
    "SPEED": ("스피드", "얼마나 빠른 말인지"),
    "RUNNING": ("주행", "달리는 방식이 이 경주에 맞는지"),
    "JOCKEY": ("기수", "기수가 잘 타는지"),
    "ENVIRONMENT": ("환경 적응력", "날씨·주로·거리에 잘 맞는지"),
    "ABILITY": ("공식 실력", "공식 등급과 통산 성적"),
}
MINW, MAXW = 0, 40          # 그룹 슬라이더 상·하한 (front fixtures 와 동일)


def q(s):
    return "'" + str(s).replace("'", "''") + "'"


def main():
    d = json.loads((OUT / "feature_master.json").read_text(encoding="utf-8"))
    ver = d["version"]
    shown = [f for f in d["features"] if f["role"] == "PRIMARY"]

    # 축 행 6개를 앞에 둔다 — 점수 계산이 읽는 유일한 행이다
    AXIS_KO = {"CONDITION": "컨디션 축", "SPEED": "스피드 축", "RUNNING": "주행 축",
               "JOCKEY": "기수 축", "ENVIRONMENT": "환경 적응력 축", "ABILITY": "공식 실력 축"}
    axis_rows = [dict(display_order=i, group_code=g, code="ax_" + g.lower(),
                      name_ko=AXIS_KO[g], role="AXIS", ref_win_rate_bp=None,
                      desc="6축 타워가 낸 %s 점수 — 경주 내 순위 백분위 0~100. "
                           "슬라이더가 곱하는 값이다" % AXIS_KO[g])
                 for i, g in enumerate(GROUP_KO, 1)]
    for f in shown:
        f["display_order"] += 6
    feats = axis_rows + shown

    L = [
        "-- 지표 마스터 시드 — AI(건모) 발행 %s" % ver,
        "--",
        "-- 팀 레포에 지표 목록 명세가 없어서(ReferenceDataSeeder.java:138 · Feature.java:16 ·",
        "-- GetRaceCardService.java:34 가 '55개'를 기다리지만 목록이 없다) AI 에서 기준을 낸다.",
        "--",
        "-- 근거",
        "--   축 배치    basemodel/config.py::AXIS_FEATURES",
        "--   설명·시점  docs/dataset/schema_v2.py 의 컬럼 desc · tier",
        "--   direction  train 에서 측정 (승리마의 경주 내 백분위 평균이 50 보다 큰가)",
        "--   ref_win_rate_bp  valid 1,254경주 실측 — 이 지표 하나로만 1등을 뽑았을 때 적중률",
        "--",
        "-- %d행 = 축 %d (is_displayed=FALSE, 계산용) + 표시 지표 %d (is_displayed=TRUE)."
        % (len(feats), len(feats) - len(shown), len(shown)),
        "-- 축 행 ax_* 여섯이 점수 계산의 전부다. 표시 지표는 화면 표시 전용이며",
        "--   EntryFeatureScoreAxisProvider 가 이들을 평균하면 안 된다 (32.9% -> 28.3%).",
        "-- ref_profit_bp 는 NULL — 단독 수익률은 확정배당이 필요한데 valid 파케이에 winOdds 가 없다.",
        "",
        "INSERT INTO feature_group (code, name_ko, name_easy, display_order, min_weight, max_weight)",
        "VALUES",
    ]
    L.append(",\n".join(
        "  (%s, %s, %s, %d, %g, %g)" % (q(c), q(ko), q(easy), i, MINW, MAXW)
        for i, (c, (ko, easy)) in enumerate(GROUP_KO.items(), 1)) + ";")

    L += [
        "",
        "INSERT INTO feature (feature_group_id, code, name_ko, name_easy, description,",
        "                     is_displayed, min_weight, max_weight,",
        "                     ref_win_rate_bp, ref_profit_bp, model_version, is_active, display_order)",
        "VALUES",
    ]
    L.append(",\n".join(
        "  ((SELECT feature_group_id FROM feature_group WHERE code = %s),\n"
        "   %s, %s, %s, %s,\n"
        "   %-5s, 0, 100, %s, NULL, %s, TRUE, %d)"
        % (q(f["group_code"]), q(f["code"]), q(f["name_ko"]), q(f["name_ko"]),
           q(f["desc"][:255]),
           "TRUE" if f["role"] == "PRIMARY" else "FALSE",
           f["ref_win_rate_bp"] if f["ref_win_rate_bp"] else "NULL",
           q(ver), f["display_order"])
        for f in feats) + ";")

    p = OUT / "feature_seed.sql"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")

    n_ins = sum(1 for x in L if x.lstrip().startswith("((SELECT")) \
        + "\n".join(L).count("),\n  ((SELECT")
    print("feature_group 6 · feature %d (축 %d + 표시 %d) → %s (%.1fKB)"
          % (len(feats), len(feats) - len(shown), len(shown), p.name,
             p.stat().st_size / 1024))
    body = (OUT / "feature_seed.sql").read_text(encoding="utf-8")
    assert body.count("(SELECT feature_group_id") == len(feats), \
        "INSERT 행 수가 지표 수와 다르다 — 생성이 깨졌다"
    print("검사 통과 — INSERT %d행" % body.count("(SELECT feature_group_id"))


if __name__ == "__main__":
    main()
