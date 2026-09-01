# -*- coding: utf-8 -*-
"""
경마 예측 학습 데이터셋 — 컬럼 스키마 단일 원천 (v1).

팀원 6명이 **동일한 학습/검증/테스트 데이터**를 쓰기 위한 계약(contract).
빌더도, 6개 모델도 전부 이 파일을 import 해서 컬럼을 고른다.
컬럼을 추가/변경하려면 이 파일을 고치고 SCHEMA_VERSION 을 올린 뒤 데이터셋을 재생성한다.

설계 근거
  - Bolton & Chapman (1986), Benter (1994): 경주는 말 단위가 아니라 **경주 단위 그룹**으로
    모델링한다(conditional logit). → race_id 가 그룹 키이고, 경주는 절대 분할되지 않는다.
  - Benter 2단계: 펀더멘털 모델 → 공개배당률과 결합. → F6(인기도) 축이 여기에 해당한다.
  - Pace figure = (Par − Actual) + Variant: 구간기록을 (경마장×거리×주로) 기준기록으로
    정규화한다. 서울/부산/제주가 서로 다른 구간 컬럼을 쓰는 문제를 이걸로 흡수한다.

사용
    from schema import features, INDEX, TARGETS, GROUPS

    cols = features(groups=["X", "F3"])          # 공통 + 기수 실력
    cols = features(tier="A")                    # 전날(D-1) 예측에 쓸 수 있는 것만
    cols = features(exclude_leaky=True)          # 기본값 — 누수 컬럼은 애초에 안 나옴
"""

SCHEMA_VERSION = "1.1.0"
DATASET_FILE = "dataset/v1/kra_ml_v1.parquet"

# ─────────────────────────────────────────────────────────────────────────
# 그룹 — 팀원 1인이 하나씩 맡아 가중치를 높인다.
# 컬럼 접두사가 곧 그룹이라, 가중치 적용이 접두사 필터 한 줄로 끝난다.
# ─────────────────────────────────────────────────────────────────────────
GROUPS = {
    "X":  "공통 기본 — 전원 항상 포함. 특정 팀원 소유가 아님",
    "F1": "최근 성적 — 폼, 컨디션, 조교, 장제",
    "F2": "부모 성적 — 부마/모마/외조부 혈통 및 자마 성적",
    "F3": "기수 실력 — 기수, 조교사, 조합 궁합",
    "F4": "거리·날씨 — 거리 적성, 주로상태, 기상",
    "F5": "주행 스타일 — 각질, 구간 페이스, 경주 전개 구도",
    "F6": "인기도 — 시장 예측 확률 (Benter 2단계)",
}

# ─────────────────────────────────────────────────────────────────────────
# 가용 시점 티어 — 2026-08-28 발주 시간대 실측 결과 (probe_realtime.py)
#   A : 경주 전날(D-1)에 이미 확정. 전날 예측 모델이 쓸 수 있다.
#   B : 발주 60~90분 전 공개. 마체중이 여기.
#   C : 발주 10분 전 공개. 날씨/주로상태가 여기.
# 티어가 섞인 모델은 그 모델의 최소 가용 시점이 곧 예측 가능 시점이 된다.
# ─────────────────────────────────────────────────────────────────────────
TIERS = {
    "A": "D-1 확정 — 전날 예측 가능",
    "B": "발주 T-60~90분 (마체중)",
    "C": "발주 T-10분 (날씨·주로상태)",
    "P": "경주 후에만 — 학습 입력 금지. 타깃/평가/베이스라인 전용",
}


def _c(name, group, tier, dtype, desc, source=""):
    return {"name": name, "group": group, "tier": tier,
            "dtype": dtype, "desc": desc, "source": source}


# ─────────────────────────────────────────────────────────────────────────
# 인덱스 — 식별자. 절대 피처로 쓰지 않는다.
# ─────────────────────────────────────────────────────────────────────────
INDEX = [
    _c("race_id", "IDX", "A", "str",
       "경주 고유키 '{rcDate}_{meet}_{rcNo}'. **랭킹 모델의 group 키**", "합성"),
    _c("row_id", "IDX", "A", "str", "'{race_id}_{chulNo}' — 행 고유키", "합성"),
    _c("rcDate", "IDX", "int32", "int32", "경주일 YYYYMMDD. 시간 분할 기준", "raceResult_3"),
    _c("meet", "IDX", "A", "int8", "1서울 2제주 3부산경남", "raceResult_3"),
    _c("rcNo", "IDX", "A", "int8", "경주번호", "raceResult_3"),
    _c("hrNo", "IDX", "A", "str", "마번 — 말 고유 ID", "raceResult_3"),
    _c("hrName", "IDX", "A", "str", "마명 (사람이 읽기용)", "raceResult_3"),
    _c("jkNo", "IDX", "A", "str", "기수번호", "raceResult_3"),
    _c("trNo", "IDX", "A", "str", "조교사번호", "raceResult_3"),
    _c("split", "IDX", "A", "category",
       "train / valid / test — **6명 전원 이 컬럼을 그대로 쓴다**", "합성"),
]

# ─────────────────────────────────────────────────────────────────────────
# 타깃
# ─────────────────────────────────────────────────────────────────────────
TARGETS = [
    _c("y_ord", "Y", "P", "int8", "착순 (1=우승)", "raceResult_3.ord"),
    _c("y_win", "Y", "P", "int8", "1착 여부 0/1 — 이진 분류 타깃", "파생"),
    _c("y_plc", "Y", "P", "int8", "3착 이내 0/1 (출주 7두 미만이면 2착 이내)", "파생"),
    _c("y_rel", "Y", "P", "int8",
       "랭킹 관련도 = max(0, dusu - ord). LightGBM lambdarank 라벨", "파생"),
    _c("y_rcTime", "Y", "P", "float32", "주파기록 초", "raceResult_3.rcTime"),
    _c("y_speed_fig", "Y", "P", "float32",
       "스피드 지수 = (기준기록 - 실주파) 표준화. 회귀 타깃 대안", "파생"),
]

# ─────────────────────────────────────────────────────────────────────────
# X — 공통 기본
# ─────────────────────────────────────────────────────────────────────────
X = [
    _c("X_age", "X", "A", "int8", "연령", "raceResult_3.age"),
    _c("X_sex", "X", "A", "category", "성별 수/암/거", "raceResult_3.sex"),
    _c("X_prd_cty", "X", "A", "category", "산지 한국/미국/일본/호주 등", "raceResult_3.prd"),
    _c("X_rcDist", "X", "A", "int16", "경주거리 m", "raceResult_3.rcDist"),
    _c("X_grade", "X", "A", "category", "경주등급 (국1~6, 오픈 등)", "raceResult_3.rank"),
    _c("X_chulNo", "X", "A", "int8", "게이트 번호 (= 출주번호)", "raceResult_3.chulNo"),
    _c("X_dusu", "X", "A", "int8", "출주두수 — 경주 난이도 정규화에 필수", "entrySheet_2.dusu"),
    _c("X_gate_rel", "X", "A", "float32", "chulNo / dusu — 안쪽/바깥 상대위치", "파생"),
    _c("X_wgBudam", "X", "A", "float32", "부담중량 kg", "raceResult_3.wgBudam"),
    _c("X_ilsu", "X", "A", "int16", "휴양일수 (직전 출전 이후 경과일)", "raceResult_3.ilsu"),
    _c("X_rating", "X", "A", "float32",
       "경주 시점 레이팅. 충전율 ~72% (2000~2026 안정) — 결측 그대로 두고 모델이 처리", "raceResult_3.rating"),
    _c("X_prize_cond", "X", "A", "float32", "1착 상금 — 경주 격 대리변수", "raceResult_3.chaksun1"),
    _c("X_wgHr", "X", "B", "float32", "마체중 kg. **발주 T-60~90분 공개**", "raceResult_3.wgHr"),
    _c("X_wgHr_delta", "X", "B", "float32",
       "전회 대비 마체중 증감. '465(+9)' 문자열의 괄호부. 컨디션 지표", "raceResult_3.wgHr 파싱"),
]

# ─────────────────────────────────────────────────────────────────────────
# F1 — 최근 성적 (폼·컨디션). 전부 rcDate 이전 경주만으로 롤링 집계.
# ─────────────────────────────────────────────────────────────────────────
F1 = [
    _c("F1_starts_life", "F1", "A", "int16", "통산 출주수 (as-of)", "원장 롤링"),
    _c("F1_win_rate_life", "F1", "A", "float32", "통산 승률 (as-of)", "원장 롤링"),
    _c("F1_plc_rate_life", "F1", "A", "float32", "통산 복승률 (3착 이내)", "원장 롤링"),
    _c("F1_starts_365", "F1", "A", "int16", "최근 365일 출주수", "원장 롤링"),
    _c("F1_win_rate_365", "F1", "A", "float32", "최근 365일 승률", "원장 롤링"),
    _c("F1_ord_last1", "F1", "A", "float32", "직전 경주 착순", "원장 롤링"),
    _c("F1_ord_last2", "F1", "A", "float32", "2경주 전 착순", "원장 롤링"),
    _c("F1_ord_last3", "F1", "A", "float32", "3경주 전 착순", "원장 롤링"),
    _c("F1_ord_avg3", "F1", "A", "float32", "최근 3경주 평균 착순", "원장 롤링"),
    _c("F1_ordpct_avg5", "F1", "A", "float32",
       "최근 5경주 평균 상대착순 (ord/dusu) — 두수 보정판", "원장 롤링"),
    _c("F1_speed_avg3", "F1", "A", "float32",
       "최근 3경주 평균 스피드지수 (거리·주로 정규화)", "원장 롤링"),
    _c("F1_speed_best365", "F1", "A", "float32", "최근 365일 최고 스피드지수", "원장 롤링"),
    _c("F1_margin_avg3", "F1", "A", "float32", "최근 3경주 평균 착차(마신)", "원장 롤링"),
    _c("F1_prize_365", "F1", "A", "float32", "최근 365일 획득 상금", "원장 롤링"),
    _c("F1_grade_last", "F1", "A", "category", "직전 경주 등급", "원장 롤링"),
    _c("F1_grade_move", "F1", "A", "int8", "등급 변동 +승급 / 0유지 / -강급", "파생"),
    _c("F1_layoff_days", "F1", "A", "int16", "직전 출전 경과일 (X_ilsu 검산용)", "원장 롤링"),
    _c("F1_first_start", "F1", "A", "int8", "신마 여부 (통산 출주 0) 0/1", "파생"),
    # 조교 — 신규 API trcontihi/gettrcontihi (2010~ 소급, 연 35~46만건)
    _c("F1_tr_sessions_28d", "F1", "A", "int16", "최근 28일 조교 횟수", "말훈련내역"),
    _c("F1_tr_minutes_28d", "F1", "A", "float32", "최근 28일 조교 총 시간(분)", "말훈련내역"),
    _c("F1_tr_days_since", "F1", "A", "int16", "최종 조교일 경과일", "말훈련내역"),
    _c("F1_tr_swim_28d", "F1", "A", "int16", "최근 28일 수영훈련 횟수", "말훈련내역.swimTr"),
    _c("F1_tr_intensity_chg", "F1", "A", "float32",
       "직전 28일 조교량 / 그 이전 28일 조교량 — 조교 강도 변화", "말훈련내역"),
    # 장제 — 신규 API API191_1/HorseShoe_1 (2010~ 소급, 연 ~3만건)
    _c("F1_shoe_days_since", "F1", "A", "int16", "최종 장제 경과일", "경주마장제정보"),
    _c("F1_shoe_type_chg", "F1", "A", "int8", "직전 대비 장제형태 변경 0/1", "경주마장제정보"),
    _c("F1_shoe_type", "F1", "A", "category", "현재 장제형태", "경주마장제정보.codeName2"),
]

# ─────────────────────────────────────────────────────────────────────────
# F2 — 부모 성적 (혈통)
#  ⚠ 마필종합 API 의 rcCnt/winRate/avgWinDist/rankTop 은 '오늘 기준 스냅샷'이라 누수다.
#     혈통 **식별자만** 가져오고, 자마 성적은 원장에서 as-of 집계한다.
# ─────────────────────────────────────────────────────────────────────────
F2 = [
    _c("F2_sire_id", "F2", "A", "category", "부마 마번", "마필종합.fhrNo"),
    _c("F2_dam_id", "F2", "A", "category", "모마 마번", "마필종합.mhrNo"),
    _c("F2_damsire_id", "F2", "A", "category", "외조부 마번", "마필종합.mhrFhrNo"),
    _c("F2_sire_cty", "F2", "A", "category", "부마 국적", "마필종합.fhrCty"),
    _c("F2_damsire_cty", "F2", "A", "category", "외조부 국적", "마필종합.mhrFhrCty"),
    _c("F2_sire_starts", "F2", "A", "int32", "부마 자마들의 누적 출주수 (as-of)", "원장 롤링"),
    _c("F2_sire_win_rate", "F2", "A", "float32", "부마 자마 승률 (as-of)", "원장 롤링"),
    _c("F2_sire_speed_avg", "F2", "A", "float32", "부마 자마 평균 스피드지수 (as-of)", "원장 롤링"),
    _c("F2_sire_dist_fit", "F2", "A", "float32",
       "부마 자마의 **이번 거리대** 승률 (as-of) — 혈통 거리적성", "원장 롤링"),
    _c("F2_sire_wet_fit", "F2", "A", "float32",
       "부마 자마의 **불량주로** 승률 (as-of) — 혈통 주로적성", "원장 롤링"),
    _c("F2_sire_avg_win_dist", "F2", "A", "float32",
       "부마 자마 우승 평균거리 (as-of 직접 집계. API avgWinDist 는 누수라 미사용)", "원장 롤링"),
    _c("F2_damsire_win_rate", "F2", "A", "float32", "외조부 자마 승률 (as-of)", "원장 롤링"),
    _c("F2_dam_progeny_starts", "F2", "A", "int16", "모마의 자마 누적 출주수 (as-of)", "원장 롤링"),
    _c("F2_dam_progeny_win_rate", "F2", "A", "float32", "모마 자마 승률 (as-of)", "원장 롤링"),
    _c("F2_sib_win_rate", "F2", "A", "float32",
       "전형제(동부동모) 승률 (as-of) — 표본 적으면 결측", "원장 롤링"),
    _c("F2_sire_prog_n", "F2", "A", "int16",
       "부마 자마 두수 — F2_sire_* 신뢰도 가중치로 쓸 것", "원장 롤링"),
]

# ─────────────────────────────────────────────────────────────────────────
# F3 — 기수 실력 (사람 요인: 기수 + 조교사 + 조합)
# ─────────────────────────────────────────────────────────────────────────
F3 = [
    _c("F3_jk_starts_life", "F3", "A", "int32", "기수 통산 출주수 (as-of)", "원장 롤링"),
    _c("F3_jk_win_rate_life", "F3", "A", "float32", "기수 통산 승률 (as-of)", "원장 롤링"),
    _c("F3_jk_plc_rate_life", "F3", "A", "float32", "기수 통산 복승률 (as-of)", "원장 롤링"),
    _c("F3_jk_starts_365", "F3", "A", "int16", "기수 최근 365일 출주수", "원장 롤링"),
    _c("F3_jk_win_rate_365", "F3", "A", "float32",
       "기수 최근 365일 승률 — 실측 2.2%~23.2%로 10배 차이. 강한 변수", "원장 롤링"),
    _c("F3_jk_win_rate_meet", "F3", "A", "float32", "기수의 이번 경마장 승률 (as-of)", "원장 롤링"),
    _c("F3_jk_win_rate_dist", "F3", "A", "float32", "기수의 이번 거리대 승률 (as-of)", "원장 롤링"),
    _c("F3_jk_ordpct_avg", "F3", "A", "float32", "기수 평균 상대착순 (as-of)", "원장 롤링"),
    _c("F3_tr_starts_365", "F3", "A", "int16", "조교사 최근 365일 출주수", "원장 롤링"),
    _c("F3_tr_win_rate_365", "F3", "A", "float32", "조교사 최근 365일 승률", "원장 롤링"),
    _c("F3_tr_win_rate_life", "F3", "A", "float32", "조교사 통산 승률 (as-of)", "원장 롤링"),
    _c("F3_jkhr_starts", "F3", "A", "int16", "이 기수-말 조합 누적 출주수 (as-of)", "원장 롤링"),
    _c("F3_jkhr_win_rate", "F3", "A", "float32", "이 기수-말 조합 승률 (as-of)", "원장 롤링"),
    _c("F3_jkhr_first", "F3", "A", "int8", "이 말에 이 기수가 처음 타는가 0/1", "파생"),
    _c("F3_jktr_win_rate", "F3", "A", "float32", "기수-조교사 조합 승률 (as-of)", "원장 롤링"),
]

# ─────────────────────────────────────────────────────────────────────────
# F4 — 거리 · 날씨 · 주로
# ─────────────────────────────────────────────────────────────────────────
F4 = [
    _c("F4_dist_band", "F4", "A", "category",
       "거리대 ~1200 / 1300-1400 / 1600-1700 / 1800-1900 / 2000+", "파생"),
    _c("F4_hr_dist_starts", "F4", "A", "int16", "이 말의 이번 거리대 출주수 (as-of)", "원장 롤링"),
    _c("F4_hr_dist_win_rate", "F4", "A", "float32", "이 말의 이번 거리대 승률 (as-of)", "원장 롤링"),
    _c("F4_hr_dist_speed", "F4", "A", "float32", "이 말의 이번 거리대 평균 스피드지수", "원장 롤링"),
    _c("F4_hr_best_dist", "F4", "A", "int16", "이 말의 최고 성적 거리 (as-of)", "원장 롤링"),
    _c("F4_dist_gap", "F4", "A", "int16", "|이번 거리 - 최적 거리|", "파생"),
    _c("F4_dist_change", "F4", "A", "int16", "직전 경주 대비 거리 변화 m", "원장 롤링"),
    _c("F4_hr_meet_starts", "F4", "A", "int16", "이 말의 이번 경마장 출주수 (as-of)", "원장 롤링"),
    _c("F4_hr_meet_win_rate", "F4", "A", "float32", "이 말의 이번 경마장 승률 (as-of)", "원장 롤링"),
    _c("F4_month", "F4", "A", "int8", "경주 월 — 계절성", "파생"),
    _c("F4_hr_wet_starts", "F4", "A", "int16", "이 말의 불량주로(함수율≥10%) 출주수 (as-of)", "원장 롤링"),
    _c("F4_hr_wet_win_rate", "F4", "A", "float32", "이 말의 불량주로 승률 (as-of)", "원장 롤링"),
    _c("F4_track_moist", "F4", "C", "float32",
       "주로 함수율 %. **발주 T-10분 공개** — 전날 예측엔 못 씀", "raceResult_3.track 파싱"),
    _c("F4_track_state", "F4", "C", "category", "주로상태 건조/양호/다습/포화/불량", "raceResult_3.track"),
    _c("F4_weather", "F4", "C", "category", "날씨 맑음/흐림/비/눈. **발주 T-10분 공개**", "raceResult_3.weather"),
]

# ─────────────────────────────────────────────────────────────────────────
# F5 — 주행 스타일 (각질) + 경주 전개 구도
#  구간기록 컬럼이 경마장마다 완전히 다르다(서울 se_*/sj_*, 부산 bu_* 22개, 제주 je_* 7개).
#  → 정규화 위치 4점(S1F / 중반 / G3F / G1F)으로 매핑한 뒤 상대위치(0~1)로 환산해서 쓴다.
#  ⚠ 이번 경주의 구간기록은 결과다. 반드시 **과거 경주** 구간기록만 집계한다.
# ─────────────────────────────────────────────────────────────────────────
F5 = [
    _c("F5_style", "F5", "A", "category",
       "각질 4분류 선행/선입/중단/추입 — 과거 경주 초반 상대위치 중앙값 기준", "구간기록 롤링"),
    _c("F5_early_pos", "F5", "A", "float32",
       "과거 평균 초반 상대위치 (S1F 순위/두수, 0=선두)", "구간기록 롤링"),
    _c("F5_mid_pos", "F5", "A", "float32", "과거 평균 중반 상대위치", "구간기록 롤링"),
    _c("F5_late_pos", "F5", "A", "float32", "과거 평균 G3F 지점 상대위치", "구간기록 롤링"),
    _c("F5_pos_gain", "F5", "A", "float32",
       "초반→결승 평균 순위 상승폭. 양수=추입형", "구간기록 롤링"),
    _c("F5_g3f_time", "F5", "A", "float32",
       "과거 평균 상3F 기록 (거리·주로 정규화) — 막판 각력", "구간기록 롤링"),
    _c("F5_g3f_best", "F5", "A", "float32", "과거 최고 상3F", "구간기록 롤링"),
    _c("F5_s1f_time", "F5", "A", "float32", "과거 평균 초반 1F — 게이트 스피드", "구간기록 롤링"),
    _c("F5_style_consistency", "F5", "A", "float32",
       "초반 상대위치 표준편차 — 낮을수록 각질이 일정", "구간기록 롤링"),
    _c("F5_race_n_front", "F5", "A", "int8",
       "**경주 레벨** 이 경주 선행마 두수. 다수면 페이스 붕괴 → 추입 유리", "경주 단위 집계"),
    _c("F5_race_pace_press", "F5", "A", "float32",
       "**경주 레벨** 페이스 압력 = 출전마 F5_early_pos 의 역수합 정규화", "경주 단위 집계"),
    _c("F5_style_fit", "F5", "A", "float32",
       "내 각질이 이 경주 구도에서 유리한 정도 (style × race_pace_press 상호작용)", "파생"),
    _c("F5_gate_style_fit", "F5", "A", "float32",
       "게이트 × 각질 적합도 (선행마 바깥 게이트는 불리)", "파생"),
    _c("F5_sect_n", "F5", "A", "int8",
       "구간기록이 있는 과거 경주 수 — F5_* 신뢰도 가중치", "구간기록 롤링"),
]

# ─────────────────────────────────────────────────────────────────────────
# F6 — 인기도 (Benter 2단계)
#  ⚠ 공공 API 는 **확정배당률만** 준다. 발주 4분 전에도 0행임을 실측 확인했다.
#     따라서 실제 배당률은 학습 입력이 될 수 없다.
#  구조:  F6X_* (경주 후 실측 시장값) 을 **타깃**으로 하는 시장예측 서브모델을 학습 →
#         그 예측값 F6_mkt_prob_pred 를 본 모델의 피처로 쓴다. 누수 없음.
# ─────────────────────────────────────────────────────────────────────────
F6 = [
    _c("F6_mkt_prob_pred", "F6", "A", "float32",
       "시장예측 서브모델이 낸 예상 인기(내재확률). **경주 전 산출 가능**", "서브모델"),
    _c("F6_mkt_rank_pred", "F6", "A", "int8", "예상 인기순위 (경주 내 순위)", "서브모델"),
    _c("F6_mkt_prob_pred_z", "F6", "A", "float32", "경주 내 z-score", "파생"),
    _c("F6_value_gap", "F6", "A", "float32",
       "펀더멘털 확률 - 예상 시장확률. 양수 = 저평가(밸류). Benter 2단계의 핵심", "파생"),
    _c("F6_field_entropy", "F6", "A", "float32",
       "예상 시장확률 분포의 엔트로피 — 혼전(높음) vs 독주(낮음)", "경주 단위 집계"),
]

# ─────────────────────────────────────────────────────────────────────────
# F6X — 경주 후에만 존재하는 실측 시장값.
#   학습 입력 금지. 용도 3가지:
#     (1) F6 시장예측 서브모델의 **타깃**
#     (2) 성능 비교 **베이스라인** (인기 1위마 단순 베팅)
#     (3) 수익률 백테스트
# ─────────────────────────────────────────────────────────────────────────
F6X = [
    _c("F6X_win_odds", "F6X", "P", "float32", "확정 단승 배당률", "raceResult_3.winOdds"),
    _c("F6X_plc_odds", "F6X", "P", "float32", "확정 복승 배당률", "raceResult_3.plcOdds"),
    _c("F6X_mkt_prob", "F6X", "P", "float32",
       "시장 내재확률 = (1/winOdds) 를 경주 내 합=1 로 정규화 (오버라운드 제거)", "파생"),
    _c("F6X_mkt_rank", "F6X", "P", "int8", "실제 인기순위 (1=최고인기)", "파생"),
    _c("F6X_overround", "F6X", "P", "float32", "경주 공제율 = Σ(1/odds) - 1", "경주 단위 집계"),
]

# ─────────────────────────────────────────────────────────────────────────
# 경주 내 정규화 (v1.1)
#   Bolton & Chapman / conditional logit 은 애초에 '경주 안에서의 상대 비교'가 전제다.
#   F3_jk_win_rate_365=0.15 가 좋은 값인지는 **같은 경주 다른 기수들이 몇인지**에 달렸다.
#   절대값만 주면 모델이 그 관계를 트리 분할로 간접 학습해야 하므로, 아래 피처들에 대해
#     _z  : 경주 내 z-score  (평균 0, 표준편차 1. 크기 정보 보존)
#     _rk : 경주 내 순위 백분위 0~1 (이상치에 강건)
#   두 개를 함께 낸다. 원본 절대값도 그대로 남는다.
NORMALIZE_WITHIN_RACE = [
    "X_wgBudam", "X_rating", "X_wgHr_delta", "X_ilsu",
    "F1_win_rate_life", "F1_win_rate_365", "F1_ordpct_avg5",
    "F1_speed_avg3", "F1_speed_best365", "F1_prize_365",
    "F2_sire_win_rate", "F2_sire_speed_avg", "F2_sire_dist_fit",
    "F3_jk_win_rate_life", "F3_jk_win_rate_365", "F3_tr_win_rate_365",
    "F3_jkhr_win_rate",
    "F4_hr_dist_win_rate", "F4_hr_dist_speed", "F4_hr_wet_win_rate",
    "F5_early_pos", "F5_g3f_time", "F5_pos_gain",
]

# 범주형 — parquet/CSV 로 저장할 때 category dtype 으로 굳힌다.
# LightGBM 은 category dtype 을 그대로 먹는다(별도 인코딩 불필요).
CATEGORICAL = [
    "X_sex", "X_prd_cty", "X_grade", "F1_grade_last", "F1_shoe_type",
    "F4_dist_band", "F4_track_state", "F4_weather", "F5_style",
    "F2_sire_id", "F2_dam_id", "F2_damsire_id", "F2_sire_cty", "F2_damsire_cty",
]


def _norm_cols():
    """NORMALIZE_WITHIN_RACE 각 항목에 대한 _z / _rk 컬럼 정의를 생성."""
    base = {c["name"]: c for blk in (X, F1, F2, F3, F4, F5, F6) for c in blk}
    out = []
    for name in NORMALIZE_WITHIN_RACE:
        src = base.get(name)
        if src is None:
            continue
        for suf, what in (("_z", "경주 내 z-score"), ("_rk", "경주 내 순위 백분위 0~1")):
            out.append(_c(name + suf, src["group"], src["tier"], "float32",
                          f"{what} — {src['desc'][:40]}", "파생"))
    return out


NORM = _norm_cols()
for _c_ in NORM:                       # 원래 그룹 블록에 편입 — features() 가 자동으로 집는다
    {"X": X, "F1": F1, "F2": F2, "F3": F3, "F4": F4, "F5": F5, "F6": F6}[
        _c_["group"]].append(_c_)

FEATURE_BLOCKS = {"X": X, "F1": F1, "F2": F2, "F3": F3, "F4": F4, "F5": F5, "F6": F6}
ALL_COLUMNS = INDEX + TARGETS + X + F1 + F2 + F3 + F4 + F5 + F6 + F6X

# ─────────────────────────────────────────────────────────────────────────
# 절대 피처로 쓰면 안 되는 원본 API 필드 (오늘 기준 스냅샷 = 미래 누수).
# 빌더는 이 이름들이 결과물에 남아있지 않은지 assert 한다.
# ─────────────────────────────────────────────────────────────────────────
FORBIDDEN_SOURCE_FIELDS = [
    # entrySheet_2 / raceHorseResult_2 / raceHorseRating / trtresult
    "rcCntT", "rcCntY", "ord1CntT", "ord1CntY", "ord2CntT", "ord2CntY",
    "ord3CntT", "ord3CntY", "chaksunT", "chaksunY", "chaksun_6m",
    "winRateT", "qnlRateT", "recentOrd", "recentRating", "recentRcDate",
    "winRateTsum", "quRateTsum", "rating1", "rating2", "rating3", "rating4",
    # 마필종합(API42_1) 누적 스냅샷
    "rcCnt", "fstCnt", "sndCnt", "trdCnt", "forthCnt", "fifthCnt",
    "winRate", "quinRate", "avgWinDist", "rankTop", "rankLast",
    "fgnRcCnt", "fgnFstCnt", "fgnSndCnt", "fgnTrdCnt", "fgnAvgWinDist",
]

# ─────────────────────────────────────────────────────────────────────────
# 데이터 분할 — 6명 전원 동일. 경주(race_id)는 절대 쪼개지 않는다.
# 시간 순서 분할이며 셔플 금지 (미래 정보 누수 방지).
# ─────────────────────────────────────────────────────────────────────────
SPLIT = {
    "train": (20100101, 20241231),
    "valid": (20250101, 20251231),
    "test":  (20260101, 20261231),   # 동결. 최종 비교 1회만 사용
}
SPLIT_NOTE = (
    "조교(2010~)·장제(2010~) 커버리지에 맞춰 2010년을 시작점으로 잡았다. "
    "원장 자체는 2000년부터 있으나 2000~2009 구간은 F1 조교/장제 피처가 전부 결측이 되어 "
    "6개 축의 조건이 달라진다. test 는 팀 전체가 최종 1회만 열어본다."
)


# ─────────────────────────────────────────────────────────────────────────
# 조회 헬퍼
# ─────────────────────────────────────────────────────────────────────────
def features(groups=None, tier=None, exclude_leaky=True):
    """학습에 넣을 컬럼명 리스트.

    groups : ["X","F3"] 처럼 그룹 선택. None 이면 X + F1~F6 전부.
    tier   : "A" 면 D-1 확정 컬럼만. "B" 면 A+B. "C" 면 A+B+C.
    """
    if groups is None:
        groups = list(FEATURE_BLOCKS)
    order = {"A": 0, "B": 1, "C": 2, "P": 3}
    cap = order.get(tier, 2) if tier else 2
    out = []
    for g in groups:
        for c in FEATURE_BLOCKS.get(g, []):
            if exclude_leaky and c["tier"] == "P":
                continue
            if order[c["tier"]] > cap:
                continue
            out.append(c["name"])
    return out


def load(split=None, base="dataset/v1/shards"):
    """팀 공유본 로더 — 6명이 같은 방식으로 읽도록.

        df = load("train")          # 학습셋 (2010~2024, 두 샤드 자동 결합)
        df = load(["train","valid"])
        df = load()                 # 전부 (test 포함 — 최종 평가 때만)

    test 는 별도 파일이라 실수로 딸려오지 않는다. 최종 1회만 명시적으로 부를 것.
    """
    import glob
    import pandas as pd
    want = [split] if isinstance(split, str) else (split or ["train", "valid", "test"])
    files = []
    for w in want:
        files += sorted(glob.glob(f"{base}/kra_ml_v1_{w}*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"{base} 에 샤드가 없다. python build_dataset.py 먼저 실행.")
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    for c in CATEGORICAL + ["split"]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df.sort_values(["rcDate", "meet", "rcNo", "X_chulNo"]).reset_index(drop=True)


def group_of(col):
    """컬럼명 → 그룹. 접두사가 곧 그룹이라 문자열만으로 판정된다."""
    return col.split("_", 1)[0] if "_" in col else "IDX"


def weights(emphasis, high=3.0, base=1.0):
    """팀원 1인의 가중치 벡터 — 자기 그룹만 올리고 나머지는 기본값.

        w = weights("F3")            # 기수 실력 강조 모델
        w = weights("F5", high=4.0)  # 주행 스타일 강조, 더 세게
    """
    return {c: (high if group_of(c) == emphasis else base)
            for c in features()}


TEAM = {
    "F1": "최근 성적",
    "F2": "부모 성적",
    "F3": "기수 실력",
    "F4": "거리·날씨",
    "F5": "주행 스타일",
    "F6": "인기도",
}


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"스키마 v{SCHEMA_VERSION}\n")
    print(f"{'그룹':<5} {'컬럼수':>5}  설명")
    print("-" * 74)
    for g, desc in GROUPS.items():
        print(f"{g:<5} {len(FEATURE_BLOCKS[g]):>5}  {desc}")
    print("-" * 74)
    print(f"{'IDX':<5} {len(INDEX):>5}  인덱스/식별자 (피처 아님)")
    print(f"{'Y':<5} {len(TARGETS):>5}  타깃")
    print(f"{'F6X':<5} {len(F6X):>5}  경주 후 시장값 (학습 입력 금지)")
    print("-" * 74)
    print(f"{'합계':<5} {len(ALL_COLUMNS):>5}  전체 컬럼")
    print(f"\n학습 피처 (기본, tier C까지): {len(features())}개")
    for t in ("A", "B", "C"):
        print(f"  tier {t} 까지: {len(features(tier=t)):>3}개  — {TIERS[t]}")
    print("\n팀원별 담당 그룹")
    for g, name in TEAM.items():
        print(f"  {g}  {name:<10} 전용 {len(FEATURE_BLOCKS[g]):>2}개 + 공통 X {len(X)}개")
