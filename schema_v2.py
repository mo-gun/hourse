# -*- coding: utf-8 -*-
"""
경마 예측 플랫폼 — 데이터 스키마 v2.

v1(연구용 172컬럼)과 목적이 다르다. v2 는 **제품**을 위한 것이고 세 갈래로 나온다.

    dataset/v2/model/   학습용. 피처 40개 내외 — 사용자가 UI 에서 만질 수 있는 규모
    dataset/v2/game/    게임 구동용. 경주 카드 + 7승식 적중배당
    dataset/v2/sim/     경기 흐름 재생용. 구간별 통과순위·시간 (★ 학습에 안 씀)

v1 대비 바뀐 것
  1. 게임풀을 **영구 예약**한다 (시간 블록 아님).
     2025~2026 도 학습에 넣어 주말 실시간 예측에 쓰려면, 게임풀이 뒤쪽 시간 블록이면
     재학습할 때마다 오염된다. 그래서 2015~2024 개최일의 15% 를 고정 시드로 뽑아
     game_holdout.json 에 박아두고, 빌더가 매번 그 파일을 읽어 학습에서 뺀다.
  2. 인기도(F6)가 실제 피처가 된다.
     과거 경주 리플레이라 그 경기 배당이 이미 확정돼 있다. v1 에서 서브모델로 우회한
     이유(경주 전 배당 조회 불가)가 리플레이 맥락에선 사라진다.
     ⚠ 단 우리가 가진 건 확정배당(경주 후)이라 발주 직전 배당보다 약간 더 '똑똑'하다.
        베이스 모델 성적을 "시장을 이겼다"로 해석하면 안 된다.
  3. 피처마다 **변동성 등급**을 붙인다. 기획의 "경기마다 변하는 것만 조절" 을 위해서다.
  4. 피처마다 **문헌 근거**를 붙인다. 근거 없는 건 lit=None 으로 명시한다.

v1 에서 발견해 제거한 것
  - X_ilsu: raceResult_3.ilsu 는 휴양일수가 아니라 **그 해 n번째 개최일**이었다.
    (2015 서울 실측: 개최일 순번과 상관 1.0000, 완전일치 97.9%. 경주 내 값이 항상 하나)
    실제 휴양일수는 F1_layoff_days 로 원장에서 직접 계산한다.
  - F5_style_fit / F5_gate_style_fit: 내가 만든 상호작용항. 문헌 근거 없고 미검증.
  - _z 의 fillna(0): z-score 에서 0 은 '평균'이라 결측을 평균으로 위장했다. NaN 유지로 바꿈.
"""

SCHEMA_VERSION = "2.2.1"

MODEL_DIR = "dataset/v2/model"
GAME_DIR = "dataset/v2/game"
SIM_DIR = "dataset/v2/sim"
HOLDOUT_FILE = "dataset/v2/game_holdout.json"

# ─────────────────────────────────────────────────────────────────────────
# 분할
#   game    : 영구 예약. 학습·검증·평가 어디에도 안 쓴다. 사용자 플레이 전용.
#   train   : 나머지 전부 (2025~2026 포함). 매주 월요일 재학습 시 최신 주말이 여기 들어온다.
#   valid   : 재학습 시점 기준 직전 26주 — 하이퍼파라미터 튜닝
#   test    : 재학습 시점 기준 직전 8주 — 팀원 6명 모델 비교용, 최종 1회
# valid/test 를 고정 연도가 아니라 **재학습 시점 기준 상대 구간**으로 잡는 이유:
# 매주 재학습하는데 평가 구간이 2023년에 고정돼 있으면 갈수록 무의미해진다.
# ─────────────────────────────────────────────────────────────────────────
HOLDOUT = {
    "year_from": 2015,
    "year_to": 2024,
    "day_frac": 0.15,      # 개최일(경마장별) 기준 15% ≈ 3,700경주
    "seed": 20260901,      # ★ 절대 바꾸지 말 것. 바꾸면 예약이 무의미해진다
}
VALID_WEEKS = 26
TEST_WEEKS = 16
# test 를 8주(395경주)에서 16주로 늘린 이유 — 검정력.
#   395경주에서 top-1 적중률의 표준오차는 √(0.36·0.64/395) ≈ 2.4%p 라
#   팀원 간 5%p 미만 차이가 구분되지 않는다. 16주면 ~790경주, 표준오차 1.7%p.
# valid/test 는 **개발 단계 전용**이다.
#   하이퍼파라미터가 정해지고 6명 비교가 끝나면, 운영 모델은 game 을 뺀 전부
#   (train+valid+test)로 매주 재학습한다. 그래야 주말 실시간 예측에 최신 10개월이 들어간다.
RETRAIN = "매주 월요일. 주말 경주는 발주 후 15분 내 원장 반영됨(실측)."

# ─────────────────────────────────────────────────────────────────────────
# 변동성 — 기획의 "경기마다 변하는 것만 사용자가 조절"
#   race   : 경주마다 바뀐다. 당일 확인 가능. **UI 슬라이더 대상**
#   form   : 몇 주 단위로 바뀐다 (최근 성적·조교)
#   static : 말 고유. 거의 안 변한다 (혈통·성별·산지)
# ─────────────────────────────────────────────────────────────────────────
VOL = {"race": "경기마다 변함 — 당일 확인 가능",
       "form": "주 단위로 변함 — 최근 폼",
       "static": "말 고유 — 거의 불변"}

# 가용 시점 — 2026-08-28 발주 시간대 실측 (probe_realtime.py)
# ★ B 의 "T-60~90분" 은 docs/주제기획 의 [추정] 이 그대로 굳은 값이었는데, 2026-09-10
#   제주 실측이 **아래쪽 끝(T-60)은 맞고 위쪽(T-90)은 아님**을 보였다 — 4R 이 T-65 에는
#   비어 있고 T-61 에는 차 있다. 전환 구간을 4분으로 좁혀 T-61~65 로 적는다.
TIER = {"A": "D-1 확정", "B": "발주 T-61~65분 (마체중)",
        "C": "발주 T-10분 (날씨·주로)", "G": "리플레이 전용 — 실시간 예측엔 못 씀",
        "P": "경주 후 — 학습 입력 금지"}

# 문헌 출처 (사용자 선행연구 문서 + 직접 검색)
LIT = {
    "K1": "최혜민 외(2015) 서울경마 우승마 예측 — 과거 우승 경력, 기수 우승 경력이 핵심 변수",
    "K2": "정준형 외(2024) LTR — Shapley 상위에 출발훈련·질병진단·훈련기록",
    "K3": "So Yubin 외(2025) LTR+웹서비스 — 최근성적/통산평균착순/부담중량/마령 4계열",
    "B7": "Borowski 외(2021) 폴란드 3,782경주 — **과거 획득 상금이 최중요 피처**",
    "P4": "Pudaruth 외(2013) — 기수·경험·배당·게이트·거리적성·중량·레이팅 가중확률",
    "BT": "Benter(1994) — 펀더멘털 모델 + 공개배당 2단계",
    "BC": "Bolton&Chapman(1986) — 경주 단위 조건부로짓",
    "LS": "Lessmann 외(2010) — 경쟁구조 반영 RF 가 조건부로짓 상회",
    "PACE": "실무(pace figure) — (Par−Actual). 선행마 다수 → 페이스 붕괴 → 추입 유리",
    "EBV": "KRA studbook 육종가 — BLUP 유전능력평가. 반기 스냅샷 20개를 as-of 조인",
}


def _c(name, grp, vol, tier, dtype, desc, lit=None, src=""):
    return {"name": name, "group": grp, "vol": vol, "tier": tier,
            "dtype": dtype, "desc": desc, "lit": lit, "src": src}


# ═════════════════════════════════════════════════════════════════════════
# 인덱스 · 타깃
# ═════════════════════════════════════════════════════════════════════════
INDEX = [
    _c("race_id", "IDX", "race", "A", "str", "'{rcDate}_{meet}_{rcNo}'. 랭킹 group 키", "BC"),
    _c("row_id", "IDX", "race", "A", "str", "'{race_id}_{chulNo}'"),
    _c("rcDate", "IDX", "race", "A", "int32", "경주일 YYYYMMDD"),
    _c("meet", "IDX", "race", "A", "int8", "1서울 2제주 3부산경남"),
    _c("rcNo", "IDX", "race", "A", "int8", "경주번호"),
    _c("hrNo", "IDX", "static", "A", "str", "마번"),
    _c("hrName", "IDX", "static", "A", "str", "마명 (표시용)"),
    _c("jkNo", "IDX", "race", "A", "str", "기수번호"),
    _c("trNo", "IDX", "static", "A", "str", "조교사번호"),
    _c("split", "IDX", "race", "A", "category", "train/valid/test/**game**"),
]

TARGETS = [
    _c("y_ord", "Y", "race", "P", "int8", "착순"),
    _c("y_win", "Y", "race", "P", "int8", "1착 0/1"),
    _c("y_plc", "Y", "race", "P", "int8", "3착 이내 0/1 (7두 미만이면 2착)"),
    _c("y_rel", "Y", "race", "P", "int8", "랭킹 라벨 max(0, dusu-ord). lambdarank", "BC"),
    _c("y_speed_fig", "Y", "race", "P", "float32", "스피드지수 (기준기록 대비)", "PACE"),
]

# ═════════════════════════════════════════════════════════════════════════
# X — 공통 (10)
# ═════════════════════════════════════════════════════════════════════════
X = [
    _c("X_age", "X", "static", "A", "int8", "마령", "K3"),
    _c("X_sex", "X", "static", "A", "category", "성별 수/암/거"),
    _c("X_prd_cty", "X", "static", "A", "category", "산지 (원장 name 컬럼)"),
    _c("X_rcDist", "X", "race", "A", "int16", "경주거리 m"),
    _c("X_grade", "X", "race", "A", "category", "등급 국1~6/혼"),
    _c("X_dusu", "X", "race", "A", "int8", "출주두수 — 경주 난이도 정규화"),
    _c("X_chulNo", "X", "race", "A", "int8", "게이트", "P4"),
    _c("X_gate_rel", "X", "race", "A", "float32", "게이트 ÷ 두수", "P4"),
    _c("X_wgBudam", "X", "race", "A", "float32", "부담중량 kg", "K3"),
    # ★ 빌더(build_v2.py:111~112)가 계산해 놓고 여기 등록이 없어 finalize 에서 버려졌다.
    #   _norm_cols() 주석에 적힌 v2.0.0 의 실수를 X 블록에서 되풀이한 것이다 —
    #   **features() 가 반환하는 것만 파케이에 남는다.**  (2026-09-10 발견·등록)
    #
    #   tier=B 근거 — 2026-09-10 제주 실측 (probe_wghr_raceday.py). 4R 이 채워지는 순간을
    #   4분 간격으로 포착했다:
    #     경주   발주     관측     T-minus   원장 wgHr   API25_1 wgHr
    #     4R    14:30    13:25     T-65분      0/8          0/8      ← 아직 없음
    #     4R    14:30    13:29     T-61분      8/8          8/8      ← 채워짐
    #     3R    14:00    13:24     T-36분      7/7          7/7      (이미 채워진 상태)
    #   즉 **각 경주 발주 약 T-61~65분에 채워진다** — 전환이 그 4분 구간 안에서 일어났다.
    #   3R 이 T-36 에 채워져 있던 것도 같은 규칙이다(그 경주는 ~13:00 에 찼을 것).
    #   발주 전에 얻을 수 있으므로 주말 실시간(73→75피처) 모델에 쓸 수 있다.
    #
    #   ⚠ tier=A 는 아니다 — D-1 에는 없다. 2026-09-10(D-1) 실측으로 다가올 경주일
    #     5건(9/11 제주·부경, 9/12 서울·제주, 9/13 서울) 전부 원장·API25_1 양쪽
    #     wgHr 충전 **0/423두**. 출전표 행은 이미 있는데 체중만 비어 있다.
    #
    #   ⚠ 운영 제약 — 이 피처를 실시간 모델에 실제로 쓰려면 추론이 **각 경주 발주
    #     T-60분 안쪽**에 돌아야 한다. 주말 배치를 아침에 한 번 돌리면 뒤 경주는
    #     체중이 아직 비어 있어 결측으로 들어간다(LightGBM 은 NaN, 신경망은 train
    #     중앙값 대치 — 둘 다 조용히 열화된다). 경주별/롤링 추론 일정이 전제다.
    #     추론 36경주가 278ms 라(basemodel/README §6) 일정만 바꾸면 되는 문제다.
    #
    #   ✔ 서빙 경로 값 일치 확인 — 학습은 원장(API4_3) wgHr, 실시간 서빙은
    #     API25_1/entryHorseWeightInfo_1 이 될 텐데 두 엔드포인트의 wgHr 가
    #     지난 경주일 4건 365두에서 **불일치 0** 이었다(2026-09-10 실측).
    #
    #   lit=None — 마체중을 직접 다룬 문헌 근거가 없다. 우리 실측만 있다:
    #   게임풀 조건부 로짓에서 우리 모델 위에 유의(LR 15.79, p=7.1e-05)하지만
    #   시장 위에는 아니고(p=0.44), **적중률 개선은 없다** — valid A/B 실측은
    #   basemodel/experiments/ab_wghr.py 와 basemodel/ledger.md 참조.
    #   ⛔ tier=P 로 **비활성** (2026-09-10 결정). 등록 자체는 위 주석의 버그 수정이지만,
    #     적중률 개선이 0 인데(리플레이 −0.16%p p=0.864 · 실시간 −0.24%p p=0.795)
    #     프로젝트 공용 어휘인 **77/73 을 79/75 로 바꿀 값이 없다.** 피처 수를 유지하고
    #     가용시점·값 일치·A/B 근거만 남긴다. `F1_prize_life` 와 같은 처리다.
    #     되살리려면 tier 를 "B" 로 바꾸면 된다 — 파케이에 컬럼이 이미 들어 있어
    #     **재빌드가 필요 없다**(features() 가 반환하는 것만 모델이 본다).
    _c("X_wgHr", "X", "race", "P", "float32",
       "마체중 kg — 당일 계체값 [비활성] 발주 T-61~65분 가용, 적중률 개선 없음"),
    _c("X_wgHr_delta", "X", "race", "P", "float32",
       "마체중 증감 kg — 직전 출전 대비 [비활성]"),
    _c("X_rating", "X", "form", "A", "float32", "경주시점 레이팅 (충전 46%)", "P4"),
]

# ═════════════════════════════════════════════════════════════════════════
# F1 — 최근 성적 · 컨디션 (12)
# ═════════════════════════════════════════════════════════════════════════
F1 = [
    _c("F1_starts_life", "F1", "form", "A", "int16",
       "통산 출주수 (as-of) — **F1 신뢰도 가중치로 쓸 것**. 전체 행의 24.5%가 3경주 이하"),
    _c("F1_win_rate_life", "F1", "form", "A", "float32", "통산 승률 (as-of)", "K1"),
    _c("F1_ord_avg3", "F1", "form", "A", "float32", "최근 3경주 평균 착순", "K3"),
    _c("F1_ordpct_avg5", "F1", "form", "A", "float32", "최근 5경주 상대착순 (두수 보정)", "K3"),
    _c("F1_speed_avg3", "F1", "form", "A", "float32", "최근 3경주 평균 스피드지수", "PACE"),
    # ⛔ tier=P — EDA(2026-09) 로 비활성. train 13.96% → valid 8.88% 로 무너진다.
    #    아무거나 찍기(9.77%)보다 낮다. 상금 액수가 해마다 올라 옛날 말과 요즘 말을
    #    비교할 수 없기 때문. Borowski 가 최중요로 꼽은 피처지만 우리 데이터에서는
    #    명목금액이라 성립하지 않는다. 되살리려면 연도별 물가·상금규모로 정규화할 것.
    #    _z / _rk 는 tier 를 상속하므로 이 한 줄로 3개가 같이 빠진다.
    _c("F1_prize_life", "F1", "form", "P", "float32",
       "통산 획득상금 (as-of) — [비활성] 연도 간 비교 불가", "B7"),
    _c("F1_prize_365", "F1", "form", "A", "float32", "최근 1년 획득상금", "B7"),
    _c("F1_grade_move", "F1", "form", "A", "int8", "등급 변동 +승급/0/-강급"),
    _c("F1_layoff_days", "F1", "race", "A", "int16",
       "**진짜 휴양일수** — 원장 ilsu 는 개최일 순번이라 못 씀. 직전 출전 이후 경과일"),
    _c("F1_first_start", "F1", "form", "A", "int8", "신마 0/1 (전체 6.7%)"),
    _c("F1_tr_sessions_28d", "F1", "form", "A", "int16", "28일 조교 횟수", "K2"),
    _c("F1_tr_jk_ridden_28d", "F1", "form", "A", "int16",
       "28일 중 **기수 기승** 조교 횟수 — 본격 준비 신호 (전체의 26%)", "K2"),
]

# 신청 완료 · 수집 대기 — 문헌 Shapley 상위. 도착하면 F1 에 합류한다.
F1_PENDING = [
    _c("F1_gate_train_28d", "F1", "form", "A", "int16",
       "28일 출발조교 횟수", "K2", "API329·331·330"),
    _c("F1_gate_train_life", "F1", "form", "A", "int16", "통산 출발조교 횟수", "K2"),
    _c("F1_clinic_365", "F1", "form", "A", "int16",
       "경주 전 1년 진료 건수 — API 이름부터 '경주 전'이라 누수 구조적으로 불가", "K2", "API141/clincList"),
    _c("F1_clinic_days_since", "F1", "form", "A", "int16", "최종 진료 경과일", "K2"),
]

# ═════════════════════════════════════════════════════════════════════════
# F2 — 부모 성적 (5). 신마 6.7% 에서 유일한 단서라 적어도 유지한다.
# ═════════════════════════════════════════════════════════════════════════
F2 = [
    # ── 육종가(EBV) — KRA studbook 반기 스냅샷 20개(2016-07~2026-06)에서 as-of 조인 ──
    #  BLUP 유전능력 추정치. 경주연도·경마장·거리·성별·연령·조교사·기수 환경효과를
    #  보정하고 직계·방계 혈통 성적까지 활용한다. 내가 만들던 단순 부마 승률 집계보다 낫다.
    #  ★ 육종가는 그 말 자신의 성적을 포함해 계산되므로 **경주일 직전 스냅샷**만 붙인다.
    #    (ebv_join.py 가 강제. 검산 결과 위반 0건)
    #  커버리지: 2017~2026 년 62~72%. 2016 년 28%, 그 이전 0%(스냅샷 시작 전).
    _c("F2_ebv_prize", "F2", "static", "A", "float32",
       "상금 육종가 — 평균100·표준편차20 표준화. 120=상위16%, 140=상위2.3%", "EBV"),
    _c("F2_ebv_acc", "F2", "static", "A", "float32",
       "상금 육종가 정확도 0~1 — **F2 신뢰도 가중치로 쓸 것**. 0.7 미만은 변동폭 큼", "EBV"),
    _c("F2_ebv_sprint", "F2", "static", "A", "float32", "단거리 육종가 — 유전적 단거리 적성", "EBV"),
    _c("F2_ebv_route", "F2", "static", "A", "float32", "중장거리 육종가", "EBV"),
    _c("F2_ebv_dist_fit", "F2", "race", "A", "float32",
       "이번 거리에 맞춘 육종가 = 단거리/중장거리 중 해당 쪽 (파생)", "EBV"),
    _c("F2_ebv_time", "F2", "static", "A", "float32",
       "주파기록 육종가(초) — 음수일수록 빠름", "EBV"),
    _c("F2_ebv_ssgblup", "F2", "static", "A", "float32",
       "SSGBLUP 육종가 — DNA 1만두 유전체 정보 반영", "EBV"),
    _c("F2_inbreeding", "F2", "static", "A", "float32", "근교계수 %", "EBV"),
    # ── API 혈통 — 육종가가 못 덮는 2016년 이전과 부마 계통 연결용 ──
    _c("F2_sire_id", "F2", "static", "A", "category", "부마 마번 (마필종합 API)"),
    _c("F2_sire_win_rate", "F2", "static", "A", "float32", "부마 자마 승률 (원장 as-of 집계)"),
    _c("F2_sire_prog_n", "F2", "static", "A", "int16", "부마 자마 두수 — 신뢰도 가중치"),
]

# ═════════════════════════════════════════════════════════════════════════
# F3 — 기수 · 조교사 (6)
# ═════════════════════════════════════════════════════════════════════════
F3 = [
    _c("F3_jk_win_rate_365", "F3", "race", "A", "float32",
       "기수 최근 1년 승률 — 실측 2.2~23.2%로 10배 차이", "K1"),
    _c("F3_jk_win_rate_life", "F3", "race", "A", "float32", "기수 통산 승률 (as-of)", "K1"),
    _c("F3_jk_win_rate_dist", "F3", "race", "A", "float32", "기수의 이번 거리대 승률", "P4"),
    _c("F3_tr_win_rate_365", "F3", "static", "A", "float32", "조교사 최근 1년 승률", "K1"),
    _c("F3_jkhr_starts", "F3", "race", "A", "int16", "기수-말 조합 누적 출주"),
    _c("F3_jkhr_win_rate", "F3", "race", "A", "float32", "기수-말 조합 승률"),
]

# ═════════════════════════════════════════════════════════════════════════
# F4 — 거리 · 날씨 · 주로 (7)
# ═════════════════════════════════════════════════════════════════════════
F4 = [
    _c("F4_hr_dist_win_rate", "F4", "race", "A", "float32", "이 말의 이번 거리대 승률", "P4"),
    _c("F4_dist_gap", "F4", "race", "A", "int16", "|이번 거리 − 최적 거리|", "P4"),
    # ⛔ tier=P — F1_win_rate_life 와 상관 0.995. 말이 대개 한 경마장에서만 뛰어
    #    사실상 같은 숫자다. 적중률도 F1 쪽이 높아(27.10 vs 22.67) 그쪽을 남긴다.
    _c("F4_hr_meet_win_rate", "F4", "race", "P", "float32",
       "이 말의 이번 경마장 승률 — [비활성] F1_win_rate_life 와 중복(r=0.995)"),
    _c("F4_hr_wet_win_rate", "F4", "race", "A", "float32", "이 말의 불량주로 승률"),
    _c("F4_track_moist", "F4", "race", "C", "float32", "주로 함수율 % — **발주 T-10분**"),
    _c("F4_weather", "F4", "race", "C", "category", "날씨 — **발주 T-10분**"),
    # ⛔ tier=P — 분할끼리 월이 겹치지 않는다(valid 11~5월, test 5~8월).
    #    PSI valid 3.14 / test 5.31 로 전 피처 중 최악. 계절성은 날씨·주로로 대신한다.
    _c("F4_month", "F4", "race", "P", "int8", "월 — [비활성] 분할 간 미겹침(PSI 5.31)"),
]

# ═════════════════════════════════════════════════════════════════════════
# F5 — 주행 스타일 (6). 시뮬레이션이 아니라 **예측용 요약**이다.
#      경기 흐름 재생에 쓸 원시 구간기록은 sim/ 로 따로 나간다.
# ═════════════════════════════════════════════════════════════════════════
F5 = [
    _c("F5_style", "F5", "form", "A", "category", "각질 선행/선입/중단/추입", "PACE"),
    _c("F5_early_pos", "F5", "form", "A", "float32", "과거 평균 초반 상대위치 (0=선두)", "PACE"),
    _c("F5_pos_gain", "F5", "form", "A", "float32", "초반→결승 순위 상승폭. 양수=추입형", "PACE"),
    _c("F5_g3f_time", "F5", "form", "A", "float32", "과거 평균 상3F (정규화) — 막판 각력", "PACE"),
    _c("F5_race_n_front", "F5", "race", "A", "int8",
       "🏁 이 경주 선행마 두수 — 많으면 페이스 붕괴 → 추입 유리", "PACE"),
    _c("F5_sect_n", "F5", "form", "A", "int8", "구간기록 보유 과거 경주 수 — 신뢰도"),
]

# ═════════════════════════════════════════════════════════════════════════
# F6 — 인기도 (4). v1 과 달리 **실제 값**이다.
#   과거 리플레이라 그 경기 배당이 확정돼 있고, 기획상 사용자에게도 보여준다.
#   ⚠ tier=G : 리플레이 전용. 주말 실시간 예측에는 쓸 수 없다(발주 전 배당 미제공, 실측).
#      실시간 모델은 features(exclude_tier="G") 로 F6 를 빼고 학습·추론해야 한다.
# ═════════════════════════════════════════════════════════════════════════
F6 = [
    _c("F6_mkt_prob", "F6", "race", "G", "float32",
       "시장 내재확률 = (1/단승배당) 경주 내 합=1 정규화 (오버라운드 제거)", "BT"),
    _c("F6_mkt_rank", "F6", "race", "G", "int8", "인기순위 1=최고인기", "BT"),
    _c("F6_mkt_prob_z", "F6", "race", "G", "float32", "경주 내 z-score", "BC"),
    _c("F6_field_entropy", "F6", "race", "G", "float32",
       "🏁 경주 내 확률분포 엔트로피 — 혼전(높음) vs 독주(낮음)"),
]

# 경주 내 정규화 — v1 처럼 무차별로 붙이지 않고, 상대비교가 실제로 의미있는 것만.
# ⚠ v1 버그: 결측 z 를 0(=평균)으로 채웠다. v2 는 NaN 유지.
NORMALIZE_WITHIN_RACE = [
    "X_wgBudam", "X_rating",
    "F1_win_rate_life", "F1_ordpct_avg5", "F1_speed_avg3", "F1_prize_life",
    "F2_ebv_prize", "F2_ebv_dist_fit",
    "F3_jk_win_rate_365", "F3_tr_win_rate_365",
    "F4_hr_dist_win_rate",
    "F5_early_pos", "F5_g3f_time",
]

CATEGORICAL = ["X_sex", "X_prd_cty", "X_grade", "F4_weather", "F5_style", "F2_sire_id"]


def _norm_cols():
    """NORMALIZE_WITHIN_RACE 각 항목의 _z / _rk 를 스키마에 등록한다.

    ★ 등록을 빠뜨리면 빌더가 계산해놓고 finalize 에서 버린다.
      v2.0.0 에서 실제로 그랬다 — 26컬럼이 만들어졌다가 파케이에 안 남았다.
      features() 가 반환하는 것만 살아남는다.
    """
    base = {c["name"]: c for blk in (X, F1, F2, F3, F4, F5, F6) for c in blk}
    out = []
    for name in NORMALIZE_WITHIN_RACE:
        src = base.get(name)
        if src is None:
            continue
        for suf, what in (("_z", "경주 내 z-score"), ("_rk", "경주 내 순위 백분위")):
            out.append(_c(name + suf, src["group"], src["vol"], src["tier"], "float32",
                          f"{what} — {src['desc'][:38]}", src["lit"]))
    return out


for _nc in _norm_cols():
    {"X": X, "F1": F1, "F2": F2, "F3": F3, "F4": F4,
     "F5": F5, "F6": F6}[_nc["group"]].append(_nc)

FEATURE_BLOCKS = {"X": X, "F1": F1, "F2": F2, "F3": F3, "F4": F4, "F5": F5, "F6": F6}


# 절대 피처로 쓰지 않는 원본 필드 (오늘 기준 스냅샷 = 미래 누수)
FORBIDDEN = [
    "rcCntT", "rcCntY", "ord1CntT", "ord1CntY", "ord2CntT", "ord3CntT",
    "chaksunT", "chaksunY", "winRateT", "qnlRateT", "recentOrd", "recentRating",
    "rcCnt", "fstCnt", "sndCnt", "trdCnt", "winRate", "quinRate", "avgWinDist",
    "rankTop", "rankLast",
    "ilsu",   # ★ 휴양일수가 아니라 개최일 순번. F1_layoff_days 를 쓸 것
]

# ═════════════════════════════════════════════════════════════════════════
# game/ — 게임 구동용. 학습셋과 스키마가 다르다.
# ═════════════════════════════════════════════════════════════════════════
GAME_CARD = [           # 경주 카드 — 방에 띄울 정보
    "race_id", "rcDate", "meet", "rcNo", "rcDist", "grade", "weather",
    "track_state", "track_moist", "dusu", "prize1", "stTime",
]
GAME_ENTRY = [          # 출주마 카드
    "race_id", "chulNo", "hrNo", "hrName", "age", "sex", "prd_cty",
    "wgBudam", "wgHr", "wgHr_delta", "jkNo", "jkName", "trNo", "trName",
    "rating", "mkt_prob", "mkt_rank", "y_ord",
]
# 7승식 적중배당 — 전 조합은 경주당 1,040행이라 과하다. **적중 조합만** 저장하면 7행/경주.
GAME_PAYOUT = ["race_id", "pool", "combo", "odds"]
POOLS = ["단승식", "연승식", "복승식", "쌍승식", "복연승식", "삼복승식", "삼쌍승식"]

# ═════════════════════════════════════════════════════════════════════════
# sim/ — 경기 흐름 재생용. ★ 학습에 절대 안 쓴다.
#   경마장마다 구간 컬럼이 다르므로 표준 4지점으로 정규화해서 낸다.
# ═════════════════════════════════════════════════════════════════════════
SIM_POINTS = ["S1F", "C2", "C4", "G3F", "FIN"]
SIM_COLS = ["race_id", "chulNo", "hrNo", "point", "pass_ord", "pass_time", "rel_pos"]


def features(groups=None, vol=None, max_tier="C", exclude_tier=()):
    """학습에 넣을 컬럼명.

        features()                          # 전체 (리플레이용, F6 포함)
        features(exclude_tier=("G",))       # 주말 실시간용 — 인기도 제외
        features(vol="race")                # UI 슬라이더 대상만
        features(groups=["X","F3"])         # 기수 담당
    """
    order = {"A": 0, "B": 1, "C": 2, "G": 3, "P": 9}
    cap = order.get(max_tier, 2)
    out = []
    for g in (groups or FEATURE_BLOCKS):
        for c in FEATURE_BLOCKS.get(g, []):
            if c["tier"] in exclude_tier or c["tier"] == "P":
                continue
            if c["tier"] != "G" and order[c["tier"]] > cap:
                continue
            if vol and c["vol"] != vol:
                continue
            out.append(c["name"])
    return out


ALL = INDEX + TARGETS + X + F1 + F2 + F3 + F4 + F5 + F6


# ═════════════════════════════════════════════════════════════════════════
# 로드 후 정리 — EDA(2026-09) 지적사항
#
# parquet 에 이미 들어간 값 중 손봐야 하는 것들이다. 재빌드 없이 쓰라고
# 여기에 둔다. 빌더도 같은 함수를 쓰므로 다음 빌드부터는 이미 정리된 상태로
# 나오고, 그때도 이 함수를 다시 부르는 것은 무해하다(멱등).
# ═════════════════════════════════════════════════════════════════════════

ODDS_NONE = 900.0          # winOdds 가 이 값 이상이면 "배당 없음"(특수값 9999.9)
GRADE_NONE = ("", "-", "nan", "None")


def normalize_grade(s):
    """등급 표기 통합. `국5등급`/`국5` → `국5`, `제오픈`/`제OPEN` → `제OPEN`.

    train 기준 42개 표기가 27개로 줄어든다. 통합 안 하면 모델이 `국5` 20,088행과
    `국5등급` 40,776행을 서로 다른 등급으로 배운다.
    """
    if s is None:
        return None
    t = str(s).strip()
    if t in GRADE_NONE:
        return None
    if t.endswith("등급"):
        t = t[:-2]
    t = t.replace("오픈", "OPEN").replace("Open", "OPEN").replace("open", "OPEN")
    return t or None


def clean(df):
    """로드 직후 한 번 호출한다. 표기·특수값을 정리한 새 DataFrame 을 돌려준다.

        import pandas as pd, schema_v2 as S
        tr = S.clean(pd.read_parquet("model/train.parquet"))

    하는 일 셋.
      1. `X_grade` 표기 통합 (42 → 27)
      2. `F2_sire_id` 의 결측 토큰 통일 — NaN 과 `'-'`(12,011행)가 섞여 있어
         범주형으로 쓰면 빈 값이 두 종류로 갈린다
      3. `winOdds` 특수값 9999.9 를 NaN 으로 — 진짜 배당으로 계산하면 수익률이
         +250% 로 터진다(실제 -21.4%)
    """
    out = df.copy()
    if "X_grade" in out.columns:
        out["X_grade"] = out["X_grade"].astype("object").map(normalize_grade)
    if "F2_sire_id" in out.columns:
        out["F2_sire_id"] = out["F2_sire_id"].astype("object").replace(
            {k: None for k in GRADE_NONE})
    for c in ("winOdds", "plcOdds", "odds"):
        if c in out.columns:
            out.loc[out[c] >= ODDS_NONE, c] = None
    # ★ dtype 복원. 범주형을 object 로 두면 LightGBM 이 그대로 거부한다
    #   ("pandas dtypes must be int, float or bool"). 값을 바꾼 뒤 category 로 되돌린다.
    for c in CATEGORICAL:
        if c in out.columns:
            out[c] = out[c].astype("category")
    return out


def flat_odds_races(df):
    """배당이 없어 경주 안 확률이 전부 같은 경주의 race_id 집합.

    2020년 15.4% / 2021년 7.0% 가 여기 걸린다. 겉보기엔 정상이라 그냥 지나간다.
    시장 베이스라인을 재거나 배당을 쓰는 실험에서는 빼야 한다.
    """
    g = df.groupby("race_id")["F6_mkt_prob"]
    span = g.transform("max") - g.transform("min")
    return set(df.loc[span.fillna(0) < 1e-6, "race_id"].unique())


def usable(df, drop_flat_odds=True):
    """학습·평가에 쓸 행만 남긴다. 기본은 배당 없는 경주 제외.

    인기도(F6_*)를 안 쓰는 실시간 모델이라면 `drop_flat_odds=False` 로 두어도 된다 —
    배당이 없을 뿐 나머지 피처는 정상이다.
    """
    if not drop_flat_odds or "F6_mkt_prob" not in df.columns:
        return df
    bad = flat_odds_races(df)
    return df[~df["race_id"].isin(bad)]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"스키마 v{SCHEMA_VERSION}\n")
    print(f"{'그룹':<4}{'수':>3}  {'race':>5}{'form':>5}{'static':>7}   문헌근거")
    print("-" * 72)
    for g, blk in FEATURE_BLOCKS.items():
        v = {k: sum(1 for c in blk if c["vol"] == k) for k in VOL}
        lits = sorted({c["lit"] for c in blk if c["lit"]})
        print(f"{g:<4}{len(blk):>3}  {v['race']:>5}{v['form']:>5}{v['static']:>7}   {','.join(lits)}")
    print("-" * 72)
    n = sum(len(b) for b in FEATURE_BLOCKS.values())
    print(f"기본 피처 {n}개 + 경주내 정규화 {len(NORMALIZE_WITHIN_RACE)*2}개 = {n + len(NORMALIZE_WITHIN_RACE)*2}")
    print(f"수집 대기(신청완료) {len(F1_PENDING)}개 → 도착 시 {n + len(F1_PENDING)}개")
    print()
    print(f"UI 슬라이더 대상 (vol=race): {len(features(vol='race'))}개")
    print(f"주말 실시간용 (인기도 제외): {len(features(exclude_tier=('G',)))}개")
    print(f"\n분할: game=영구예약({HOLDOUT['year_from']}~{HOLDOUT['year_to']} 개최일 "
          f"{HOLDOUT['day_frac']*100:.0f}%, seed={HOLDOUT['seed']})")
    print(f"      valid=직전 {VALID_WEEKS}주 / test=직전 {TEST_WEEKS}주 / train=나머지 전부")
    print(f"      재학습: {RETRAIN}")
    print("\n문헌")
    for k, v in LIT.items():
        print(f"  {k}  {v}")
