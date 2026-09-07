# 다음 작업 지시서 (새 Claude 세션용)

**갱신 2026-09-07** — API 상태를 다시 실측해서 반영했다. 아래 코드블록을 통째로 복붙하면 된다.

---

```
경마 예측 프로젝트 작업이야. 아래 순서대로 진행해줘.

## 환경

- 개인 작업 폴더: C:\Users\SSAFY\Desktop\주제선정
  - data/raw/          원본 (ledger 190MB, aux_train 139MB, aux_shoe 65MB, ebv 23MB) ★ 여기에만 있음
  - dataset/v2/        빌드 출력
  - tools/             API 백필 스크립트
  - schema_v2.py build_v2.py validate_v2.py aux_join.py ebv_join.py
  - kra_client.py      ★ 검증된 KRA 엔드포인트 20종 + 누수 필드 목록 + 함정 주석. 먼저 읽어라
  - .env               KRA_API_KEY_ENCODED
- 팀 공유 레포: C:\Users\SSAFY\Desktop\말고리즘\S15P21A304  (GitLab, 브랜치 develop)
  - docs/dataset/      parquet + 빌드 코드 사본 + README (규약·한계 전부 여기)
  - docs/eda/          팀원 EDA — 고쳐야 할 7가지
  - docs/model/        common.py, baseline_lgbm.py — 팀 공통 평가 모듈
  - docs/model/result_dl.md  팀원 딥러닝 결과

시작 전에 kra_client.py 주석, docs/dataset/README.md, docs/eda/README.md 를 읽어라.

## 1단계 — 아직 미신청인 API 2종만 신청·확인

2026-09-07 실측 기준이다. 대부분 이미 열려 있다.

열린 것 (호출 가능):
  API4_3/raceResult_3           경주기록 원장
  API26_2/entrySheet_2          출전표 (발주시각·출주두수)
  API72_2/racePlan_2            경주계획표
  API156/raceRsutDtl            ★ AI연구용 상세 — 아래 3단계에서 쓴다
  API9_1/raceHorseCancelInfo_1  출전취소
  API10_1/jockeyChangeInfo_1    기수변경
  trcontihi/gettrcontihi        조교
  API191_1/HorseShoe_1          장제
  API42_1/totalHorseInfo_1      마필종합(혈통 3대)
  racedetailresult/...          경주별상세성적표
  + kra_client.ENDPOINTS 참조

아직 미신청 (403 SERVICE_KEY_IS_NOT_REGISTERED_ERROR):
  API320/textDataHoldSeChgInfo  서울 기수변경·말취소
  API154/racePlan               AI학습용 경주계획

★ API317/textDataHoldSeWegInfo (서울 당일 마체중) 은 **신청은 반영됐는데 어떤
  파라미터를 줘도 totalCount=0 이다.** 빈손({}, meet, rc_date, rc_month, stdate 전부 0).
  홈페이지 자료실 계열이라 당주 개최분만 있을 가능성이 크다.
  **경마 시행일(금·토·일)에 다시 호출해서 확인해줘.** 이게 되면 마체중을 발주 전에
  쓸 수 있어서 새 피처가 생긴다 — 지금 wgHr 는 경주가 끝나야 들어온다.

## 2단계 — 2004~2009년 원장 백필

현재 데이터셋은 2010~2026. API 는 2000년부터 있고 2004년부터 품질이 동등하다(실측).
2004~2009 = 158,532행, train 385,514행 대비 +41%.

    cd C:\Users\SSAFY\Desktop\주제선정
    python tools/backfill_ledger.py 2004 2009

- data/raw/ledger_2010_2026.csv 에 append 된다 (파일명 그대로 둬도 build 가 읽는다)
- data/raw/ledger_progress.json 으로 중단 시 이어받기
- 약 216콜 / 30분. 개발계정 한도 3,000콜/일

★ 2000~2003 은 하지 마라. 구간기록이 2000년 0%, 2002년 8%, 2004년 98% 다.
  2004년이 경계선이라 그 전은 F5_* 가 전부 결측이 된다.

## 3단계 — API156 으로 2004~ 구간 보강 (선택이지만 권장)

API156/raceRsutDtl 은 44필드이고 **2004년부터 전 구간이 조회된다**(실측).
원장(API4_3)에 없는 필드가 있고, 옛 구간에서도 충전율이 높다:

  필드                  2004  2010  2018  2026   설명
  pthrLatstPtinDt        94%   96%   91%   95%   직전 출전일 → 휴양일수 교차검증
  hrmOwnerNm            100%  100%  100%  100%   마주
  pthrBthd              100%  100%  100%  100%   생년월일 → 일 단위 월령(age 보다 정밀)
  rsutRlStrtTim         100%  100%  100%  100%   실제 발주시각
  pthrEquip               0%    0%   85%   92%   장구 — 2018년부터만

⚠⚠ **파라미터명이 race_dt 다.** rc_date / rc_month / rc_year 를 주면 **에러 없이
   조용히 무시되고 최신 경주일 데이터가 돌아온다.** 과거 데이터인 줄 알고 쓰면
   전량 오염된다. 실제로 걸린 함정이니 반드시 race_dt + rccrs_cd 로 호출해라.

✔ pthrLatstPtinDt 는 누수 없음이 확인됐다(같은 말을 여러 시점에 조회했을 때
  값이 각각 그 시점의 직전 출전일과 일치, 20두 확인). 스냅샷이 아니다.

## 4단계 — 재빌드

    cp -r dataset/v2 dataset/v2_before_2004      # ★ A/B 비교용 백업. 먼저 해라
    python build_v2.py all
    python validate_v2.py

build_v2.py 에 연도 필터가 없어서 원장에 붙이면 자동 반영된다.
validate_v2.py 가 전 항목 통과·경고 0건이어야 한다.

## 5단계 — A/B 테스트 (이게 핵심)

**행이 늘었다고 좋아진다고 단정하면 안 된다.** 근거:

- EDA 실측: 시장 1등 적중률이 2010~2015 39~42% → 2023~2026 35~37% 로 변했다.
  20년간 경기 양상이 바뀌었다는 뜻이다.
- valid/test 는 2025~2026 이다. train 을 더 옛날로 늘리면 분포 이동(PSI)이 악화될 수 있다.
  EDA 가 이미 PSI 심각 11개를 셌다.
- 2004~2009 행은 아래가 전부 결측이다 — 즉 추가되는 건 "빈약한 행"이다:
    X_rating    레이팅이 2014~2015년경부터 시작 (2013년 0%, 2015년 61%)
                ※ 지금 train 의 2010~2013 구간도 rating 이 비어 있다. README §8 에 없는 사실
    F1_tr_*     조교 aux_train.csv 가 2010년부터
    F2_ebv_*    육종가 스냅샷이 2016-07부터

측정 방법:

1. 팀 레포 docs/model/common.py + baseline_lgbm.py 를 그대로 쓴다 (팀 공통 기준)
2. 2010~ 빌드 vs 2004~ 빌드를 valid 1,254경주로 비교
3. **시드를 반드시 3개 이상** 돌려 평균과 표준편차를 같이 본다.
   같은 구성도 시드만 바꾸면 top-1 이 ±1%p 움직인다. 시드 1개로 비교하면
   없는 차이를 있다고 읽게 된다 (실제로 그렇게 잘못 읽은 적이 있다)
4. 인기도 포함(77피처)과 인기도 제외(73피처) 둘 다 잰다
5. 차이가 표준오차(±0.85%p)의 2배를 못 넘으면 "효과 없음"으로 읽는다

같이 재볼 것: 2004~2009 를 넣되 **최근 데이터에 가중치를 주는 것**
(LightGBM sample_weight, 예: 경주일 기준 지수 감쇠)이 그냥 넣는 것보다 나은지.
분포 이동이 원인이라면 이쪽이 답일 수 있다.

## 6단계 — 결과 반영

- 좋아졌으면: docs/dataset/ 에 새 parquet 복사 + README §1 행수·기간 갱신 + §5-2 에 결과 추가
- 안 좋아졌으면: 되돌리고 README §8 에 "2004~2009 시도, 효과 없음(수치)" 로 기록
  → 다음 사람이 같은 실험을 반복하지 않게

어느 쪽이든 **별도 브랜치 + MR** 로 올린다. develop 직접 푸시 금지.
push 할 때 이렇게 하면 MR 이 자동 생성된다:

    git push -o merge_request.create -o merge_request.target=develop \
             -o merge_request.title="..." -u origin <브랜치명>

## 함정 (전부 실측으로 확인된 것)

1. **API156 은 race_dt 파라미터다.** rc_date 를 주면 조용히 최신 데이터가 온다 (위 3단계)
2. **참고문서(docx)의 엔드포인트가 폐기된 게 있다.** 정답은 openapi.do 에 인라인된
   Swagger JSON. 단 Swagger 의 host 는 API 번호가 잘려 있는 경우가 있다
   (예: B551015/raceResult → 실제는 B551015/API155/raceResult)
3. **에러코드로 원인이 갈린다.** 403 SERVICE_KEY_IS_NOT_REGISTERED = 신청 안 됨,
   400 NO_OPENAPI_SERVICE = 경로가 틀림
4. **통산전적 API 는 전부 "오늘 기준 스냅샷"이다.** 과거 경주에 붙이면 미래가 샌다.
   같은 말의 rcCntT 가 2025-01/2025-07/2026-07 조회에서 전부 36 으로 동일했다.
   목록은 kra_client.LEAKY_FIELDS / LEAKY_TOTAL_HORSE
5. **배당 9999.9 는 "배당 없음" 특수값.** 안 거르면 수익률이 +250% 로 나온다(실제 -21.4%)
6. **schema_v2.clean() 을 반드시 거쳐라.** X_grade 표기 통합(41→26레벨), F2_sire_id
   결측 토큰 통일, 배당 특수값 제거. common.load() 가 이미 호출한다
7. **정렬을 바꾸지 마라.** parquet 이 race_id 로 묶여 있고 LightGBM group 이 그 순서를
   가정한다. 셔플하면 에러 없이 조용히 틀린 모델이 나온다
8. **game 은 학습·검증·평가 어디에도 쓰지 마라.** 사용자가 플레이할 경주다
9. **test 는 열지 마라.** 튜닝·비교는 valid 로

## 현재 성능 기준선 (valid 1,254경주, 시드 3개 평균)

  시장 (인기 1위마)          39.47%
  인기도만 4피처             39.10%
  LightGBM 77피처           38.12%
  LightGBM 73피처 (실시간)   32.11%

아직 시장을 못 이긴다. 인기도 4개만 쓴 모델이 77개 쓴 모델보다 낫다 —
펀더멘털을 얹으면 배당 정보가 희석된다는 뜻이라, 방향은 Benter 2단계 결합이다.
팀원 딥러닝 결과(docs/model/result_dl.md)도 같은 방향을 가리킨다:
top-1 로는 LightGBM 을 못 넘었지만 S3(말별 GRU)가 확률품질(logloss)에서 일관되게 이겼다.
```

---

## 이 세션에서 확인한 근거

| 항목 | 수치 |
|---|---|
| 2000~2009 미사용 행 | 241,615 (train 385,514 대비 +62%) |
| **2004~2009 (권장 구간)** | **158,532 (+41%)** |
| 구간기록 시작 | 2004년 (2000년 0%, 2002년 8%, 2004년 98%) |
| rating 시작 | 2014~2015년경 (2013년 0%, 2015년 61%) |
| 조교(aux_train) 시작 | 2010년 |
| 육종가(ebv) 시작 | 2016-07 |
| 현재 원장 | 2010~2026, 456,210행 |
| API156 커버리지 | **2004년부터 전 구간** (실측: 2004/2006/2008/2010/2012/2015/2018/2024/2026 전부 응답) |
