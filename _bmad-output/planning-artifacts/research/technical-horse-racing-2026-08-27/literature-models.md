---
title: '경마 예측 모델 — 선행연구 정리'
type: 'academic-lit'
topic: '경마 착순/승자 예측에 사용된 모델과 보고 성능'
decision: 'BASE 모델로 무엇을 학습시킬 것인가'
status: complete
created: '2026-08-31'
---

# 경마 예측 모델 — 선행연구 정리

**조사 범위:** 국내 KRA 데이터 논문 3편 + 해외 9편(체계적 문헌고찰 §4.7 기준) + Benter 원전
**핵심 2차 출처:** Galekwa et al., *A Systematic Review of Machine Learning in Sports Betting*, arXiv:2410.21484 — §4.7 Horse Racing 및 Table 9

---

## 0. 결론 먼저

1. **모델 트렌드가 두 번 바뀌었다.** ANN·SVM(2010~) → RF·GBDT(2018~) → **Learning-to-Rank(2024~)**. 국내 최신 2편이 모두 LTR이다.
2. **보고된 정확도 90%+는 믿으면 안 된다.** 클래스 불균형 아티팩트다(§4). 신뢰할 수 있는 숫자는 **단승 적중률 40%대** 또는 **ROI**다.
3. **모델 종류보다 "무엇을 예측 단위로 두느냐"가 성능을 가른다.** 말 단위 이진분류 < 경주 단위 상대비교(랭킹/조건부로짓).
4. **선행연구가 쓴 피처 중 우리가 아직 안 본 것이 있다** — 출발훈련(게이트) 횟수, 질병 진단, 과거 획득 상금, 맞대결 그래프(§5).

---

## 1. 국내 — KRA 데이터 기반

| # | 논문 | 데이터 | 모델 | 결과 |
|---|---|---|---|---|
| K1 | 최혜민·황나영·황찬경·송종우 (2015)<br>**서울 경마 경기 우승마 예측 모형 연구**<br>응용통계연구 28(6):1133-1146, 이화여대 | KRA 성적표·경주마·기수·조교사 | 선형회귀 · **랜덤포레스트** · 로지스틱회귀 | 세 베팅 방식 모두 **양의 수익**. 핵심 변수 = 말 기본정보 + **과거 우승 경력**, 기수 과거 우승 경력 |
| K2 | 정준형·신동욱·황세용·박건웅 (2024)<br>**Horse race rank prediction using learning-to-rank approaches**<br>응용통계연구 37(2):239-253, 서울대 통계학과 | 경주정보 + 기수정보 + **말 훈련기록** + 조교사정보 통합 | **point-wise**(선형회귀, RF) vs **pair-wise**(RankNet, LambdaMART = XGBoost/LightGBM/**CatBoost** Ranker) | **pair-wise > point-wise.** CatBoost Ranker 최고. Shapley 상위 변수에 **출발훈련 횟수·총 출발훈련·질병 진단** 포함 |
| K3 | So Yubin·Woo Eunbi·Lee Hanjun (2025-11)<br>**LTR 기법을 활용한 경마 예측 및 웹 서비스 개발**<br>한국컴퓨터정보학회논문지 30(11):311-318 | KRA 2024.5~2025.4, **9,140경주** | LightGBM · XGBoost · CatBoost + **Listwise LambdaRank** | CatBoost 랭킹 품질 최고(**NDCG 0.8895**, MAP 0.4204). 그러나 **실제 베팅 시나리오에서는 LightGBM·XGBoost가 더 정확**. 피처 4계열(최근 성적, 통산 평균 착순, 부담중량, 마령) |

> **K2와 K3의 결론이 엇갈린다** — K2는 CatBoost Ranker 최고, K3은 랭킹 지표는 CatBoost지만 베팅은 LightGBM/XGBoost.
> 이건 모순이 아니라 **평가 지표가 다르면 승자가 바뀐다**는 증거다. 우리 평가축을 먼저 정해야 한다.

### 인접 종목 (경륜) — 참고
- 머신러닝 적용 경륜 경주 순위/베팅방식별 예측 (2023): 로지스틱회귀·RF·AdaBoost·GradientBoost·LightGBM·MLP·XGBoost 7종 → **로지스틱회귀 최고**
- 경륜 순위 예측 (2016~2022 출주표): 로지스틱회귀 단승 88.19% / 복승 80.07%, 삼복승 AdaBoost 78.17%
  → **이 88%는 §4의 불균형 함정으로 보인다. 그대로 인용하면 안 된다.**

---

## 2. 해외 — 모델별 정리

| # | 연구 | 데이터 | 모델 | 보고 성능 |
|---|---|---|---|---|
| F1 | **Benter (1994)**<br>*Computer Based Horse Race Handicapping and Wagering Systems* | 홍콩, 5년 실전 운용 | **조건부로짓 2단계** — ①펀더멘털 모델 ②공중 배당률(내재확률)과 결합. 말당 **120+ 변수** | **5년간 실전 수익 실증.** 이 분야의 원전 |
| F2 | Davoodi & Khanteymoori (2010) | AQUEDUCT(뉴욕) **100경주**, 8피처 | ANN 5종 (BP, BPM, Quasi-Newton, Levenberg-Marquardt, CGD) | BP 77%(1착), LM 최속, CGD는 꼴찌 예측 우수. **표본 100경주 — 과적합 의심** |
| F3 | **Lessmann et al. (2010)** | 홍콩 **1,000경주 / 12,902두** (2005.1~2006.12) | **경쟁구조를 반영한 RF** vs 조건부로짓(CL) vs SVM | **RF 수익률 20.26% vs CL 8.84%** (500경주). NDCG + Kelly 베팅 |
| F4 | Pudaruth et al. (2013) | 모리셔스 Champ de Mars, 2010시즌 | 가중 확률 방식 (기수·경험·배당·과거성적·게이트·거리적성·중량·레이팅·마방) | **58%** (전문 예상가 44%) |
| F5 | Selvaraj (2017) | Sporting Life 스크래핑 + Kaggle | k-NN, LDA, 로지스틱회귀 + 정보이득비 피처선택 | 데이터셋1 **33.97% / 33.33%** · 데이터셋2 **91.40% / 92% / 89.58%** ← §4 참조 |
| F6 | Gulum (2018) | UK Equibase 2015~2017 | ANN + 로지스틱회귀, **그래프 기반 피처**(승패 스프레드, 노드 점수 — 말 간 맞대결 방향그래프) | 그래프 피처 추가 모델이 기본 피처 모델을 상회 |
| F7 | **Borowski et al. (2021)** | 폴란드 **3,782경주** (2011~2020, 아라비아·서러브레드) | CART, Glmnet, **XGBoost**, RF, **NN**, **LDA** | **LDA·NN 최고 — 단승 적중 41%, 복승 36%+.** 최중요 피처 = **과거 획득 상금**. 기수·조교사는 상대적으로 덜 중요 |
| F8 | Terawong & Cliff (2024) | Bristol Betting Exchange ABM **합성 데이터** | XGBoost + 에이전트 기반 모델 | 원 에이전트 대비 수익성 우위. **합성 데이터라 실전 일반화는 별개** |
| F9 | Gupta & Singh (2024) | 인도 **14,750행 / 56속성** | k-NN, 선형회귀, **RF**, GNB, AdaBoost, Bagging + 정보이득·카이제곱 선택 | **RF 93.1%** ← §4 참조 |
| F10 | Tondapu (2024) | Betfair UK **1,056,766 가격변동 신호** | 시계열 분석 (KPSS, ADF) | **경마 베팅시장은 전통 금융시장보다도 효율적.** 자기상관 급감, 장기기억 없음 |

---

## 3. 모델 계보 — 무엇이 언제 쓰였나

```
2010 ─── ANN(BP/LM/CGD) ─── SVM ─── 조건부로짓(Benter 계열)
2013 ─── 가중 확률 휴리스틱
2017 ─── k-NN / LDA / 로지스틱회귀 + 피처선택
2018 ─── ANN + 그래프 피처
2021 ─── CART / Glmnet / XGBoost / RF / NN / LDA 비교
2024 ─── GBDT + 에이전트모델 · RF + 피처선택
2024 ─── ★ Learning-to-Rank (pair-wise: RankNet, LambdaMART)      ← 국내 K2
2025 ─── ★ Learning-to-Rank (list-wise: LambdaRank) + 웹서비스      ← 국내 K3
```

**요약:** 단일 모델 승부에서 → **경주 단위 상대비교(LTR)**로 프레이밍 자체가 이동했다. 우리가 §2단계로 잡은 `LGBMRanker(lambdarank)`가 현재 최신 흐름 위에 있다.

---

## 4. ⚠ 가장 중요한 경고 — 90%대 정확도를 인용하지 말 것

Selvaraj(F5) 91~92%, Gupta & Singh(F9) 93.1%, 국내 경륜 88.19% — **이 숫자들은 승자 예측 성능이 아니다.**

**이유:** 10두 경주에서 말별 "1착 여부"를 이진분류하면 양성:음성 = 1:9다.
**전부 "아님"이라고 찍기만 해도 정확도 90%**가 나온다. 모델이 아무것도 학습하지 않아도 그렇다.

**근거 3가지:**
1. **우리 실측 시장 베이스라인이 38.3%다** (2025년 KRA 2,478경주, 인기 1위마 1착 적중률). 모델이 93%라면 세계 경마 시장은 진작 붕괴했다.
2. **Selvaraj 논문 안에서 데이터셋1은 33%, 데이터셋2는 91%**로 갈린다. 같은 모델·같은 저자다. 33%는 경주당 top-1 정확도로, 91%는 말별 이진 정확도로 읽으면 정확히 설명된다.
3. **Tondapu(F10)**가 100만 건 이상으로 경마 시장이 **전통 금융시장보다 효율적**임을 보였다. 90%대 예측이 가능한 시장이 아니다.

**신뢰할 수 있는 성능 보고 형태:**

| 형태 | 예 | 왜 신뢰되나 |
|---|---|---|
| 경주 단위 **단승 적중률** | Borowski 41%, Pudaruth 58% | 분모가 경주라 불균형이 안 생김 |
| **ROI / 수익률** | Lessmann **RF 20.26% vs CL 8.84%** | 베팅 비용까지 반영 |
| 시장 대비 상대 개선 | Pudaruth 58% vs 전문가 44% | 기준선이 명시됨 |
| 경주 단위 **로그로스 / NDCG** | K3 NDCG 0.8895 | 확률 품질 지표 |

**우리 발표의 목표 수치:** 시장 38.3%를 유의미하게 상회하는 **40%대 초반**이 현실적 상한이다. Borowski가 폴란드에서 41%를 낸 것과 정합한다. **93%를 목표로 잡으면 그건 버그를 만든 것이다.**

---

## 5. 선행연구가 쓴 피처 중 우리가 아직 안 본 것

| 피처 | 출처 | 우리 확보 가능성 |
|---|---|---|
| **출발훈련(게이트) 횟수 / 총 출발훈련** | K2 (Shapley 상위 10) | ✅ `출발훈련 15059043` API 존재 — **미신청** |
| **질병 진단 이력** | K2 (Shapley 상위 10) | ⚠ 별도 확인 필요. 훈련·검진 계열 API 조사 대상 |
| **일별 조교(훈련) 기록** | K2 | ✅ `일별훈련 상세 15058782` — **미신청** |
| **과거 획득 상금** | F7 (최중요 변수) | ✅ 이미 확보 — `chaksun1~5`. rolling 누적으로 파생 |
| **맞대결 그래프 피처**(승패 스프레드, 노드 점수) | F6 | ✅ 원장에서 직접 계산 가능. **차별화 아이디어** |
| **거리 적성 / 마방(조교사) 평판** | F4 | ✅ 원장 rolling 집계로 파생 |
| **공중 배당률 내재확률** | F1 Benter 2단계 | ✅ 확보. 단 **확정치만** → 발주 전 입력 불가, 베이스라인·사후분석용 |

> **가장 실행 가능한 차별화 2가지:**
> ① **훈련(조교) 데이터 결합** — K2가 Shapley 상위에 올린 변수인데 K3(최신 LTR 논문)는 안 썼다. API도 있다.
> ② **맞대결 그래프 피처** — F6이 효과를 보였고, 국내 논문 중 쓴 곳이 없다.

---

## 6. 우리 모델 선택에 주는 함의

| 판단 | 근거 |
|---|---|
| **경주 단위 LTR을 주력으로** | K2·K3 모두 LTR, K2는 pair-wise > point-wise를 명시적으로 검증 |
| **LightGBM/XGBoost를 CatBoost보다 먼저** | K3이 "랭킹 지표는 CatBoost, **베팅 정확도는 LightGBM/XGBoost**"를 보고. 우리 목표는 베팅 성과 |
| **RF도 비교군에 넣을 가치 있음** | F3에서 RF가 조건부로짓을 ROI로 2배 이상 앞섬(20.26% vs 8.84%). K1·F9도 RF 채택 |
| **조건부로짓(Benter)은 차별화 카드** | F1이 원전이고 5년 실전 수익 실증. 2단계 결합은 국내 논문 중 아무도 안 했다 |
| **신경망은 후순위** | F2(100경주)·F7(NN 상위) 모두 소규모. GBDT 대비 이점이 뚜렷하지 않고 7주 안에 리스크 |
| **평가 지표를 먼저 고정할 것** | K2/K3의 승자가 지표에 따라 갈렸다. 적중률·ROI·로그로스를 동시에 보고할 것 |

---

## 7. 출처

| 구분 | 출처 |
|---|---|
| 체계적 문헌고찰 (F2~F10 근거) | [Galekwa et al., arXiv:2410.21484](https://arxiv.org/pdf/2410.21484) §4.7 + Table 9 |
| K1 | [KCI ART002068008](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002068008) |
| K2 | [Korea Science JAKO202414143309228](https://koreascience.kr/article/JAKO202414143309228.page) |
| K3 | [KCI ART003266151](https://journal.kci.go.kr/jksci/archive/articleView?artiId=ART003266151) |
| F1 Benter 원전 | [gwern.net 1994-benter.pdf](https://gwern.net/doc/statistics/decision/1994-benter.pdf) |
| 경륜 (인접) | [KCI ART002971216](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002971216) · [ART002954741](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002954741) |
| 시장 베이스라인 38.3% | 자체 실측 (KRA `API4_3/raceResult_3`, 2025년 2,478경주) |

**주의:** "700,000경주 × 117피처 CatBoost" 자료가 검색에 잡히나, 이는 **Medium 블로그 글(Çağdaş Gül, 2025-11)이며 동료심사 논문이 아니다.** 인용 시 출처 성격을 명시할 것.
