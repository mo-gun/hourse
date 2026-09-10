# 인계 — 베이스 모델(6축 딥러닝 랭커) 작업 이어받기

**작성 2026-09-10, 맥북에서 작업한 것을 SSAFY 노트북으로 넘긴다.**
데이터 백필 작업 인계는 [HANDOFF.md](HANDOFF.md) 를 보라 — 이 문서는 **모델** 쪽이다.

---

## 0. 결론부터 — 무엇이 생겼나

기존 HANDOFF.md 마지막 줄이 이렇게 끝나 있었다.

> 인기도 4개만 쓴 모델이 77개 쓴 모델보다 낫다 — 펀더멘털을 얹으면 배당 정보가
> 희석된다는 뜻이라, **방향은 Benter 2단계 결합이다.**

**그 방향으로 만든 것이 이번 작업이다.** 배당을 6축 중 하나로 넣어 1/6 로 희석시키는 대신,
**7번째 타워로 빼내 프리셋마다 고정 계수 `m` 으로** 얹었다.

```
점수 = m · s_시장 + Σ w_k · s_k        Σ w_k = 100 (유저 슬라이더), m = 프리셋 상수
```

| | top-1 | top-3 | logloss |
|---|---:|---:|---:|
| 시장 (인기 1위마) | 39.47 | 69.46 | 1.7504 |
| LightGBM 77 | 39.07 | 68.58 | 1.7387 |
| 정원님 S5 타워 균등가중 (배당을 1/6 로) | 35.33 | 65.63 | 1.8656 |
| **배당형** (m=+1.86) | **37.88** | **68.82** | **1.7995** |
| 기본형 (m=0, 배당 안 봄) | 33.33 | 63.00 | 1.9480 |

- 배당형 − 타워 **+2.55%p [+0.48, +4.47] p=0.014** → 유의
- 배당형 − 시장 −1.59%p [−3.63, +0.36] p=0.29 → **차이 없음**
- 기본형 − 시장 −6.14%p p<0.001 → 유의하게 열세

**시장은 못 이긴다. 다만 배당을 쓰면 대등하고, 시장에 없는 정보는 갖고 있다**
(조건부 로짓 우도비 LR 19.11 p=1.2e−05, 무작위 대조군은 p=0.86 으로 걸러짐).
결합해도 적중률은 안 오르고 logloss 만 유의하게 좋아진다(1.7504 → 1.7412).

---

## 1. 코드가 어디 있나

**정원님 GitHub 레포의 브랜치에 있다. 이 레포에는 코드를 복사해 두지 않았다** —
두 곳에 두면 갈라진다.

- 레포: https://github.com/JeongWon4034/horse-pred-engine
- 브랜치: `feat/basemodel-axis-ranker`
- PR: https://github.com/JeongWon4034/horse-pred-engine/pull/13 (리뷰 대기)
- 폴더: `basemodel/` (최상위, `pipeline/` 과 나란히). **정원님 코드는 한 줄도 안 건드렸다**

---

## 2. SSAFY 노트북 세팅 (git bash 기준)

```bash
# 1) 코드
cd /c/Users/SSAFY/Desktop/말고리즘          # 팀 레포가 있는 폴더 옆에
git clone https://github.com/JeongWon4034/horse-pred-engine.git
cd horse-pred-engine
git checkout feat/basemodel-axis-ranker

# 2) 데이터 — 팀 레포에서 가져온다 (pipeline/data/, gitignore, 65MB)
cd pipeline
TEAM_REPO=/c/Users/SSAFY/Desktop/말고리즘/S15P21A304 bash sync_dataset.sh
cd ..

# 3) 파이썬 환경 (uv)
cd basemodel
uv sync

# 4) 확인 — 시장 39.3/69.3 이 나오면 정상
PYTHONUTF8=1 uv run python -c "import sys;sys.path.insert(0,'.');from basemodel.team import C;print(C.report({},C.load('valid')))"
```

**Windows 주의 3가지**

| | |
|---|---|
| `PYTHONUTF8=1` | **모든 명령 앞에 붙여라.** 안 붙이면 cp949 에서 한글 출력이 깨진다. 정원님 장부에도 같은 주의가 있다 |
| torch | `pyproject.toml` 은 PyPI 기본 휠이다(맥 MPS 기준). **NVIDIA GPU 가 있으면** `pipeline/pyproject.toml` 처럼 cu130 인덱스를 걸어라 — 주석에 적어 뒀다 |
| 학습 시간 | 맥 M4 에서 모델 1개 2분, `run_all.sh` 전체 25분. **노트북 CPU 면 3~4배** 잡아라. GPU 있으면 훨씬 빠르다 |

### 돌려보기

```bash
cd basemodel
PYTHONUTF8=1 uv run python -m basemodel.train --epochs 30 --tag _final              # 77피처
PYTHONUTF8=1 uv run python -m basemodel.train --no-market --epochs 30 --tag _final  # 73피처
PYTHONUTF8=1 uv run python -m basemodel.export --ckpt artifacts/runs/axis_77_s20260901_final.pt
PYTHONUTF8=1 uv run python -m basemodel.significance --ckpt artifacts/runs/axis_77_s20260901_final.pt
bash run_all.sh    # 위 전부 + 시드 5개 + 분해표 + 성능 + 허깅페이스 폴더
```

읽을 것은 `basemodel/README.md`(설계·결과)와 `basemodel/ledger.md`(실험 장부 + **실수 기록 7건**).

---

## 3. 다음 할 일 — 우선순위

### ① 마체중(`X_wgHr`) 을 스키마에 등록 ★ 이 노트북에서만 할 수 있다

**원장(`data/raw/ledger_2010_2026.csv`, 199MB)이 이 노트북에만 있어서** 맥에서는 못 했다.

- `build_v2.py:103` 이 `X_wgHr`·`X_wgHr_delta` 를 **계산은 하는데**,
  `schema_v2.py` 의 `X` 블록(10개)에 **등록이 안 돼 있어 학습셋에서 빠진다.**
  스키마 주석이 경고하는 바로 그 실수다 — "features() 가 반환하는 것만 살아남는다"
- 게임풀에서 재본 값: **우리 모델 위에 유의**(LR 15.79, p=7.1e−05), **시장 위에는 없음**(p=0.44).
  적중률 개선은 확인 안 됨(−0.26%p [−0.88, +0.36]) → **기대치를 낮게 잡아라**

⚠ **tier 를 반드시 확인하고 넣어라.** `schema_v2.TIER` 는 마체중을 `B`(발주 T-60~90분)로
가정하는데, HANDOFF.md 에 적힌 대로 **API317(당일 마체중)이 아직 totalCount=0** 이다.
즉 실제로는 **경주가 끝나야 들어온다.** 그대로 tier=A/B 로 넣으면 **주말 실시간 모델
(73피처)에 미래 정보가 샌다.**
→ API317 이 경마 시행일에 되는지 먼저 확인하고, 안 되면 **tier=G**(리플레이 전용)로 넣어라.
그러면 `features(exclude_tier=("G",))` 가 실시간 모델에서 자동으로 뺀다.

### ② "비 오면 가벼운 말" 가설 — 이미 기각됐다. 반복하지 마라

게임풀 3,063경주 실측. **모든 주로 상태에서 무거운 말이 더 이긴다.**

| 주로 | 가벼운쪽 승률 | 무거운쪽 승률 | 차이 |
|---|---:|---:|---:|
| 건조·양호 | 6.52% | 11.55% | −5.03%p |
| 다습·포화·불량 | 7.10% | 11.06% | −3.96%p |

젖으면 격차가 줄긴 하지만 차이의 차이 **+1.07%p, CI [−0.62, +2.76] → 유의하지 않다.**

다만 **모델이 상호작용을 배우는 것 자체는 확인됐다.** 입력의 선행마 수만 1두→5두로 바꾸면
추입 +0.0285 · 중단 +0.0126 · 선입 −0.0120 · **선행 −0.0359** 로 움직인다.
기획서의 "선행마 몰리면 오버페이스 → 추입 유리"를 손으로 안 넣었는데 스스로 배웠다.

### ③ 결합식 확정 — 기획·도연님과

정원님 `exp/tower-hist` §6-3 은 `점수 = 베이스(S3) + λ·Σw·s` 를 제안하고,
이 작업은 `점수 = m·s_시장 + Σw·s` 를 택했다. **진단은 같고 처방이 다르다.**
둘 다 넣을 수도 있다. 도연님이 별도 6축 파이프라인(`ai/model/`)을 갖고 있어
소유 범위도 같이 정해야 한다(정원님 §6-4).

### ④ 그 외

- **실력 축 상한** — 77피처에서 단독 21.0% 로 합격선(25) 미달. EBV 결측(2016 이전 0%)이 원인.
  `feature_group.max_weight` 로 막을 것을 BE 에 제안해 둠
- **이력 길이 L** — 정원님 L 곡선상 L=5 로 줄여도 같을 가능성이 높은데 지금 L=20. 미측정
- **test 미개봉** · **실경기 사전 예측 미실시**
- **표본** — 2%p 차이를 잡으려면 주말 13주치가 더 필요하다(검출 한계 top-1 ±2.39%p)

---

## 4. 주의 — 이 작업에서 실제로 밟은 지뢰

전문은 `basemodel/ledger.md` §실수 기록. 특히 셋은 다시 만날 수 있다.

1. **시장 축을 학습할 때 계수를 음수까지 샘플링하면 타워가 망가진다.** "1착마 점수를
   올려라"와 "내려라"가 상쇄되어 배울 방향이 없어진다. top-1 이 5.7%(무작위 10.2% 미만)까지
   떨어졌다. 부호 뒤집기는 **프리셋의 몫**이지 학습의 몫이 아니다
2. **시장을 손실에 넣으면 나머지 축이 무임승차한다.** 실력 축 단독이 8.0% 까지 떨어졌다.
   손실을 둘로(`PL(m=0) + PL(m~U)`) 나눠야 회복된다
3. **시장 확률을 점수로 그냥 쓰면 안 된다.** softmax 가 한 번 더 걸려 거의 균등이 된다.
   그렇게 재면 시장 logloss 가 2.21(무작위 2.38 근처)로 나온다. `log(확률)` 을 써라

---

## 5. 새 Claude 세션에 붙여넣을 것

아래 블록을 통째로 복붙하면 된다.

```
경마 예측 프로젝트 모델 작업을 이어받는다. 맥북에서 하던 걸 이 노트북으로 옮겼다.

## 지금 상태

베이스 모델(6축 딥러닝 랭커)을 만들어 정원님 GitHub 에 PR 을 올려둔 상태다.
  레포     https://github.com/JeongWon4034/horse-pred-engine
  브랜치   feat/basemodel-axis-ranker
  PR       #13 (리뷰 대기)
  폴더     basemodel/  ← 여기가 내 작업. pipeline/ 은 정원님 것이니 건드리지 마라

## 먼저 읽어라

1. basemodel/README.md   설계와 결과 전부
2. basemodel/ledger.md   실험 장부 + 실수 기록 7건 ★ 같은 실수 반복 방지
3. (이 레포) HANDOFF-basemodel.md   이 문서
4. (이 레포) HANDOFF.md   데이터 백필 쪽 인계

## 환경

- 이 노트북 개인 폴더: C:\Users\SSAFY\Desktop\주제선정
    data/raw/ledger_2010_2026.csv  199MB ★ 이 노트북에만 있다
- 팀 레포: C:\Users\SSAFY\Desktop\말고리즘\S15P21A304 (GitLab, develop)
- 모델 레포는 위 GitHub. 데이터는 pipeline/sync_dataset.sh 로 팀 레포에서 가져온다

세팅:
    cd /c/Users/SSAFY/Desktop/말고리즘
    git clone https://github.com/JeongWon4034/horse-pred-engine.git
    cd horse-pred-engine && git checkout feat/basemodel-axis-ranker
    cd pipeline && TEAM_REPO=/c/Users/SSAFY/Desktop/말고리즘/S15P21A304 bash sync_dataset.sh
    cd ../basemodel && uv sync

★ 모든 python 명령 앞에 PYTHONUTF8=1 을 붙여라. 안 붙이면 한글 출력이 깨진다.
★ NVIDIA GPU 가 있으면 basemodel/pyproject.toml 주석대로 cu130 인덱스를 걸어라.

## 첫 작업 — 마체중(X_wgHr) 스키마 등록

build_v2.py:103 이 X_wgHr·X_wgHr_delta 를 계산하는데 schema_v2.py 의 X 블록에
등록이 안 돼 학습셋에서 빠져 있다. 원장이 이 노트북에만 있어 여기서만 할 수 있다.

먼저 확인할 것:
  API317/textDataHoldSeWegInfo (서울 당일 마체중) 이 경마 시행일에 응답하는지.
  - 응답하면 → 발주 전에 알 수 있으므로 tier=B 로 등록
  - 안 하면  → 경주 후에만 아는 값이므로 반드시 tier=G (리플레이 전용).
              tier=A/B 로 넣으면 주말 실시간 모델에 미래 정보가 샌다

등록 후:
  1. python build_v2.py all → validate_v2.py 전 항목 통과 확인
  2. 팀 docs/dataset 에 반영 (별도 브랜치 + MR, develop 직접 푸시 금지)
  3. basemodel 재학습 후 significance.py 로 시장 대비 재측정

기대치는 낮게 잡아라 — 게임풀에서 미리 재봤더니 우리 모델 위에는 유의하지만
(LR 15.79, p=7.1e-05) 적중률 개선은 확인 안 됐다(-0.26%p, CI [-0.88, +0.36]).

## 규칙

- game 은 학습·검증·튜닝 금지. test 는 열지 않는다. 비교는 valid
- 정렬(race_id 연속 블록)을 바꾸지 마라
- 시드 3개 이상 돌려 평균±표준편차로 읽어라. top-1 검출 한계가 ±2.39%p 다
- 팀 채점기(common.py)를 거치지 않고 데이터를 직접 읽지 마라
- 정원님 pipeline/ 코드는 수정하지 마라. 필요하면 basemodel/ 안에서 해라
```
