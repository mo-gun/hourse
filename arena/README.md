# arena/ — 기법 계열 비교장

> 같은 데이터·같은 자로 **모델 계열(family)을 바꿔 가며** 한 표에 올린다.
> 묻는 것 하나: **"기법을 바꾸면 73피처 top-1 의 선(35.19)을 넘을 수 있나."**

담당: 이건모 · 데이터 `pipeline/data` (팀 schema v2.2.0) · 채점기 `common.py`

---

## 0. 이 폴더가 왜 따로 있나

| | `pipeline/` (정원) | `basemodel/` (건모) | **`arena/` (건모)** |
|---|---|---|---|
| 묻는 것 | 어떤 **구조**가 기여하는가 | 그 구조를 **제품 계약**에 얹는 법 | 어떤 **계열**이 기여하는가 |
| 바꾸는 것 | 신경망 부품 (S1→S4) | 축 구성·프리셋·결합계수 | **모델 계열과 목적함수·타깃** |
| 고정하는 것 | 데이터·채점기 | 데이터·채점기 | **데이터·채점기·피처(77/73)** |

`pipeline/` 이 신경망 안쪽을 파고, `basemodel/` 이 화면 계약을 얹는다면, 여기는
**바깥쪽 선택지**를 훑는다 — GBDT 구현을 바꾸면? 목적함수를 바꾸면? 타깃을 바꾸면?
배깅이면? 레포가 `docs/02-모델.md` 에서 "모델 자체로는 차별화 못 한다. 다들 비슷한 걸
쓴다"고 **예측**만 해두고 재보지 않은 부분이다.

## 1. 고정한 것 — 바꾸면 레포의 다른 표와 비교가 무효다

- **피처**: `common.feature_cols()` 의 **77 / 73** 그대로. 늘리지도 줄이지도 않는다
- **데이터**: 팀 파케이 `pipeline/data`. `usable()` 후 train 381,966두 36,249경주 /
  valid 13,252두 1,254경주. **game 미사용**(하네스가 안 내준다) · **test 미개봉**
- **전처리**: `common.load()` = `schema_v2.clean()` + `usable()`
- **정렬 유지** — race_id 연속 블록. LightGBM `group` / XGBoost `qid` / CatBoost
  `group_id` 가 전부 그 순서를 가정한다. `harness.qid()` 가 assert 로 강제한다
- 데이터를 직접 읽지 않는다. 전부 `harness` → `common` 을 거친다

## 2. 계열을 고른 근거 — 레포가 안 덮은 곳만

| 축 | 왜 | 계열 |
|---|---|---|
| GBDT 다른 구현·목적함수 | "표 데이터는 GBDT" 가 정설인데 레포는 LightGBM `lambdarank` **한 설정**만 재봤다 | `lgb_xendcg` `lgb_binary` `xgb_pairwise` `xgb_ndcg` `hist_gb` |
| **PL 우도를 트리로** | 레포의 Plackett-Luce 손실은 전부 **신경망**(S1~S5·AxisRanker)에만 얹혀 있었다. Benter 의 모델식을 GBDT 로 푸는 셈 | `cat_querysoftmax` |
| 범주형 native | 팀 README 의 "범주형 native 악화(29.2)"는 LightGBM `categorical_feature` 얘기. CatBoost 의 ordered target statistics 는 기제가 다르다 | `cat_yetirank` `cat_querysoftmax` |
| 배깅 | Lessmann 외(2010)가 "경쟁구조 반영 RF 가 조건부로짓 상회" 라고 보고했고 `schema_v2.LIT["LS"]` 에 등재돼 있는데 아무도 안 돌렸다 | `rf` `extra_trees` `rf_norm` |
| **타깃 교체** | 레포 전 모델의 목적함수가 순위 아니면 1착확률이다. "몇 초에 달릴까"를 맞혀 빠른 순으로 세우는 실무 고전 방식(`LIT["PACE"]`)은 미실행 | `lgb_speedfig` |
| 라벨 프레이밍 | `y_rel = max(0, dusu−ord)` 는 **두수에 비례**한다(16두 1착 15 / 8두 1착 7). group 안에서만 비교하는 랭킹엔 무해하지만 행 단위 회귀 계열엔 불공정 | `rf_norm` `hist_gb_norm` |
| **결합** | 레포가 세 번 같은 결론에 도달했다 — "결합식이 전부다". 최고 기록도 단일 모델이 아니라 앙상블(LGB+S3 35.3) | `arena.run_combine` |

**전 브랜치 확인** — `xgboost`·`catboost`·`RandomForest` 는 레포에 **코드로 존재하지 않는다**
(`docs/02-모델.md` 에 "기성품이 있긴 하다… 그래도 직접 짠다"로 한 번 언급되고 기각된 게 전부).
`y_speed_fig` 도 GRU 이력의 *과거* 피처로만 쓰이고 학습 라벨로 쓴 곳이 없다.

## 3. 재는 법 — 두 가지를 일부러 다르게 한다

**① 적중률은 팀 채점기 그대로, 유의성은 쌍체.** 절대 적중률의 SE ±0.85%p 는 경주
1,254개라는 표본 한계라 시드를 더 돌려도 안 줄지만, 두 모델은 **같은 경주**를 맞히므로
쌍으로 묶으면 공통 분산이 상쇄돼 3배 예민해진다. 판정은 쌍체로만 한다.
동률은 검정에서만 1/k 기대값으로 처리한다(표는 팀 채점기와 같은 자를 유지).

**② logloss 는 온도 보정 후에 비교한다.** ★ 계열마다 점수 척도가 다르고 온도 1 의
경주 내 softmax 는 그 척도를 그대로 확률로 읽는다. 확률을 그대로 내놓는 계열은
softmax 가 한 번 더 걸려 **거의 균등**이 되고(`lgb_binary` 2.2511 → 보정 후 1.9542),
회귀 계열은 출력 범위가 좁아 같은 일이 생긴다. **모델이 나쁜 게 아니라 척도가 다른 것**이다.
온도는 valid 2-fold 교차적합으로 적합한다. 표에 두 열을 다 둔다 — `logloss`(온도 1,
레포의 다른 표와 같은 조건) 와 `logloss*`(보정 후, 계열 간 비교용).

## 4. 실행

```bash
cd pipeline && TEAM_REPO=<...>/S15P21A304 bash sync_dataset.sh && cd ../arena
uv sync

uv run python -m arena.run --seeds 0                 # 선별 (1시드)
uv run python -m arena.run --seeds 0,1,2              # 판정 (시드 3개)
uv run python -m arena.run --pop --seeds 0,1,2        # 77피처 갈래
uv run python -m arena.run_combine --seeds 0,1,2      # 앙상블 + Benter 2단계

uv run python -m arena.run --families cat_querysoftmax,xgb_ndcg --seeds 0
```

예측은 `artifacts/pred/<family>_<77|73>_<seed>.npy` 에 캐시된다 — 결합 단계가 다시
학습하지 않고 집어 쓰고, 같은 명령을 다시 돌리면 캐시를 읽는다.
Windows 에서는 `PYTHONUTF8=1` 을 붙일 것.

## 5. 구조

```
arena/
  harness.py    팀 하네스 진입점 — 피처·전처리·정렬을 고정한다
  evaluate.py   top1/top3(팀 채점기) · 온도보정 logloss · 쌍체 부트스트랩 · McNemar
  models.py     계열 정의. 각 함수는 (tr, ev, feats, seed) → 점수
  combine.py    경주내 z 앙상블 · 조건부로짓 MLE · Benter 2단계(2-fold 교차적합)
  run.py        계열 비교 CLI
  run_combine.py 결합 CLI + 계열 간 상관 진단
  ledger.md     실험 장부 — 한 실험 = 한 줄
```

## 6. 결과

→ `ledger.md`
