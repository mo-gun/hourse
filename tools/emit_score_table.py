# -*- coding: utf-8 -*-
"""점수 테이블 적재기 (AI-05) — entry_feature_score 에 넣을 행을 만든다.

## 무엇을 내보내나

경주마다 출전마 × 지표 41칸. API명세서 §11-1 `POST /ai/score-tables` 응답 형식 그대로다.

    featureKeys  ax_condition … ax_ability (6)  +  표시 지표 35
    scores       0~100. 축은 타워 출력의 경주 내 순위 백분위, 표시 지표는 지표값의 순위 백분위
    rawValue     표시 지표의 원본값 (출전표에 그대로 찍힌다). 축 행은 null
    rawProb      6축 균등 가중치일 때의 1착 확률 — 참고용

**계산의 전부는 ax_* 여섯 칸이다.** 표시 지표 35는 화면용이며, 이걸 평균해서 축을
만들면 top-1 이 32.9% → 28.3% 로 떨어진다 (tools/fit_group_inner_weights.py 실측).

## 확률 계약

    raw = Σ( groupWeights[k] × ax_k ) / 100
    p   = softmax(raw × 0.0445),  경주 안에서

0.0445 는 독립 측정 다섯 번에서 0.0406~0.0478 로 재현됐다. 축을 순위 환산하기 때문에
밑에 어떤 모델이 오든 이 상수가 버틴다.

## 규약
    valid 로만 만든다 · test 는 열지 않고 game 은 쓰지 않는다 · 정렬을 바꾸지 않는다

    python tools/emit_score_table.py                 # valid 앞 20경주
    python tools/emit_score_table.py --races 200
    python tools/emit_score_table.py --sql            # INSERT 문도 같이
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_temperature import AXES, race_index, C                  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"
CKPT = Path(r"c:/Users/SSAFY/Desktop/말고리즘/horse-pred-engine/basemodel/artifacts/runs")
STEM = "axis_73_s20260901_final"
SPEC_T = 0.0445
MODEL_VERSION = "2026-09-15-axis73"
MEETNM = {1: "SEOUL", 2: "JEJU", 3: "BUKYEONG"}


def rank100(v, rid):
    return (pd.Series(v).groupby(rid.values).rank(pct=True) * 100).to_numpy(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--races", type=int, default=20)
    ap.add_argument("--sql", action="store_true")
    a = ap.parse_args()

    master = json.loads((OUT / "feature_master.json").read_text(encoding="utf-8"))
    shown = [f for f in master["features"] if f["role"] == "PRIMARY"]

    va = C.load("valid")
    Z = np.load(CKPT / (STEM + ".valid_axes.npy"))
    assert len(Z) == len(va), "행 수가 valid 와 다르다 — 정렬이 어긋났다"
    rid = va["race_id"]

    # 축 6칸 — 타워 출력의 경주 내 순위 백분위
    AX = np.stack([rank100(Z[:, i], rid) for i in range(6)], 1)
    # 표시 지표 35칸 — 지표값의 경주 내 순위 백분위 (방향 반영)
    disp, raws = [], []
    for f in shown:
        col = f["source_column"]
        v = pd.to_numeric(va[col], errors="coerce")
        p = (v.groupby(rid).rank(method="average", pct=True) * 100).fillna(50.0).to_numpy(float)
        disp.append(p if f["direction"] > 0 else 100.0 - p)
        raws.append(v.to_numpy(float))
    D, RV = np.stack(disp, 1), np.stack(raws, 1)

    keys = ["ax_" + g.lower() for g in AXES] + [f["code"] for f in shown]
    S = np.concatenate([AX, D], 1)

    # 참고용 rawProb — 균등 가중치
    q = race_index(va); nq = q.max() + 1
    raw = AX.mean(1)
    z = raw * SPEC_T
    m = np.zeros(nq); np.maximum.at(m, q, z)
    e = np.exp(z - m[q]); s = np.zeros(nq); np.add.at(s, q, e)
    P = e / s[q]

    races = list(dict.fromkeys(rid))[:a.races]
    now = datetime(2026, 9, 15, 11, 0, 0).isoformat()
    tables, n_rows = [], 0
    for r in races:
        idx = np.flatnonzero((rid == r).to_numpy())
        g = va.iloc[idx]
        meet = int(pd.to_numeric(g["meet"].iloc[0]))
        key = "%s-%s-%02d" % (MEETNM.get(meet, meet), str(g["rcDate"].iloc[0]),
                              int(pd.to_numeric(g["rcNo"].iloc[0])))
        entries = []
        for j, i in enumerate(idx):
            entries.append(dict(
                entryId=str(va["row_id"].iloc[i]),
                score=[round(float(x), 1) for x in S[i]],
                rawValue=[None] * 6 + [None if RV[i, k] != RV[i, k] else round(float(RV[i, k]), 3)
                                       for k in range(RV.shape[1])],
                rawProb=round(float(P[i]), 5)))
            n_rows += len(keys)
        tables.append(dict(raceKey=key, modelVersion=MODEL_VERSION,
                           featureKeys=keys, scores=entries, computedAt=now))

    p = OUT / "score_tables_sample.json"
    p.write_text(json.dumps(dict(
        contract="raw = (Σ_k groupWeights[k] * ax_k) / 100 ; p = softmax(raw * temperature) 경주 내",
        temperature=SPEC_T, modelVersion=MODEL_VERSION,
        note="계산은 ax_* 6칸으로만 한다. 표시 지표 35는 화면용이며 평균해서 축을 만들면 안 된다",
        tables=tables), ensure_ascii=False, indent=1), encoding="utf-8")

    print("경주 %d · 출전마 %d · 지표 %d칸 (축 6 + 표시 %d)"
          % (len(tables), sum(len(t["scores"]) for t in tables), len(keys), len(shown)))
    print("entry_feature_score 행 %d개 → %s (%.0fKB)" % (n_rows, p.name, p.stat().st_size / 1024))

    # ── 왕복 검증 — 쓴 파일을 **다시 읽어** 채점한다 ───────────────────
    # 메모리의 배열을 그대로 다시 재면 직렬화 버그(소수 1자리 반올림으로 생기는
    # 동점, 행 순서 뒤바뀜)를 못 잡는다. 파일에서 복원한 값으로만 채점한다.
    back = json.loads(p.read_text(encoding="utf-8"))
    by_row = {}
    for t in back["tables"]:
        ax_at = [t["featureKeys"].index("ax_" + g.lower()) for g in AXES]
        for e in t["scores"]:
            by_row[e["entryId"]] = [e["score"][i] for i in ax_at]

    sub = np.flatnonzero(va["row_id"].isin(by_row).to_numpy())
    assert len(sub) == sum(len(t["scores"]) for t in back["tables"]), "복원 행 수 불일치"
    A_back = np.array([by_row[va["row_id"].iloc[i]] for i in sub], float)

    # ⚠ 검증 가중치는 **슬라이더로 도달 가능한 것**을 쓴다.
    #   정확한 균등(16.67)은 정수·합100 제약상 만들 수 없는데, 그 값으로 재면
    #   축 순위 여섯의 평균이 자주 겹쳐 1위 동점이 3.3% 경주에서 생긴다.
    #   도달 가능한 정수 가중치에서는 0.2~0.6% 다 — 동점 깨기 차이를 성적 차이로
    #   잘못 읽게 된다. 기본형(17·17·12·22·19·13)으로 잰다.
    eq = np.array([17, 17, 12, 22, 19, 13], float)
    mm = C.evaluate(va.iloc[sub], A_back @ eq / 100.0)
    mem = C.evaluate(va.iloc[sub], AX[sub] @ eq / 100.0)
    full = C.evaluate(va, AX @ eq / 100.0)
    se = float(np.sqrt(0.33 * 0.67 / len(races)) * 100)

    print("")
    print("왕복 검증 — 쓴 파일을 다시 읽어 ax_* 여섯 칸만으로 채점 (기본형 가중치)")
    print("  파일에서 복원   top-1 %.1f%%  top-3 %.1f%%   (%d경주, 표준오차 ±%.1f%%p)"
          % (mm["top1"], mm["top3"], len(races), se))
    print("  메모리 원본     top-1 %.1f%%  top-3 %.1f%%   %s"
          % (mem["top1"], mem["top3"],
             "일치" if abs(mm["top1"] - mem["top1"]) < 1e-9 else
             "★ 어긋남 — 직렬화에서 값이 바뀌었다"))
    print("  valid 전체      top-1 %.1f%%  top-3 %.1f%%   (1,254경주, ±0.85%%p)"
          % (full["top1"], full["top3"]))

    if a.sql:
        L = ["-- entry_feature_score 적재 예시 (%d경주) — AI(건모) %s" % (len(races), MODEL_VERSION),
             "-- race_entry_id 는 팀 DB 의 값으로 바꿔야 한다. 여기 entryId 는 원장 row_id 다.",
             "INSERT INTO entry_feature_score",
             "  (race_entry_id, feature_id, raw_value, score, raw_prob, model_version, computed_at)",
             "VALUES"]
        rows = []
        for t in tables[:2]:
            for e in t["scores"]:
                for k, code in enumerate(t["featureKeys"]):
                    rv = e["rawValue"][k]
                    rows.append(
                        "  (/*%s*/ NULL, (SELECT feature_id FROM feature WHERE code='%s'), %s, %.1f, %.5f, '%s', '%s')"
                        % (e["entryId"], code, "NULL" if rv is None else rv,
                           e["score"][k], e["rawProb"], MODEL_VERSION, now))
        L.append(",\n".join(rows) + ";")
        q2 = OUT / "entry_feature_score_sample.sql"
        q2.write_text("\n".join(L) + "\n", encoding="utf-8")
        print("→ %s (앞 2경주 %d행)" % (q2.name, len(rows)))


if __name__ == "__main__":
    main()
