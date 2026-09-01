# -*- coding: utf-8 -*-
"""
게임풀 영구 예약 — 한 번만 실행하고 결과를 커밋한다.

왜 파일로 박아두나
  2025~2026 도 학습에 넣고 매주 월요일 재학습하기로 했다. 게임풀이 '뒤쪽 시간 블록'이면
  재학습할 때마다 게임 경주가 학습에 흡수된다. 그래서 개최일 단위로 15% 를 고정 시드로
  뽑아 파일에 박고, 빌더가 매번 이 파일을 읽어 학습에서 뺀다.

  ★ 이 파일이 바뀌면 예약이 무의미해진다. 재생성 금지.
  ★ 개최일(경마장별) 통째로 빼는 이유: 경주 단위로 흩뿌리면 같은 날 앞뒤 경주가
    학습에 남아 애매해진다.

실행: python tools/make_holdout.py
"""
import os, sys, json, io
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import schema_v2 as S

LEDGER = "data/raw/ledger_2010_2026.csv"


def main():
    if os.path.exists(S.HOLDOUT_FILE):
        d = json.load(io.open(S.HOLDOUT_FILE, encoding="utf-8"))
        print(f"이미 존재 — {len(d['days']):,}개 개최일 예약됨 (seed={d['seed']}).")
        print("재생성하면 예약이 무의미해진다. 지우고 다시 만들려면 의도적으로 파일을 삭제할 것.")
        return

    led = pd.read_csv(LEDGER, dtype=str, encoding="utf-8-sig",
                      usecols=["rcDate", "meet", "rcNo", "ord"], low_memory=False)
    led["ord"] = pd.to_numeric(led["ord"], errors="coerce")
    led = led[led["ord"].between(1, 16)]
    led["rcDate"] = pd.to_numeric(led["rcDate"], errors="coerce").astype("Int64")
    led["day"] = led["rcDate"].astype(str) + "_" + led["meet"].astype(str).str.strip()
    led["race_id"] = led["day"] + "_" + led["rcNo"].astype(str).str.strip()

    y = led["rcDate"] // 10000
    pool = led[(y >= S.HOLDOUT["year_from"]) & (y <= S.HOLDOUT["year_to"])]
    days = np.array(sorted(pool["day"].unique()))

    rng = np.random.default_rng(S.HOLDOUT["seed"])
    n = int(round(len(days) * S.HOLDOUT["day_frac"]))
    sel = sorted(rng.choice(days, n, replace=False).tolist())

    n_races = pool[pool["day"].isin(sel)]["race_id"].nunique()
    total = led["race_id"].nunique()
    out = {
        "_주의": "게임 전용 영구 예약. 학습·검증·평가 어디에도 쓰지 말 것. 재생성 금지.",
        "schema_version": S.SCHEMA_VERSION,
        "seed": S.HOLDOUT["seed"],
        "year_from": S.HOLDOUT["year_from"],
        "year_to": S.HOLDOUT["year_to"],
        "day_frac": S.HOLDOUT["day_frac"],
        "n_days": len(sel),
        "n_races": int(n_races),
        "days": sel,
    }
    os.makedirs(os.path.dirname(S.HOLDOUT_FILE), exist_ok=True)
    io.open(S.HOLDOUT_FILE, "w", encoding="utf-8").write(
        json.dumps(out, ensure_ascii=False, indent=1))
    print(f"저장 {S.HOLDOUT_FILE}")
    print(f"  예약 개최일 {len(sel):,}일 → 게임풀 {n_races:,}경주 "
          f"(전체 {total:,}경주의 {n_races/total*100:.1f}%)")
    print(f"  학습 가용 {total-n_races:,}경주")
    print(f"  시드 {S.HOLDOUT['seed']} — 절대 변경 금지")


if __name__ == "__main__":
    main()
