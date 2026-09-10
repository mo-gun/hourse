# -*- coding: utf-8 -*-
"""9월 원장 보강 + 내일(9/11) 출전표 분리 저장.

원장이 2026-08-31 까지라 as-of 피처가 열흘 낡아 있다. API4_3 은 rc_month 한 번으로
**완료분과 예정분을 같이** 준다(예정 경기는 결과 필드만 빈다) — 그래서 세 콜로 끝난다.

  data/raw/ledger_2010_2026.csv   ← 완료분만 append (ord 가 찬 행)
  data/raw/upcoming_20260911.csv  ← 9/11 출전표 (예측 대상, 결과 필드 빈 채로)

★ 완료분과 예정분을 **절대 같은 파일에 섞지 않는다.** build_v2.load_ledger 가
  ord 로 유효 완주를 거르므로 섞으면 조용히 사라지거나, 더 나쁘게는 가짜 착순이
  as-of 집계에 들어간다.
멱등 — 원장의 최대 rcDate 보다 큰 완료분만 붙인다. 두 번 돌려도 중복이 안 생긴다.
"""
import csv, io, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kra_client import KRA

LEDGER = "data/raw/ledger_2010_2026.csv"
COLS = json.load(io.open("data/raw/ledger_columns.json", encoding="utf-8"))
TARGET = 20260911
MONTH = "202609"


def is_done(r):
    """완주 결과가 실제로 들어온 행인가. 이 API 는 빈 값을 '0' 플레이스홀더로 준다."""
    o = str(r.get("ord", "")).strip()
    return o.isdigit() and int(o) >= 1


def main():
    import pandas as pd
    cur = pd.to_numeric(pd.read_csv(LEDGER, usecols=["rcDate"], dtype=str,
                                    encoding="utf-8-sig")["rcDate"], errors="coerce")
    hi = int(cur.max())
    print(f"원장 현재 최대 rcDate = {hi}")

    kra = KRA()
    done_rows, target_rows = [], []
    for meet, nm in ((1, "서울"), (2, "제주"), (3, "부산경남")):
        rows = kra.fetch("race_result", rc_month=MONTH, meet=meet)
        d = [r for r in rows if is_done(r) and int(r["rcDate"]) > hi]
        t = [r for r in rows if int(r["rcDate"]) == TARGET]
        done_rows += d
        target_rows += t
        print(f"  {nm:6s} 전체 {len(rows):>4} · 신규 완료 {len(d):>4} · {TARGET} 출전표 {len(t):>4}")

    if done_rows:
        days = sorted({int(r["rcDate"]) for r in done_rows})
        with io.open(LEDGER, "a", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore", restval="")
            w.writerows(done_rows)
        print(f"\n원장에 {len(done_rows):,}행 append — 개최일 {days}")
    else:
        print("\n새로 붙일 완료분 없음 (이미 최신)")

    if target_rows:
        out = f"data/raw/upcoming_{TARGET}.csv"
        with io.open(out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore", restval="")
            w.writeheader(); w.writerows(target_rows)
        races = sorted({(int(r["meet"] == "서울") or 0, r["meet"], int(r["rcNo"])) for r in target_rows})
        print(f"{TARGET} 출전표 {len(target_rows)}두 / "
              f"{len({(r['meet'], r['rcNo']) for r in target_rows})}경주 → {out}")
        # 결과 필드가 정말 비어 있는지 확인 — 차 있으면 미래 정보다
        for k in ("ord", "rcTime", "winOdds", "wgHr"):
            n = sum(1 for r in target_rows if str(r.get(k, "")).strip() not in ("", "0", "0()", "-"))
            print(f"    {k:9s} 채워진 행 {n}/{len(target_rows)}" + ("  ← 확인 필요" if n else "  (비어 있음, 정상)"))
    print(f"\n{kra.calls}콜 사용")


if __name__ == "__main__":
    main()
