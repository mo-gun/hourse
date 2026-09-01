# -*- coding: utf-8 -*-
"""
혈통 일일 수집 러너 — Windows 작업 스케줄러가 매일 부른다.

마필종합 API 가 벌크 조회를 막아놔서 말당 1콜이고 개발계정이 3,000콜/일이다.
원장 미커버 24,308두를 하루 2,500두씩 **출전 빈도 높은 순으로** 채운다.

다 채우면 스스로 스케줄 작업을 지우고 끝난다. 로그는 data/pedigree_daily.log.

수동 실행도 가능:  python tools/pedigree_daily.py
스케줄 해제:       schtasks /Delete /TN KRA_Pedigree /F
"""
import os, sys, io, subprocess, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")

LOG = "data/raw/pedigree_daily.log"
DAILY = 2500
TASK = "KRA_Pedigree"


def log(msg):
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M}] {msg}"
    print(line, flush=True)
    with io.open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def coverage():
    import pandas as pd
    ped = pd.read_csv("data/raw/aux_pedigree.csv", dtype=str, encoding="utf-8-sig")
    have = set(ped["hrNo"])
    led = pd.read_csv("data/raw/ledger_2010_2026.csv", dtype=str, encoding="utf-8-sig",
                      usecols=["hrNo"], low_memory=False)
    c = led["hrNo"].isin(have)
    missing = led[~c]["hrNo"].nunique()
    return len(ped), c.mean() * 100, missing


def main():
    before_n, before_cov, missing = coverage()
    log(f"시작 — 보유 {before_n:,}두 / 커버리지 {before_cov:.1f}% / 미보유 {missing:,}두")
    if missing == 0:
        log("미보유 0두 — 수집 완료. 스케줄 작업을 삭제한다.")
        subprocess.run(["schtasks", "/Delete", "/TN", TASK, "/F"],
                       capture_output=True)
        return

    r = subprocess.run([sys.executable, "tools/backfill_aux.py", "pedigree",
                        "--topN", str(DAILY)],
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        log(f"수집 실패 rc={r.returncode}: {(r.stderr or '')[:200]}")
        return

    after_n, after_cov, missing = coverage()
    log(f"완료 — 보유 {after_n:,}두 (+{after_n-before_n:,}) / "
        f"커버리지 {after_cov:.1f}% (+{after_cov-before_cov:.1f}%p) / 미보유 {missing:,}두")

    if missing == 0:
        log("전량 수집 완료. 스케줄 작업 삭제. build_dataset.py 를 다시 돌려 F2 를 채울 것.")
        subprocess.run(["schtasks", "/Delete", "/TN", TASK, "/F"], capture_output=True)
    else:
        log(f"남은 {missing:,}두 — 약 {-(-missing // DAILY)}일 더 필요")


if __name__ == "__main__":
    main()
