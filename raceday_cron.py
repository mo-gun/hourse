# -*- coding: utf-8 -*-
r"""경마 시행일 자동 실행 — 서버/PC 에 이것 하나만 걸면 된다.

## 무엇을 하나

    ① 편성 확인   오늘 경주가 있나. 없으면 **아무것도 안 하고 끝낸다** (주 3일만 있다)
    ② 원장 보강   지난 개최일 결과를 원장에 덧붙인다 (멱등)
    ③ 프레임 빌드 as-of 롤링 + 보조 인덱스 조인. 4~6분
    ④ 자동 예측   경주마다 T-10분 창에서 발사. 마지막 경주까지 돌고 끝난다

②③ 은 아침에, ④ 는 첫 발주 전에 시작하면 된다. 한 번에 다 걸어도 된다 —
④ 가 첫 경주 창까지 알아서 기다린다.

## 호출량 — 이게 사고의 핵심이었다

    자동 예측   편성 3콜 + 경주당 2콜   17경주면 **약 37콜**
    원장 보강   개최일당 3~6콜
    배당 관측기 --watch 를 회수 안 하면 **하루 6,912콜** ← 2026-09-14 에 할당량을 말린 원인

공공데이터포털 개발계정은 보통 일 10,000콜이다. **예측 파이프라인은 태우지 않는다.**
그래도 예산을 건다 — 어떤 버그도 하루 몫을 다 먹지 못하게.

## 서버에 거는 법

    # 리눅스 crontab — 경마는 금·토·일에 열린다
    0 8 * * 5,6,0  cd /srv/horse && /usr/bin/python3 raceday_cron.py >> logs/cron.log 2>&1

    # Windows 작업 스케줄러
    schtasks /create /tn "raceday" /tr "python C:\horse\raceday_cron.py" /sc weekly /d FRI,SAT,SUN /st 08:00

    환경변수
        KRA_API_KEY_ENCODED   필수
        KRA_CALL_BUDGET       기본 1500. 이 프로세스가 넘으면 즉시 멈춘다
        KRA_CALL_LEDGER       날짜별 누적 호출 기록 파일 (선택)

    python raceday_cron.py                 # 오늘
    python raceday_cron.py --date 20260919
    python raceday_cron.py --skip-predict  # 빌드까지만
    python raceday_cron.py --dry-run       # 편성만 확인
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("KRA_CALL_BUDGET", "1500")
os.environ.setdefault("KRA_CALL_LEDGER", str(HERE / "out" / "kra_calls.json"))
sys.path.insert(0, str(HERE))


def log(msg):
    print("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), msg), flush=True)


def run(args, why):
    log("· %s  (%s)" % (why, " ".join(args[1:])))
    t0 = time.time()
    r = subprocess.run([sys.executable] + args, cwd=str(HERE))
    log("  %s %s (%.0f초)" % (why, "완료" if r.returncode == 0 else
                             "실패 rc=%d" % r.returncode, time.time() - t0))
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=int, default=int(datetime.now().strftime("%Y%m%d")))
    ap.add_argument("--skip-predict", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    from kra_client import KRA, KRAError, KRAQuotaError            # noqa: E402

    log("=" * 60)
    log("경마 시행일 자동 실행 %d" % a.date)
    log("호출 예산 %s · 기록 %s" % (os.environ["KRA_CALL_BUDGET"],
                                Path(os.environ["KRA_CALL_LEDGER"]).name))

    # ── ① 편성 확인 — 경주가 없으면 여기서 끝낸다 ─────────────────────
    try:
        kra = KRA()
        n = 0
        for meet in (1, 2, 3):
            try:
                n += len(kra.fetch("race_plan", meet=meet, rc_date=str(a.date)))
            except KRAError:
                pass
    except KRAQuotaError as e:
        log("★ 할당량 초과 — 오늘은 아무것도 못 한다. %s" % e)
        return 2
    except Exception as e:
        log("★ API 확인 실패: %s: %s" % (type(e).__name__, e))
        return 1

    if not n:
        log("편성 없음 — 경마 시행일이 아니다. 종료 (%d콜)" % kra.calls)
        return 0
    log("편성 %d경주 (%d콜)" % (n, kra.calls))
    if a.dry_run:
        return 0

    # ── ② 원장 보강 ─────────────────────────────────────────────────
    if not a.skip_build:
        if not run(["tools/append_september.py"], "원장 보강"):
            log("  ⚠ 보강 실패 — 그래도 계속한다 (기존 원장으로 예측 가능)")

        # ── ③ 프레임 빌드 ───────────────────────────────────────────
        if not run(["raceday.py", "build", "--date", str(a.date)], "프레임 빌드"):
            log("★ 빌드 실패 — 예측을 할 수 없다")
            return 1

    # ── ④ 자동 예측 ─────────────────────────────────────────────────
    if a.skip_predict:
        log("예측 건너뜀 (--skip-predict)")
        return 0
    ok = run(["raceday.py", "auto", "--date", str(a.date)], "자동 예측")
    log("=" * 60)
    log("끝. 예측 기록 out/live_pred_%d.csv" % a.date)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
