# -*- coding: utf-8 -*-
"""
1단계 — 경주기록 원장 백필 (스트리밍).

kra_client.backfill_csv 는 전량을 메모리에 쌓는다(697k행 × 93열 ≈ 수 GB).
여기서는 월 단위로 받아 즉시 디스크에 흘려보내고, 중단되면 이어받는다.

산출물
  data/ledger_2010_2026.csv   경주기록 원장 (말-경주 단위)
  data/ledger_progress.json   완료한 (year, month, meet) 목록 — 재실행 시 스킵

실행
  python tools/backfill_ledger.py                # 2010~2026
  python tools/backfill_ledger.py 2010 2026      # 범위 지정
"""
import urllib.request, urllib.parse, ssl, json, csv, io, os, sys, time

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
KEY = [l.split("=", 1)[1].strip() for l in open(".env", encoding="utf-8")
       if l.startswith("KRA_API_KEY_ENCODED")][0]
BASE = "https://apis.data.go.kr/B551015"
PATH = "API4_3/raceResult_3"
ROWS = 5000                      # 실측: 1000 상한 없음. 월×경마장 최대 ~1,100행이라 1콜로 끝난다

OUT = "data/raw/ledger_2010_2026.csv"
PROG = "data/raw/ledger_progress.json"
MEETS = (1, 2, 3)                # 1서울 2제주 3부산경남


def call(page=1, **p):
    q = {"serviceKey": KEY, "_type": "json", "numOfRows": str(ROWS), "pageNo": str(page)}
    q.update({k: str(v) for k, v in p.items()})
    url = f"{BASE}/{PATH}?" + "&".join(
        f"{k}={v if k == 'serviceKey' else urllib.parse.quote(str(v))}" for k, v in q.items())
    last = None
    for attempt in range(4):
        try:
            raw = urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                timeout=90, context=CTX).read().decode("utf-8", "replace")
            if not raw.lstrip().startswith("{"):
                raise RuntimeError("non-json: " + raw[:90].replace("\n", " "))
            d = json.loads(raw)
            hdr = d.get("response", {}).get("header", {})
            if hdr.get("resultCode") not in ("00", None):
                raise RuntimeError(f"{hdr.get('resultCode')} {hdr.get('resultMsg')}")
            b = d.get("response", {}).get("body", {}) or {}
            it = (b.get("items") or {})
            it = it.get("item", []) if isinstance(it, dict) else []
            it = [it] if isinstance(it, dict) else it
            return int(b.get("totalCount") or 0), it
        except Exception as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"4회 실패: {last}")


def discover_columns():
    """경마장마다 구간기록 컬럼이 달라서(서울 se*/sj*, 부산 bu*, 제주 je*) 합집합을 먼저 만든다."""
    cols = []
    for meet in MEETS:
        _, rows = call(meet=meet, rc_month="202605")
        if not rows:
            _, rows = call(meet=meet, rc_month="202405")
        for r in rows[:5]:
            for k in r:
                if k not in cols:
                    cols.append(k)
    return cols


def main(y0=2010, y1=2026):
    done = set()
    if os.path.exists(PROG):
        done = {tuple(x) for x in json.load(io.open(PROG, encoding="utf-8"))}
        print(f"이어받기: 이미 {len(done)}개 (년,월,경마장) 완료")

    new_file = not os.path.exists(OUT) or not done
    if new_file:
        cols = discover_columns()
        io.open("data/raw/ledger_columns.json", "w", encoding="utf-8").write(
            json.dumps(cols, ensure_ascii=False, indent=1))
        print(f"컬럼 합집합 {len(cols)}개 확정")
    else:
        cols = json.load(io.open("data/raw/ledger_columns.json", encoding="utf-8"))

    f = io.open(OUT, "w" if new_file else "a", encoding="utf-8-sig", newline="")
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", restval="")
    if new_file:
        w.writeheader()

    total, calls, t0 = 0, 0, time.time()
    for y in range(y0, y1 + 1):
        ycount = 0
        for m in range(1, 13):
            for meet in MEETS:
                if (y, m, meet) in done:
                    continue
                try:
                    tc, rows = call(meet=meet, rc_month=f"{y}{m:02d}")
                    calls += 1
                    page = 2
                    while len(rows) < tc:                     # 안전장치 — 5000 초과 월 대비
                        _, more = call(page=page, meet=meet, rc_month=f"{y}{m:02d}")
                        calls += 1
                        if not more:
                            break
                        rows += more
                        page += 1
                except Exception as e:
                    print(f"  [실패] {y}-{m:02d} meet={meet}: {str(e)[:90]}", file=sys.stderr)
                    continue
                if rows:
                    w.writerows(rows)
                    f.flush()
                total += len(rows)
                ycount += len(rows)
                done.add((y, m, meet))
        io.open(PROG, "w", encoding="utf-8").write(
            json.dumps(sorted(done), ensure_ascii=False))
        print(f"  {y}: {ycount:>7,}행  (누적 {total:,} / {calls}콜 / {time.time()-t0:.0f}s)",
              flush=True)

    f.close()
    print(f"\n완료 {OUT}: 이번 실행 {total:,}행 추가, {calls}콜, {time.time()-t0:.0f}s")
    if os.path.exists(OUT):
        print(f"파일 크기: {os.path.getsize(OUT)/1e6:.0f} MB")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 2010, int(a[1]) if len(a) > 1 else 2026)
