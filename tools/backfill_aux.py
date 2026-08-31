# -*- coding: utf-8 -*-
"""
보조 데이터 백필 — 조교 · 장제 · 혈통.

조교(trcontihi)  : 월 단위. 원본 7.8M행은 과하므로 **(마번, 훈련일) 일별 롤업**으로 줄인다.
                   F1 이 필요한 건 28일 창의 횟수·시간·수영 여부뿐이다.
장제(HorseShoe_1): 연 단위. 원본 그대로 (연 ~3만행).
혈통(API8_2)     : 경마장별 벌크 3콜. **현역마만 4,088두** — 과거 출전마는 못 덮는다.
                   나머지는 마필종합(API42_1) 말 단위 조회가 필요하고 1두=1콜이라
                   개발계정 3,000콜/일 제한에 걸린다. --pedigree-topN 으로
                   원장 출전 빈도 상위 N두만 우선 수집한다(하루치씩 나눠 실행).

실행
  python tools/backfill_aux.py train          # 조교
  python tools/backfill_aux.py shoe           # 장제
  python tools/backfill_aux.py pedigree       # 현역마 벌크
  python tools/backfill_aux.py pedigree --topN 2500   # 과거마 우선순위 수집
"""
import urllib.request, urllib.parse, ssl, json, csv, io, os, sys, time, argparse
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
KEY = [l.split("=", 1)[1].strip() for l in open(".env", encoding="utf-8")
       if l.startswith("KRA_API_KEY_ENCODED")][0]
BASE = "https://apis.data.go.kr/B551015"
ROWS = 5000


def call(path, page=1, rows=ROWS, **p):
    q = {"serviceKey": KEY, "_type": "json", "numOfRows": str(rows), "pageNo": str(page)}
    q.update({k: str(v) for k, v in p.items()})
    url = f"{BASE}/{path}?" + "&".join(
        f"{k}={v if k == 'serviceKey' else urllib.parse.quote(str(v))}" for k, v in q.items())
    last = None
    for att in range(4):
        try:
            raw = urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                timeout=90, context=CTX).read().decode("utf-8", "replace")
            if not raw.lstrip().startswith("{"):
                raise RuntimeError(raw[:80].replace("\n", " "))
            d = json.loads(raw)
            b = d.get("response", {}).get("body", {}) or {}
            it = (b.get("items") or {})
            it = it.get("item", []) if isinstance(it, dict) else []
            it = [it] if isinstance(it, dict) else it
            return int(b.get("totalCount") or 0), it
        except Exception as e:
            last = e
            time.sleep(1.5 * (att + 1))
    raise RuntimeError(f"4회 실패: {last}")


def page_all(path, **p):
    tc, rows = call(path, **p)
    pg = 2
    while len(rows) < tc:
        _, more = call(path, page=pg, **p)
        if not more:
            break
        rows += more
        pg += 1
    return rows


# ══════════════════════════════════════════════════════════════════════
def do_train(y0=2010, y1=2026):
    """조교 → (hrNo, trDate) 일별 롤업."""
    out = "data/aux_train.csv"
    prog_f = "data/aux_train_progress.json"
    done = set(json.load(io.open(prog_f, encoding="utf-8"))) if os.path.exists(prog_f) else set()
    new = not done or not os.path.exists(out)
    f = io.open(out, "w" if new else "a", encoding="utf-8-sig", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["hrNo", "trDate", "n_sessions", "minutes", "swim", "chul"])
    calls = tot = 0
    t0 = time.time()
    for y in range(y0, y1 + 1):
        for m in range(1, 13):
            tag = f"{y}{m:02d}"
            if tag in done:
                continue
            last = 31
            try:
                rows = page_all("trcontihi/gettrcontihi",
                                tr_date_fr=f"{tag}01", tr_date_to=f"{tag}{last}")
            except Exception as e:
                print(f"  [실패] {tag}: {str(e)[:70]}", file=sys.stderr)
                continue
            calls += 1
            agg = defaultdict(lambda: [0, 0.0, 0, ""])
            for r in rows:
                hn = str(r.get("hrNo", "")).strip()
                dt = str(r.get("trDate", "")).strip()
                if not hn or not dt:
                    continue
                a = agg[(hn, dt)]
                a[0] += 1
                try:
                    a[1] += float(str(r.get("time", "0")).strip() or 0)
                except Exception:
                    pass
                if str(r.get("swimTr", "")).strip() in ("Y", "1", "수영"):
                    a[2] += 1
                a[3] = str(r.get("chulGubun", "")).strip() or a[3]
            for (hn, dt), a in agg.items():
                w.writerow([hn, dt, a[0], round(a[1], 1), a[2], a[3]])
            f.flush()
            tot += len(agg)
            done.add(tag)
        io.open(prog_f, "w", encoding="utf-8").write(json.dumps(sorted(done)))
        print(f"  {y}: 누적 롤업 {tot:,}행 / {calls}콜 / {time.time()-t0:.0f}s", flush=True)
    f.close()
    print(f"완료 {out}: {tot:,}행, {calls}콜, {os.path.getsize(out)/1e6:.0f} MB")


def do_shoe(y0=2010, y1=2026):
    out = "data/aux_shoe.csv"
    cols = ["hrNo", "hrName", "shoeDate", "codeName2", "codeName3", "prName", "meet"]
    f = io.open(out, "w", encoding="utf-8-sig", newline="")
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", restval="")
    w.writeheader()
    tot = calls = 0
    t0 = time.time()
    for y in range(y0, y1 + 1):
        for meet in (1, 2, 3):
            try:
                rows = page_all("API191_1/HorseShoe_1", meet=meet,
                                shoe_date_fr=f"{y}0101", shoe_date_to=f"{y}1231")
            except Exception as e:
                print(f"  [실패] {y} meet={meet}: {str(e)[:70]}", file=sys.stderr)
                continue
            calls += 1
            w.writerows(rows)
            tot += len(rows)
        f.flush()
        print(f"  {y}: 누적 {tot:,}행 / {calls}콜 / {time.time()-t0:.0f}s", flush=True)
    f.close()
    print(f"완료 {out}: {tot:,}행, {calls}콜")


PED_COLS = ["hrNo", "hrName", "birthday", "sex", "faHrNo", "faHrName",
            "moHrNo", "moHrName", "damsireNo", "damsireName", "src"]


def _load_ped():
    if not os.path.exists("data/aux_pedigree.csv"):
        return {}
    with io.open("data/aux_pedigree.csv", encoding="utf-8-sig") as fh:
        return {r["hrNo"]: r for r in csv.DictReader(fh)}


def _save_ped(d):
    with io.open("data/aux_pedigree.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PED_COLS, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(d.values())


def do_pedigree(topn=0):
    """현역마 벌크(3콜) → 필요시 원장 빈도 상위 N두를 말 단위로 보충."""
    ped = _load_ped()
    print(f"기존 혈통 {len(ped):,}두")

    for meet in (1, 2, 3):                       # 현역마 벌크
        rows = page_all("API8_2/raceHorseInfo_2", meet=meet)
        for r in rows:
            hn = str(r.get("hrNo", "")).strip()
            if not hn or hn in ped:
                continue
            ped[hn] = {"hrNo": hn, "hrName": r.get("hrName", ""),
                       "birthday": r.get("birthday", ""), "sex": r.get("sex", ""),
                       "faHrNo": r.get("faHrNo", ""), "faHrName": r.get("faHrName", ""),
                       "moHrNo": r.get("moHrNo", ""), "moHrName": r.get("moHrName", ""),
                       "damsireNo": "", "damsireName": "", "src": "API8_2"}
    print(f"현역마 벌크 후 {len(ped):,}두")
    _save_ped(ped)

    if not topn:
        print("과거마 보충은 --topN 으로. 예: --topN 2500 (하루 한도 내)")
        return

    # 원장에서 출전 빈도 상위 말부터 — 행 커버리지를 빨리 올린다
    import pandas as pd
    led = pd.read_csv("data/ledger_2010_2026.csv", dtype=str,
                      encoding="utf-8-sig", usecols=["hrNo"], low_memory=False)
    freq = led["hrNo"].value_counts()
    todo = [h for h in freq.index if h and h not in ped][:topn]
    print(f"보충 대상 {len(todo):,}두 (원장 미커버 상위)")

    t0 = time.time()
    for i, hn in enumerate(todo, 1):
        try:
            _, it = call("API42_1/totalHorseInfo_1", rows=1, hr_no=hn)
        except Exception:
            continue
        if not it:
            continue
        r = it[0]
        ped[hn] = {"hrNo": hn, "hrName": r.get("hrName") or r.get("hrNm", ""),
                   "birthday": r.get("birthDt", ""), "sex": r.get("sex", ""),
                   "faHrNo": r.get("fhrNo", ""), "faHrName": r.get("fhrNm", ""),
                   "moHrNo": r.get("mhrNo", ""), "moHrName": r.get("mhrNm", ""),
                   "damsireNo": r.get("mhrFhrNo", ""), "damsireName": r.get("mhrFhrNm", ""),
                   "src": "API42_1"}
        if i % 250 == 0:
            _save_ped(ped)
            print(f"  {i:,}/{len(todo):,}  {time.time()-t0:.0f}s", flush=True)
    _save_ped(ped)
    cov = len(ped)
    print(f"완료 data/aux_pedigree.csv: {cov:,}두")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["train", "shoe", "pedigree"])
    ap.add_argument("--topN", type=int, default=0)
    ap.add_argument("--y0", type=int, default=2010)
    ap.add_argument("--y1", type=int, default=2026)
    a = ap.parse_args()
    if a.what == "train":
        do_train(a.y0, a.y1)
    elif a.what == "shoe":
        do_shoe(a.y0, a.y1)
    else:
        do_pedigree(a.topN)
