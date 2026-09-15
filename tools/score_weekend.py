# -*- coding: utf-8 -*-
"""블라인드 예측 여러 날을 합쳐 채점한다 — 하루 표본으로는 아무것도 판정할 수 없다.

raceday.py score 는 하루씩 채점하고 out/scored_{날짜}.csv 를 남긴다. 그런데 하루는
10~17경주라 표준오차가 ±12%p 근처다. 34% 와 50% 를 구별하지 못한다는 뜻이다.
그래서 날짜를 모아 한 번에 읽는다.

같이 재는 것 — 셋은 뜻이 다르다
    1착 적중    1픽이 그대로 우승               기대 34%
    1픽 연승권  1픽이 3착 이내                  기대 63%
    3픽중 1착   우리 1·2·3픽 중에 우승마가 있음   기대 66%

시장(인기 1위마)과 나란히 놓고, **픽이 갈린 경주**만 따로 센다. 둘이 같은 말을 고른
경주는 우열을 못 가리므로 전체 적중률 비교는 대부분 같은 정보를 두 번 세는 것이다.

    python tools/score_weekend.py 20260912 20260913
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).resolve().parent.parent / "out"
EXP = dict(적중=34, 연승=63, 삼픽=66)


def wilson(k, n, z=1.96):
    """윌슨 구간 — 표본이 작을 때 정규근사보다 정직하다."""
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h) * 100, min(1, c + h) * 100)


def main():
    days = sys.argv[1:] or ["20260912", "20260913"]
    frames = []
    for d in days:
        p = OUT / ("scored_%s.csv" % d)
        if not p.exists():
            print("  없음 %s — raceday.py score --date %s 를 먼저 돌려라" % (p.name, d))
            continue
        t = pd.read_csv(p, encoding="utf-8-sig")
        t.insert(0, "날짜", d[4:6] + "/" + d[6:])
        frames.append(t)
    if not frames:
        raise SystemExit("채점 파일이 없다.")
    T = pd.concat(frames, ignore_index=True)
    D = T[T["적중"].isin(["O", "X"])].copy()
    if not len(D):
        raise SystemExit("결과가 확정된 경주가 없다.")

    print(D[["날짜", "장", "R", "발주", "두수", "픽1", "일착", "배당",
             "적중", "연승", "삼픽", "순위", "시장", "갈림"]].to_string(index=False))

    n = len(D)
    print("")
    print("=" * 74)
    print("합계 %d경주 (%s) · 평균 %.1f두" % (n, " + ".join(days), D["두수"].mean()))
    print("=" * 74)
    print("%-14s%10s%10s%22s" % ("", "적중", "비율", "95% 구간"))
    print("-" * 60)
    for c, lab in (("적중", "1착 적중"), ("연승", "1픽 연승권"), ("삼픽", "3픽중 1착")):
        k = int((D[c] == "O").sum())
        lo, hi = wilson(k, n)
        print("%-14s%7d/%-3d%9.1f%%%12.1f ~ %.1f%%   기대 %d%%"
              % (lab, k, n, k / n * 100, lo, hi, EXP[c]))

    # 시장 — 인기 1위마가 우승했는가
    mk = (D["시장"].astype(str).str.strip() == D["일착"].astype(str).str.strip())
    lo, hi = wilson(int(mk.sum()), n)
    print("%-14s%7d/%-3d%9.1f%%%12.1f ~ %.1f%%   (참고)"
          % ("시장 1착", int(mk.sum()), n, mk.mean() * 100, lo, hi))

    # 갈린 경주만 — 여기가 실제 비교 지점이다
    S = D[D["갈림"] == "★"]
    print("")
    print("픽이 갈린 경주 %d/%d — 같은 말을 고른 경주는 우열을 못 가린다" % (len(S), n))
    if len(S):
        u = int((S["적중"] == "O").sum())
        m = int((S["시장"].astype(str).str.strip() == S["일착"].astype(str).str.strip()).sum())
        print("    우리 적중 %d · 시장 적중 %d · 둘 다 틀림 %d" % (u, m, len(S) - u - m))
        for r in S.itertuples():
            who = "우리" if r.적중 == "O" else ("시장" if r.시장 == r.일착 else "둘 다 틀림")
            print("      %s %s %2dR  우리 %-8s / 시장 %-8s → 1착 %-8s  %s"
                  % (r.날짜, r.장, r.R, r.픽1, r.시장, r.일착, who))

    # 우승마가 우리 예측에서 몇 위였나 — 적중/실패보다 정보가 많다
    rk = D[D["순위"] > 0]["순위"]
    if len(rk):
        print("")
        print("우승마가 우리 순위에서 몇 위였나 (%d경주)" % len(rk))
        vc = rk.value_counts().sort_index()
        for k, v in vc.items():
            print("    %2d위  %s %d" % (k, "■" * v, v))
        print("    중앙값 %.0f위 · 평균 %.1f위" % (rk.median(), rk.mean()))

    p = OUT / "scored_weekend.csv"
    T.to_csv(p, index=False, encoding="utf-8-sig")
    print("")
    print("→ %s" % p.name)


if __name__ == "__main__":
    main()
