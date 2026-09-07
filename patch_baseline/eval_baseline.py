# -*- coding: utf-8 -*-
"""
시장 베이스라인(인기 1위마 전략) 재계산 — 동률 처리 교정.

문제
----
`F6_mkt_rank == 1` 인 행을 전부 세면 분모가 부풀려진다.
배당률이 같은 말이 2두 이상이면 rank 가 둘 다 1이 되기 때문이다(표준 min-rank, 버그 아님).
실제 베팅에서는 그중 **한 두만** 고르므로, 그렇게 계산한 적중률은 실제 전략보다 낮게 나온다.

교정
----
경주 단위로 집계하고, 동률이면 무작위 1두를 고르는 것의 기대값을 쓴다.

    경주 적중 기여 = (rank1 인 말 중 승리한 두수) / (rank1 인 말 두수)

동률이 없으면 0 또는 1이므로 기존 계산과 동일하다. 표본(경주 수)도 줄지 않는다.

    python eval_baseline.py <dataset_dir>
    python eval_baseline.py ../말고리즘/S15P21A304/docs/dataset
"""
import sys, os, collections
import pyarrow.parquet as pq

COLS = ['race_id', 'F6_mkt_rank', 'F6_mkt_prob', 'y_win', 'y_plc']


def load(path):
    t = pq.read_table(path, columns=COLS)
    per = collections.defaultdict(list)
    for r, k, q, w, p in zip(t.column('race_id').to_pylist(),
                             t.column('F6_mkt_rank').to_pylist(),
                             t.column('F6_mkt_prob').to_pylist(),
                             t.column('y_win').to_pylist(),
                             t.column('y_plc').to_pylist()):
        per[r].append((k, q, w, p))
    return per


def old_way(per, idx):
    """기존: rank==1 인 '행'을 전부 계수 (동률이면 분모가 늘어남)."""
    rows = [x[idx] for v in per.values() for x in v if x[0] == 1]
    return (sum(rows) * 100 / len(rows) if rows else 0.0), len(rows)


def fixed(per, idx, drop_no_odds=False):
    """교정: 경주 단위 집계 + 동률은 1/k 가중(무작위 1두 선택의 기대값)."""
    no_odds = {r for r, v in per.items()
               if len({round(q, 6) if q is not None else None for _, q, _, _ in v}) == 1}
    num = den = 0
    for r, v in per.items():
        if drop_no_odds and r in no_odds:
            continue
        tops = [x[idx] for x in v if x[0] == 1]
        if not tops:
            continue                       # 인기1위가 없는 경주는 제외
        num += sum(tops) / len(tops)
        den += 1
    return (num * 100 / den if den else 0.0), den, len(no_odds)


def diagnose(per):
    tie = sum(1 for v in per.values() if sum(1 for k, _, _, _ in v if k == 1) > 1)
    no_rank = sum(1 for v in per.values() if not [1 for k, _, _, _ in v if k == 1])
    return tie, no_rank


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else 'model'
    d = os.path.join(base, 'model') if os.path.isdir(os.path.join(base, 'model')) else base
    files = [f for f in ('test.parquet', 'valid.parquet', 'train.parquet')
             if os.path.exists(os.path.join(d, f))]
    if not files:
        raise SystemExit(f'parquet 을 찾지 못함: {d}')

    print(f'{"분할":8s} {"경주":>7s} {"동률":>6s} {"무배당":>6s}   '
          f'{"기존 top-1":>10s} {"교정 top-1":>10s}   {"기존 top-3":>10s} {"교정 top-3":>10s}')
    print('-' * 88)
    out = {}
    for f in files:
        per = load(os.path.join(d, f))
        tie, no_rank = diagnose(per)
        o1, n_rows = old_way(per, 2)
        o3, _ = old_way(per, 3)
        f1, n_races, n_odds = fixed(per, 2)
        f3, _, _ = fixed(per, 3)
        name = f.replace('.parquet', '')
        out[name] = (f1, f3)
        print(f'{name:8s} {len(per):>7,} {tie:>6,} {n_odds:>6,}   '
              f'{o1:>9.1f}% {f1:>9.1f}%   {o3:>9.1f}% {f3:>9.1f}%')
        if no_rank:
            print(f'         ※ 인기1위가 없는 경주 {no_rank}개는 집계에서 제외')

    print()
    print('README §5 표에 넣을 값 (test 기준):')
    if 'test' in out:
        t1, t3 = out['test']
        print(f'  | **시장 (인기 1위마)** | **{t1:.1f}%** | {t3:.1f}% |')
    print()
    print('※ 모델 행(인기도 포함/제외 등)은 이미 경주당 1두를 고르므로 재계산 불필요.')
    print('  바뀌는 것은 시장 행뿐이며, 그에 따라 "모델 − 시장" 격차가 줄어든다.')


if __name__ == '__main__':
    main()
