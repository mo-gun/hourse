# -*- coding: utf-8 -*-
"""신경망 계열 — **앙상블 다양성을 위한** 한 칸.

출처: Plackett-Luce 손실과 경주 패딩은 `pipeline/model`(정원 S1~S4) 의 개념을 같은
규칙으로 다시 구현했다. 구조 자체는 새로울 게 없다(S1·S2 가 이미 같은 계열).

**왜 그런데도 넣는가** — arena 1차 실측에서 트리 계열 12종의 예측 상관이 중앙값 0.943
(최대 0.992)로 나왔다. 계열을 늘려도 같은 경주에서 같이 틀리니 앙상블 이득이 없다.
레포 최고 기록(LGB+S3 35.3)이 GBDT+신경망 조합이었던 것도 같은 방향이다.
그래서 여기 목적은 "이 MLP 가 트리를 이기나"가 아니라 **"트리와 얼마나 다르게 틀리나"** 다.

★ 실측 결과는 예상과 달랐다 — mlp_pl 의 트리 대비 상관은 0.87~0.95 로 **여전히 높다**
  (lgb_lambdarank 와 0.95). 가장 탈상관된 칸은 신경망이 아니라 **타깃을 바꾼**
  `lgb_speedfig`(0.72~0.85)였다. 다양성의 지렛대는 구조가 아니라 타깃 쪽이다.
  단 top-1·logloss 단독 성적은 이 칸이 arena 최고다(34.53 / 1.9078).

레포와 다르게 한 것 하나 — **epoch 을 valid 로 고르지 않는다.**
정원님 장부에 "epoch 선택: valid top-1 최고 epoch … valid 숫자는 그만큼 낙관적이다"
라고 적혀 있다. arena 는 train 의 **마지막 10% 개최일**을 내부 검증으로 떼어 epoch 을
고르고, valid 는 채점에만 쓴다. 그래서 이 칸의 valid 숫자는 낙관 보정이 필요 없다
(대신 레포의 S1~S4 숫자와 직접 비교하면 이쪽이 불리하다 — 조건이 다르다).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import harness as H

HID = (128, 64)
DROPOUT = 0.2
EPOCHS = 20
BATCH_RACES = 256
LR = 3e-3
WD = 1e-2
INNER_FRAC = 0.10          # train 뒤쪽 10% 개최일 = epoch 선택용 내부 검증
# ⚠ 정직하게 — 전처리 통계(중앙값·평균·표준편차·원핫 어휘·빈도)는 **train 전체**에서
#   적합한다. 내부 검증 구간도 그 안에 든다. epoch 선택에 아주 약한 in-sample 이
#   섞이지만 valid 로는 새지 않는다. valid 는 채점에만 쓰고 어떤 통계도 여기서 적합하지
#   않는다 — 그게 이 칸의 valid 숫자에 낙관 보정이 필요 없는 이유다.
MAX_HORSES = 16


def _design(tr: pd.DataFrame, ev: pd.DataFrame, feats: list[str]):
    """수치는 z-score, 저차원 범주형은 원핫, 고차원(F2_sire_id)은 빈도 인코딩.

    전처리 통계는 **train 에서만** 적합한다. 피처를 빼거나 더하지 않는다 —
    F2_sire_id 는 937레벨이라 원핫이 비현실적이어서 빈도로 한 칸에 담는다.
    """
    cat = [c for c in feats if c in H.CATEGORICAL or not pd.api.types.is_numeric_dtype(tr[c])]
    hi = [c for c in cat if tr[c].astype(str).nunique() > 40]
    lo = [c for c in cat if c not in hi]
    num = [c for c in feats if c not in cat]

    blocks_tr, blocks_ev = [], []
    A, B = tr[num].apply(pd.to_numeric, errors="coerce"), ev[num].apply(pd.to_numeric, errors="coerce")
    med = A.median()
    mu, sd = A.fillna(med).mean(), A.fillna(med).std().replace(0, 1.0)
    blocks_tr.append(((A.fillna(med) - mu) / sd).to_numpy(np.float32))
    blocks_ev.append(((B.fillna(med) - mu) / sd).to_numpy(np.float32))
    # 결측 표시 — 트리는 결측을 직접 다루지만 MLP 는 대치하면 정보가 사라진다
    blocks_tr.append(A.isna().to_numpy(np.float32))
    blocks_ev.append(B.isna().to_numpy(np.float32))

    for c in lo:
        lv = sorted(set(tr[c].dropna().astype(str)))
        for d, out in ((tr, blocks_tr), (ev, blocks_ev)):
            v = d[c].astype(str)
            out.append(np.stack([(v == k).to_numpy(np.float32) for k in lv], axis=1)
                       if lv else np.zeros((len(d), 0), np.float32))
    for c in hi:
        freq = tr[c].astype(str).value_counts(normalize=True)
        for d, out in ((tr, blocks_tr), (ev, blocks_ev)):
            out.append(d[c].astype(str).map(freq).fillna(0.0).to_numpy(np.float32)[:, None])
    return np.concatenate(blocks_tr, axis=1), np.concatenate(blocks_ev, axis=1)


def _pack(df: pd.DataFrame, X: np.ndarray):
    """경주 단위 패딩 — (경주, 최대두수, 피처) + 마스크 + 1착 인덱스."""
    q = H.qid(df)
    n = q.max() + 1
    sizes = np.bincount(q, minlength=n)
    assert sizes.max() <= MAX_HORSES, f"두수 {sizes.max()} > {MAX_HORSES}"
    out = np.zeros((n, MAX_HORSES, X.shape[1]), np.float32)
    mask = np.zeros((n, MAX_HORSES), bool)
    win = np.zeros(n, np.int64)
    pos = np.zeros(n, int)
    y = df["y_win"].to_numpy(bool)
    for i, r in enumerate(q):
        out[r, pos[r]] = X[i]
        mask[r, pos[r]] = True
        if y[i]:
            win[r] = pos[r]
        pos[r] += 1
    return out, mask, win, q


def mlp_pl(tr, ev, feats, seed):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    Xtr, Xev = _design(tr, ev, feats)

    # train 뒤쪽 10% 개최일을 내부 검증으로 (시간 순서 유지 — 미래로 고르지 않는다)
    days = np.sort(tr["rcDate"].unique())
    cut = days[int(len(days) * (1 - INNER_FRAC))]
    inner = tr["rcDate"].to_numpy() >= cut
    packs = {}
    for key, m in (("fit", ~inner), ("inner", inner)):
        packs[key] = _pack(tr[m], Xtr[m])
    Pev = _pack(ev, Xev)

    d = Xtr.shape[1]
    layers, prev = [], d
    for h in HID:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(DROPOUT)]
        prev = h
    layers += [nn.Linear(prev, 1)]
    net = nn.Sequential(*layers)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)

    def score(pack, train=False):
        x, mask, win, _ = pack
        xt = torch.from_numpy(x); mt = torch.from_numpy(mask)
        s = net(xt).squeeze(-1)
        return s.masked_fill(~mt, -1e9), torch.from_numpy(win)

    def pl_loss(s, win):
        return nn.functional.cross_entropy(s, win)

    xf, mf, wf, _ = packs["fit"]
    n_races = len(xf)
    best, best_state = np.inf, None
    for ep in range(EPOCHS):
        net.train()
        order = torch.randperm(n_races)          # 경주끼리만 섞는다 (경주 안 순서는 유지)
        for b in range(0, n_races, BATCH_RACES):
            idx = order[b:b + BATCH_RACES]
            s = net(torch.from_numpy(xf[idx])).squeeze(-1)
            s = s.masked_fill(~torch.from_numpy(mf[idx]), -1e9)
            loss = pl_loss(s, torch.from_numpy(wf[idx]))
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            s, w = score(packs["inner"])
            v = float(pl_loss(s, w))
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_state)

    net.eval()
    with torch.no_grad():
        x, mask, _, q = Pev
        s = net(torch.from_numpy(x)).squeeze(-1).numpy()
    flat = np.zeros(len(ev), np.float64)
    pos = np.zeros(q.max() + 1, int)
    for i, r in enumerate(q):
        flat[i] = s[r, pos[r]]
        pos[r] += 1
    return flat
