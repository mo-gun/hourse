# 이 폴더의 원 위치

`arena/` 의 **정본은 팀 실험 레포** `JeongWon4034/horse-pred-engine` 의 `arena/` 다.
`pipeline/data` 하네스(팀 파케이 + `common.py`)를 전제로 동작하므로 그 레포 안에서 돌아간다.

여기(`mo-gun/hourse`)에 사본을 둔 이유 — 팀 레포에는 **요청이 있을 때만** 올리기로 했고
(2026-09-10 결정), 그 전에 작업이 유실되지 않게 보존한다.

돌리는 법:
```
cd <horse-pred-engine>/pipeline && TEAM_REPO=<...>/S15P21A304 bash sync_dataset.sh
cd ../arena && uv sync
TEAM_HARNESS=<...>/pipeline/data PYTHONUTF8=1 uv run python -m arena.run --seeds 0,1,2
```

★ 두 곳이 갈릴 수 있다. 수정은 팀 레포 쪽에서 하고 여기로 복사하는 방향을 지킬 것.
