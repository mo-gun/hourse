# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is **not an application codebase** — it is a **BMAD Method v6.10.0 workspace** (`_bmad/_config/manifest.yaml`). BMAD is a methodology-driven, agent- and skill-based framework for taking a product from idea → planning → UX → implementation → test. The repository currently contains only the installed framework; all artifact folders (`docs/`, `design-artifacts/*`, `_bmad-output/*`) are empty, meaning no planning or product work has started yet.

- Project name: **주제선정** ("Topic Selection")
- **Output/communication language is Korean** (`document_output_language = "Korean"`, `communication_language = "Korean"`). Write generated documents and user-facing prose in Korean unless the user asks otherwise. Product UI language, however, is English (`wds.product_languages = ["en"]`).

## How work happens here: skills, not scripts

Work is driven by **skills** (surfaced as slash commands, invoked via the `Skill` tool), installed under `.claude/skills/` — 74 `bmad-*`, 13 `wds-*`, plus `sync`/`memory`. There is no build/lint/test-the-app loop because there is no app yet. "Running" this project means invoking the right skill for the current phase.

Two overlapping methodologies are installed; pick based on the user's intent:

- **WDS (Whiteboard-to-Dev)** — the numbered design→dev pipeline whose output maps 1:1 to the `design-artifacts/` folders. Ordered stages:
  - `wds-0-project-setup` → `wds-0-alignment-signoff` → `wds-1-project-brief` (→ `design-artifacts/A-Product-Brief`) → `wds-2-trigger-mapping` (→ `B-Trigger-Map`) → `wds-3-scenarios` (→ `C-UX-Scenarios`) → `wds-4-ux-design` → `wds-7-design-system` (→ `D-Design-System`) → `wds-5-agentic-development` (→ `E-Development`).
  - WDS agents: **Saga** (analyst, `wds-agent-saga-analyst`), **Freya** (UX designer, `wds-agent-freya-ux`), **Mimir** (builder, `wds-agent-mimir-builder`).
- **BMM (core BMAD software-development team)** — classic agile-doc flow: `bmad-product-brief` / `bmad-prd` → `bmad-architecture` → `bmad-create-epics-and-stories` → `bmad-sprint-planning` → `bmad-create-story` → `bmad-dev-story`. Agents: Mary (analyst), John (PM), Sally (UX), Winston (architect), Amelia (dev), Paige (tech writer).

Other installed modules: **cis** (creative-intelligence: brainstorming, design-thinking, storytelling, innovation), **tea** (test architecture — Murat), **bmad-loop** (unattended dev/review loop), **bmb** (agent/skill/module builder). When unsure which skill fits, invoke `bmad-help` — it inspects state and recommends the next skill.

## Where things go (output conventions)

Paths come from `_bmad/config.toml` (`{project-root}` = repo root):

- `_bmad-output/planning-artifacts/` — BMM planning docs (PRD, architecture, epics)
- `_bmad-output/implementation-artifacts/` — BMM stories/implementation
- `_bmad-output/test-artifacts/` — TEA test-design, test-reviews, traceability
- `design-artifacts/A-Product-Brief … E-Development/` — WDS pipeline outputs
- `docs/` — shared project knowledge (BMM `project_knowledge` and WDS `project_knowledge`)

## Configuration model — do not hand-edit generated config

Config resolves through a **four-layer TOML merge** (highest priority last), via `_bmad/scripts/resolve_config.py`:

1. `_bmad/config.toml` — installer-owned team config **(read-only; regenerated on every install)**
2. `_bmad/config.user.toml` — installer-owned personal config **(read-only)**
3. `_bmad/custom/config.toml` — human-authored team overrides (committed)
4. `_bmad/custom/config.user.toml` — human-authored personal overrides (gitignored)

To change any config value durably, edit the `_bmad/custom/*` files — **never** the top-level `_bmad/config.*` files, whose edits are wiped on the next install. The installer-managed files carry this warning in their own headers.

Helper scripts (`_bmad/scripts/`, Python **3.11+** via `uv run` or `python3`, stdlib-only):
- `resolve_config.py --project-root <abs> [--key <section>]` — emit merged config as JSON
- `resolve_customization.py` — resolve per-skill `customize.toml` overrides
- `memlog.py` — memory logging

## Working conventions

- To adjust an agent's behavior or a skill's defaults, prefer that skill's `customize.toml` (see `bmad-customize`) over editing skill bodies — skills are updated by the installer.
- The `.claude/skills/` tree is installed framework content. Treat it as vendored: don't refactor it to "clean it up"; modify behavior through customization files.
