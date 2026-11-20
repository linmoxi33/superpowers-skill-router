# superpowers-skill-router

Deterministic routing-reachability audit for Agent Skills: parse every `SKILL.md` description, replay real and derived trigger prompts through a fixed lexical router, and report **which skills collide and which prompt an explicit name mention still loses** — because "a skill fires when it shouldn't, stays silent when it should" is currently diagnosed by hand.

Upstream: https://github.com/obra/superpowers

## Why now

`obra/superpowers` (322k★, MIT) ships 15 skills that are selected **only** by their `description` text. Its own project documents the failure mode and cannot measure it:

- README, "When Something Goes Wrong": *"a skill fires when it shouldn't, **stays silent when it should**"* — https://github.com/obra/superpowers/blob/main/README.md#L329
- #2386 (open, 7 comments): `test-driven-development` **never invoked, 0/3 runs**, even with the bootstrap injection — https://github.com/obra/superpowers/issues/2386
- #2699 (open): the pressure wording **over-triggers** skills — https://github.com/obra/superpowers/issues/2699
- #2685 (open): after plan-mode approval only 2 of 15 skills fire — https://github.com/obra/superpowers/issues/2685
- `AGENTS.md`: *"Without it, the skills are dead weight — present on disk but never invoked."* — https://github.com/obra/superpowers/blob/main/AGENTS.md#L82

The repo's existing gates only check **frontmatter form** (`name` regex, ≤1024 chars, banned words, 1000-word body) and **explicit** name requests (`tests/explicit-skill-requests/`). Nothing computes whether two descriptions compete for the same request, or which skill can never win. This tool closes that measurement gap without touching upstream.

## What it does / does not do

**Does:** parse `name`/`description` (plain, quoted, folded `>`/`|` frontmatter), build an IDF-weighted lexical router over name + description tokens (bigram bonus, `MUST/ALWAYS` strength multiplier, explicit-name boost), run two query families — each skill's own **name-masked trigger clause** and any **real prompt corpus** you pass in — and emit per-query rankings, margins, collisions, unreachable skills and de-confliction term suggestions. Fully offline and deterministic.

**Does not:** model any vendor LLM, call a network, edit skills, or claim that a lexical collision is a guaranteed production misroute. It is a comparable, reproducible proxy — that limitation is the point of the margin column.

## Install

Zero dependencies, Python ≥ 3.9. Nothing to install:

```bash
git clone git@github-linmoxi33:linmoxi33/superpowers-skill-router.git
cd superpowers-skill-router
python3 -m unittest discover -s tests      # 22 tests
python3 bin/skill_router.py --skills examples/in/skills --queries examples/in/real_prompts.jsonl
```

## Usage

```bash
# self-retrieval only (15 derived trigger queries)
python3 bin/skill_router.py --skills examples/in/skills

# add the real prompts vendored from upstream's explicit-skill-requests tests
python3 bin/skill_router.py --skills examples/in/skills --queries examples/in/real_prompts.jsonl --json

# write the full report and list only unreachable skills
python3 bin/skill_router.py --skills examples/in/skills --report /tmp/report.json --only-unreachable
```

Real output on the vendored 15-skill snapshot (see `examples/out/report.json`):

```text
skills=15 queries=24 reachable=15 unreachable=0 collisions=2 mean_margin=0.5774
expected_hits=19/20
digest=sha256:286d5430bf966e09437fdbf68ecb239699234f1a4e95d0f4d7b159381bb4c45b

COLLISIONS (margin < 0.15):
  - subagent-driven-development | subagent-driven-development(21.8742) vs diagnosing-superpowers(18.6066) | Great, the plan is complete...
  - subagent-driven-development | subagent-driven-development(22.1434) vs diagnosing-superpowers(21.1264) | [Previous assistant message]: Plan complete...
```

### Exit codes

| code | meaning |
| --- | --- |
| `0` | no findings (no collision, no unreachable skill) |
| `1` | findings detected (collisions and/or unreachable skills) |
| `2` | usage or input error (missing dir, bad frontmatter, invalid JSONL) |

`--json` (and `--report`) emit a single object whose `digest` is `sha256` of canonical JSON **excluding** `digest`; two runs over the same inputs produce the same digest.

## 本轮创新

Bench ledger has ~23 offline **structural** skill linters and an explicit-name trigger test, but no entry measures **routing reachability**: the first tool to turn "skills don't fire" from an issue anecdote into a per-skill top-1 reachability + margin report, using name-masked self-retrieval queries, an explicit-name boost, and de-confliction term suggestions — reproducible offline to a stable digest (B tier: new measurement, new conclusion; not a re-skin of a frontmatter linter).

## Limitations

- Lexical proxy only: no embeddings, no LLM. It measures **description collision**, not guaranteed host behaviour; margins are relative, not probabilities.
- The 9 real prompts are vendored from upstream's `tests/explicit-skill-requests/` (MIT, © 2025 Jesse Vincent); they only cover explicit naming, which is exactly the blind spot.
- The 15 `examples/in/skills/*/SKILL.md` fixtures are a **name + description frontmatter snapshot** of upstream at commit `HEAD` (2026-11-20) — no upstream body text is copied — included under MIT for offline reproduction; see `LICENSE`.
- `--min-margin` (default `0.15`) is a reporting threshold, not a calibrated probability.
