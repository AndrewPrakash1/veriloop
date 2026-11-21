# NOTION.md — where this project lives in Notion

**Page:** VeriLoop — Hacking-Resistant Post-Training for Compliance Agents
**URL:** https://app.notion.com/p/VeriLoop-Hacking-Resistant-Post-Training-for-Compliance-Agents-41fa9a9820cb81d7b2e0c5ed0b581ae9
**Page id:** `41fa9a98-20cb-81d7-b2e0-c5ed0b581ae9`
**Parent:** `3f3a9a98-20cb-81df-ab44-e414dbf22d6a` (same parent as the CERTAIN page)
**Created:** 2026-11-21

## What the page contains

| Section | Contents |
|---|---|
| Header | one-line claim + an honest-status blockquote |
| 1 | the problem stated as four measurable failures |
| 2 | the one idea: the reward reads the world, never the model's words |
| 3 | **system architecture** — component map (mermaid) + where AgentTune and AuditKIT meet |
| 4 | one episode end to end (mermaid sequence diagram) |
| 5 | the episode contract, with a worked YAML example |
| 6 | the reward written out: `Gamma, M, S, F, E, D, L`, the naive comparison arm, and the evasion table |
| 7 | **self-healing loop** (mermaid) — detect, classify, synthesise, retrain, gate |
| 8 | the metric scoreboard, with directions and sources |
| 9 | the ablation arms and pre-registered contrasts |
| 10 | stack, scope, constraints, event deadlines |
| 11 | threat model |
| 12 | the build plan in one screen, plus the compressed calendar |
| 13 | what is **not** claimed |

The long-form version of the same material lives in `README.md`, `PLAN.md` and
`ARCHITECTURE.md` in this folder — the Notion page is the overview plus architecture.

## Maintenance notes

- The Notion API here is **append-only** for existing content: a block cannot be edited in
  place after it is written. Corrections go in a new appended section (or the block is
  deleted and re-added by hand in the Notion UI).
- Mermaid renders in Notion only inside a code block whose language is `mermaid`. The
  diagram syntax on the page follows the safe-by-construction rules (every node label
  quoted, uppercase node ids, no bare lowercase `end` as text). **Rendering was not
  verified** — eyeball the three diagrams and tell me if any shows an error box.
- When results arrive, the honest-status blockquote at the top is the first thing to
  replace, and §13's last bullet ("not yet true: any trained result") goes with it.
- **Known small drift:** §12's build-plan row says "10 tools"; the tool surface in
  `veriloop/env/tools.py` is **eleven** (ten + the terminal `finish`). Fold the correction
  in whenever that section next gets edited — an append just for this is noise.
- The long-form tree, the ownership table and the fill order live in
  `~/Documents/veriloop/STRUCTURE.md`.
- **§14 (appended 21 Nov 2026) supersedes §6's formula.** The page is append-only, so §6
  still shows the old `0.1·S` term, which is now known to be a bug — see the appended
  section for the correction, the renormalised weights, and the first MEASURED audit table.
  Fold the numbers back into §6 whenever that section is next edited by hand.
