# Submission

The organiser's write-up content list, as a checklist. Tick nothing that has not happened.

## Required

- [ ] Title + one-line claim
- [ ] Tracks + problem statement numbers (3.9 · 3.7 · 3.6 · 3.8, plus 3.1/3.2 as outcomes)
- [ ] Content warning, if any harmful text appears in the samples
- [ ] Abstract, under 500 words
- [ ] **Headline before → after number**
- [ ] Base model + method in one line
- [ ] "For judges: one minute" box
- [ ] Why this problem / real use case
- [ ] **Pre-registered hypotheses** (and the deviations, with reasons)
- [ ] **Threat model [T3]**
- [ ] Related work — papers, not blogs
- [ ] Hackathon kits used, and exactly which features of each
- [ ] Same model before vs after (table)
- [ ] Confidence intervals (95% paired bootstrap) and significance tests
- [ ] Findings as named subsections
- [ ] **What didn't work** — negative results published, not hidden
- [ ] Sample inputs and outputs, with summarised tool-call logs

## Artifacts

- [ ] HF datasets collection (splits, gold routes, per-arm trajectories, eval logs)
- [ ] HF models collection (adapters, metrics, configs) — add a GGUF/4-bit quant if time allows
- [ ] GitHub repo with runnable commands
- [ ] Demo / Space
- [ ] Video or slides

## Documentation

- [ ] Environment requirements (Python 3.12, one 24GB card for training, CPU for the audit)
- [ ] Repo layout and run instructions (what each folder does, the reproduction commands)
- [ ] **Declaration of work done outside the hackathon**

## Then

- [ ] Discussion / what we learned
- [ ] Implications
- [ ] Future work

## House style, non-negotiable

- Lead with an **honest-status block**: one line of claim, then what exists and what does not.
- **Label every unmeasured number as a prediction, inline.** Never a scoreboard a reader
  could mistake for a run.
- Have a **"what is not claimed"** section. Name the claim you are *not* making before a
  reviewer raises it.
- Report the **step-count checkpoint, not the best checkpoint**.
- Never a mean without its **across-seed spread**.
