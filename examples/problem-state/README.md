<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Problem state of 2026-11-21

`problems.toml` is the problem state of the experience-to-competence programme on 2026-11-21: six diagnosed problems,
the candidate interventions against each with a prior belief and why, and links to the knowledge entries, ledger items,
results and commits behind them. `verdicts.toml` lists the verdicts the programme had reached by then, in order. Each
verdict that carries an effect is the one the problem state's rule gives for that effect against the problem's target.

Seed a campaign's problem state with the document, record a verdict, and ask for the next hypothesis (the campaign file
names the campaign; its other settings do not matter until an arm is run):

```bash
splinter campaign problems add campaign.toml examples/problem-state/problems.toml
splinter campaign problems update campaign.toml --candidate distill-transfer.v1-thinking --experiment item25-E1b \
  --verdict falsified --reason "pooled +0.016 [-0.009, +0.040]: a +0.05 gain is ruled out" --evidence knowledge:202
splinter campaign next campaign.toml
```

With every verdict of `verdicts.toml` recorded, the next hypothesis is `overthink.concise-traces` (priority 0.55),
which item 25's E2b and item 24b's E2a measure. The campaign's integration specs replay both files and hold that order.
The rules, the schema and how they differ from the paper they adapt are in the user guide's campaign page.
