<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Choose an acceptance gate

A self-improvement loop proposes a candidate, measures it against the incumbent on its own dev data and keeps one of
the two. The gate that decides is the loop's weak point: a gate that accepts any measured gain also accepts changes
that only look better (noise, and dev items reused until the loop has adapted to them), and a worse change, once
accepted, stays. Splinter has two gates, chosen as a setting of the loop; neither is ever the release gate, which only
an independent exam passes (see [the release policy](../concepts/release-policy.md)).

| | `greedy` (default) | `evidence` |
|---|---|---|
| Accepts when | the candidate wins more dev items than it loses, within the retention margin | a noninferiority e-process over the round's dev items reaches the round's level, within the retention margin |
| Dev items | the whole dev set, every round | a draw per round from the dev set as a pool, fewer than it holds |
| Controls | nothing across rounds; significance is reported | the false evolution rate over all planned rounds |
| On rejection | the incumbent stays | the incumbent stays |

## The evidence gate

The evidence gate tests, each round, whether the candidate is no worse than the incumbent by a tolerance `eps`, and
holds the **false evolution rate** (the expected share of accepted rounds whose candidate was in truth worse by `eps`
or more) at or below a budget `q`, across every round the loop plans. It is the construction of Gao, Qi and Cui,
"Trustworthy recursive self-improvement with statistical guarantees" (arXiv 2611.10787), without its model-guided level
allocation:

- Each dev item gives a paired outcome: `1` when only the candidate passes it, `-1` when only the incumbent does,
  `0` for a tie.
- The evidence is an e-process: a product of bets on the items, averaged over the betting fractions 0.10, 0.25, 0.50
  and 0.75. A tie counts slightly for the candidate when `eps > 0`; a reused item must be more convincing (below).
- A round accepts at the first item at which the evidence reaches `1 / alpha_t`. The test is valid at any stopping
  point, so a loop may grade in batches and stop as soon as it crosses.
- The round's level `alpha_t` comes from what is left of the budget, spread over the remaining planned rounds and
  scaled up by the number of rounds accepted so far. Rounds past the planned horizon test at level zero: nothing is
  graded and the incumbent stays.
- **Overlap correction.** A dev item an earlier round graded may have shaped the candidate (the loop saw it), so the
  bar on it is raised by `beta` times the chance the draw was such an item. Each round draws its items without
  replacement from a pool larger than the draw, and every graded item is remembered as used.

### The settings

| Setting | Meaning | How to choose it |
|---|---|---|
| `eps` | The tolerated loss on the paired difference scale `[-1, 1]` | On the scale of your changes. With 400 dev items of which a few percent differ between candidate and incumbent, the paper's 0.10 makes every tie evidence of noninferiority and accepts almost anything; 0 to 0.02 is the range for such sets |
| `q` | The false evolution rate budget over all planned rounds | 0.1 to 0.3; larger accepts more true gains and more false ones |
| `beta` | The optimism allowed on a reused dev item | Between 0 and `(1 + eps) / 2`; `eps / 2` makes a tie on a fully reused item neutral |
| draw | Dev items per round | Fewer than the pool, so later rounds still see fresh items |

The levels the paper leaves open are parameters of `EvidenceGate` with conservative defaults: no round tests above the
whole budget (`alpha_max = q`), and no round spends more than half of what is left (`rho_max = 0.5`; `rho_min =
0.01`). All four settings are fixed before a loop runs; the guarantee holds for the settings a loop began with, and
a lineage refuses to continue under other ones.

What it costs: a gate that controls false acceptances also accepts fewer true gains on a small dev set. Measured on
the RSI3 lineages' own dev counts, the greedy gate accepted 12 of 12 steps, 2 of which bench's sealed exam found to be no
improvement; the evidence gate at `eps 0.02, beta 0.01, q 0.3` would have accepted about 7.9 of them, 0.13 of them not
improvements, and kept the incumbent at several steps the exam shows were real gains. Run several rounds per
generation instead of one.

## Where to set it

**The Lua lineage** (`samples/campaign-lua`, the `lineage` driver): the budget is planned over `--generations` x
`--rounds`, so give a generation several rounds; the champion stays on a rejected round.

```bash
cargo run --release -p splinter-campaign-lua --bin lineage -- --help
```

```text
--gate evidence --gate-eps 0.02 --gate-q 0.3 --gate-beta 0.01 --dev-draw 150 --dev-batch 32
```

Only drawn dev programs are graded, and the proposer is steered by the champion's results on spent programs alone.
The rounds journal records each round's e-value, level, budget left, exposure and the programs it spent.

**Campaigns** (`splinter campaign`) do not use the evidence gate: a campaign decides each generation with the one
promotion gate on the examiner's signed verdicts, over a sealed slice spent by that decision alone
([Campaigns](campaigns.md)).

**Harness optimisation** (`splinter harness optimise`): `--gate evidence --gate-eps EPS --gate-q Q --gate-beta BETA`
adds the evidence gate over the paired tasks beside the promotion rule (which always decides the gain, not
exam-verified); every evaluation is a round of one
budget planned over `--evaluations`, its level taken before its arm runs. A family that got worse still rejects the
candidate. The frozen task set is reused by every evaluation, so from the second on every task counts as reused.
