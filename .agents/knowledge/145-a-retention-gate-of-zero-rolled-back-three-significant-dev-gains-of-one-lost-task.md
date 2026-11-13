<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 145. A retention gate of zero rolled back three significant dev gains over one lost retention task

**Lesson:** In the campaign controller's dev gate a candidate that loses any retention pass is rolled back. On Qwen3-4B
and Lua output prediction (one GH200, 12 steps per arm, lr 1e-5, dev 400 programs, retention 40 programs of levels 1 and
2) every arm of a 4-arm campaign lost exactly one retention task and was rolled back, although the dev gain of three of
them was significant. Allowing two retention losses let the same first arm advance (the dev numbers of arm 1 were
identical in both runs, 27 wins and 8 losses: the seeded run is reproducible).

## Measured

- Zero allowed: dev 0.3375 (base) against 0.385 (27/8, p 0.0009), 0.435 (46/7), 0.395 (32/9, p 0.0002), 0.355 (12/5,
  p 0.07); every arm lost one retention task, every arm was rolled back, the champion stayed the base.
- Two allowed: dev 0.3375, 0.385 (27/8, p 0.0009 at level 0.025), 0.4475 (27/2, p below 1e-5 at 0.0083), 0.475 (11/0,
  p 0.0005 at 0.0042), 0.49 (15/9, p 0.15 at 0.0025: advanced on the point estimate, not significant). 0.65 GPU-hours
  in all; retention losses 1, 0, 0, 0. Informative-group fraction 0.39, 0.12, 0.10, 0.09 as the pool's frontier tasks
  were used up.

## What this does and does not establish

Forty retention programs cannot tell one lost task from noise, so a bound of zero measures the probe's resolution, not
forgetting; it is a design question for the retention set (size it for the loss it must detect, as the dev set is). Not
established: that the gains are transfer (dev programs are other draws of the same generator) or survive a held-out
family; no sealed exam was used or should be. The policy-gradient KL reported by brain for these arms is the artefact of
finding 144 (1e2 to 1e7), so the gain is a measurement of the dev gate, not of a validated trainer.
