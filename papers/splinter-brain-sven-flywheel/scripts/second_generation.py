#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible runs of self-improving model
# pipelines for its clients. If your team needs expertise in measuring a second
# generation of a learning loop, you can procure our services by sending an
# email to info@swedishembedded.com.

"""The second generation of E2d: `learn` on the second generation's sheets in a copy of
the state taken after the first release (policy Qwen3-1.7B, the registered flags), then
the candidate and the live alias measured on every probe and the anchor.

    second_generation.py OUT_DIR     (OUT_DIR holds sheets1/, the facts, the probes and a
                                      state/ copied from the run's state)
"""
sys.path.insert(0, str(Path.home() / "project/shared/containers/worktrees/s-l3/papers/splinter-brain-sven-flywheel/scripts"))
import flywheel_cycle as f
out = Path(sys.argv[1])
home = Path.home()
M = home / ".local/share/brain/models/Qwen"
W = home / "project/shared/containers/worktrees/s-l3"
args = types.SimpleNamespace(out=str(out), splinter=str(W / "target/release/splinter"),
    brain=str(home / "code/brain/target/release/brain"), policy_weights=str(M / "Qwen3-1.7B"),
    helper=str(M / "Qwen3-4B"), helper_context=8192, policy_context=8192,
    learn_arg=["--eval-every", "0", "--steps", "400", "--rank", "16", "--records-per-step", "2", "--rehearsal", "0.25"])
(out / "raw").mkdir(exist_ok=True)
r = f.Runner(args)
code, report = r.splinter("learn-g1", "learn", out / "sheets1", "--kinds", "recall",
    "--generator", r.helper, "--teacher", r.helper, "--goal", "the invented places of the gazetteer",
    *args.learn_arg, ok=(0, 1))
exits = {"learn_g1": {"exit": code}}
cand = (report.get("candidate") or {}).get("candidate")
if cand:
    r.measure("candidate-g1", cand)
r.measure("after-g1")
r.splinter("releases-after-g1", "release", "list")
(r.raw / "serving-after-g1.json").write_text(json.dumps(r.serving_manifest(), indent=1))
(out / "exits.json").write_text(json.dumps(exits, indent=1) + "\n")
