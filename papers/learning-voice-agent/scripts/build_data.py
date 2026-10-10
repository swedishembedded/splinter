#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements voice agents that learn how to speak from
# their users, with every claim backed by a reproducible measurement, for its
# clients. If your team needs expertise in evaluating spoken dialogue systems,
# you can procure our services by sending an email to info@swedishembedded.com.

"""Builds everything the paper's tables and macros show from data/.

Nothing is typed by hand into the paper. data/runs.csv names every measured
run and the file holding its report; data/claims.csv holds the status of each
claim and the run that supports it. A claim may be marked `met` only with an
evidence run that exists, and a claim with no evidence prints as pending: an
unmeasured number is absent, never zero. Standard library only.
"""

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "generated"
STATUSES = {"pending", "partial", "met", "failed", "limitation"}


def rows(name):
    with open(DATA / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fail(message):
    sys.exit(f"build_data: {message}")


def tex(text):
    """`text` safe in running LaTeX."""
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"), ("<=", "$\\leq$"), (">=", "$\\geq$")):
        text = text.replace(a, b)
    return text


def spread(report, key):
    column = report.get(key)
    if not column or "p50" not in column or "p95" not in column:
        fail(f"a latency report lacks {key}")
    return column["p50"], column["p95"]


def wilson(k, n, z=1.96):
    """The Wilson score interval of k successes in n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (c - h) / d, (c + h) / d


def main():
    runs = {r["id"]: r for r in rows("runs.csv")}
    for run in runs.values():
        if not (ROOT / run["file"]).is_file():
            fail(f"run {run['id']}: {run['file']} does not exist")

    claims = rows("claims.csv")
    for claim in claims:
        status, evidence = claim["status"], claim["evidence"]
        if status not in STATUSES:
            fail(f"claim {claim['id']}: status {status!r} is not one of {sorted(STATUSES)}")
        if status in {"met", "partial"} and evidence not in runs:
            fail(f"claim {claim['id']} is {status} but its evidence {evidence!r} is not a run in runs.csv")
        if status == "pending" and evidence:
            fail(f"claim {claim['id']} is pending yet names evidence")

    OUT.mkdir(exist_ok=True)
    macros = {}

    # Claims table.
    lines = ["\\begin{tabular}{@{}l p{6.2cm} l p{4.6cm}@{}}", "\\toprule", "Claim & What is claimed & Status & Gate\\\\", "\\midrule"]
    for claim in claims:
        lines.append(f"{claim['id']} & {tex(claim['claim'])} & {claim['status']} & {tex(claim['gate'])}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "claims_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # First-sound latency, from the stored run.
    run = runs["latency-stream-20261009"]
    report = json.loads((ROOT / run["file"]).read_text(encoding="utf-8"))
    n = report["turns"]
    if str(n) != run["n"]:
        fail(f"latency run: the report holds {n} turns, runs.csv says {run['n']}")
    table = ["\\begin{tabular}{@{}l r r@{}}", "\\toprule", "Stage (seconds from the end of the question) & p50 & p95\\\\", "\\midrule"]
    for label, key in (("first word written", "first_text_seconds"), ("first piece handed to be spoken", "first_piece_seconds"), ("first sound", "first_audio_seconds"), ("whole turn", "total_seconds")):
        p50, p95 = spread(report, key)
        table.append(f"{label} & {p50:.2f} & {p95:.2f}\\\\")
        macros[{"first_audio_seconds": "FirstSound", "first_text_seconds": "FirstWord", "first_piece_seconds": "FirstPiece", "total_seconds": "WholeTurn"}[key] + "Pfifty"] = f"{p50:.2f}"
        macros[{"first_audio_seconds": "FirstSound", "first_text_seconds": "FirstWord", "first_piece_seconds": "FirstPiece", "total_seconds": "WholeTurn"}[key] + "Pninetyfive"] = f"{p95:.2f}"
    table += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "latency_table.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    macros["LatencyTurns"] = str(n)
    macros["LatencyGpu"] = tex(run["gpu"])
    macros["LatencyRunId"] = tex(run["id"])

    # Pilot: the recogniser as a judge of pronunciation. Counted here, not by hand.
    pilot = runs["pilot-names-20261009"]
    items = json.loads((ROOT / pilot["file"]).read_text(encoding="utf-8"))["items"]
    if str(len(items)) != pilot["n"]:
        fail(f"pilot run: the report holds {len(items)} sentences, runs.csv says {pilot['n']}")

    def words(text):
        return "".join(c.lower() if c.isalnum() or c == " " else " " for c in text).split()

    differing = [i for i in items if words(i["text"]) != words(i["heard"])]
    empty = [i for i in differing if not i["heard"].strip()]
    macros["PilotSentences"] = str(len(items))
    macros["PilotDiffering"] = str(len(differing))
    macros["PilotEmpty"] = str(len(empty))
    macros["PilotNonEmptyDiffering"] = str(len(differing) - len(empty))

    # Pilot: a tail of silence before recognition, two recognisers.
    takes = runs["takes-pilot-20261009"]
    summary = json.loads((ROOT / takes["file"]).read_text(encoding="utf-8"))
    if str(summary["takes"]) != takes["n"]:
        fail(f"takes run: the summary holds {summary['takes']} takes, runs.csv says {takes['n']}")
    first, second = "nvidia/nemotron-3.5-asr-streaming-0.6b", "Qwen/Qwen3-ASR-1.7B"
    macros["TakesN"] = str(summary["takes"])
    macros["TakesAgreed"] = str(summary["agreed"])
    macros["TakesContested"] = str(summary["contested"])
    for key, name in (("FirstRecogniser", first), ("SecondRecogniser", second)):
        macros[f"{key}Raw"] = str(summary["differing_raw"][name])
        macros[f"{key}Tailed"] = str(summary["differing_tailed"][name])

    # Machinery gate: the policy learns a planted reward and generalizes to held-out sentences.
    gate = runs["rl-speed-gate-20261010"]
    report = json.loads((ROOT / gate["file"]).read_text(encoding="utf-8"))
    if str(report["held_out_sentences"]) != gate["n"]:
        fail(f"gate run: the report holds {report['held_out_sentences']} held-out sentences, runs.csv says {gate['n']}")
    first_eval, last_eval = report["evals"][0], report["evals"][-1]
    macros["GateUpdates"] = str(last_eval["after_updates"])
    macros["GateGroup"] = str(report["group"])
    macros["GateFramesBefore"] = f"{first_eval['mean_frames_per_word']:.2f}"
    macros["GateFramesAfter"] = f"{last_eval['mean_frames_per_word']:.2f}"
    macros["GateWerBefore"] = f"{first_eval['word_error_rate']:.3f}"
    macros["GateWerAfter"] = f"{last_eval['word_error_rate']:.3f}"
    macros["GateKlLast"] = f"{report['steps'][-1]['mean_kl']:.3f}"

    # The same planted reward and budget under three training methods.
    methods = [("GRPO (group advantages)", "rl-speed-gate-20261010"), ("DPO (best vs worst take)", "rl-speed-dpo-20261010"), ("Best-of-n (imitate the best take)", "rl-speed-best-of-n-20261010")]
    lines = ["\\begin{tabular}{@{}l r r r r@{}}", "\\toprule", "Method & Frames/word before & after & WER before & after\\\\", "\\midrule"]
    for label, rid in methods:
        ev = json.loads((ROOT / runs[rid]["file"]).read_text(encoding="utf-8"))["evals"]
        lines.append(f"{label} & {ev[0]['mean_frames_per_word']:.2f} & {ev[-1]['mean_frames_per_word']:.2f} & {ev[0]['word_error_rate']:.3f} & {ev[-1]['word_error_rate']:.3f}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "methods_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Teaching: facts, rules and procedures, untaught against taught, on the pre-registered test split.
    teach = json.loads((ROOT / runs["teaching-test-20261010"]["file"]).read_text(encoding="utf-8"))
    kinds = {}
    for r in teach["results"]:
        k = kinds.setdefault(r["kind"], {"scenarios": 0, "asks": 0, "untaught": 0, "taught": 0, "understood": 0, "teachings": 0})
        k["scenarios"] += 1
        k["understood"] += r["understood"]
        k["teachings"] += r["teachings"]
        for o in r["outcomes"]:
            k["asks"] += 1
            k["untaught"] += o["untaught_passes"]
            k["taught"] += o["taught_passes"]
    labels = {"fact": "a fact", "correction": "a corrected fact", "indirect": "a fact that must be used", "scale": "sixteen facts at once",
              "procedure": "a procedure", "rule": "a rule", "rules2": "two rules at once", "retract": "taught, then forgotten", "unrelated": "unrelated questions"}
    lines = ["\\begin{tabular}{@{}l r r r l@{}}", "\\toprule", "Taught & Scenarios & Asks & Untaught passes & Taught passes (95\\% interval)\\\\", "\\midrule"]
    for key, label in labels.items():
        k = kinds[key]
        lo, hi = wilson(k["taught"], k["asks"])
        lines.append(f"{label} & {k['scenarios']} & {k['asks']} & {k['untaught']} & {k['taught']} ({lo:.2f}--{hi:.2f})\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "teaching_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    macros["TeachScenarios"] = str(len(teach["results"]))
    macros["TeachAsks"] = str(sum(k["asks"] for k in kinds.values()))
    macros["TeachUnderstood"] = str(sum(k["understood"] for k in kinds.values()))
    macros["TeachTeachings"] = str(sum(k["teachings"] for k in kinds.values()))
    macros["TeachFactTaught"] = f"{kinds['fact']['taught']}/{kinds['fact']['asks']}"
    macros["TeachFactUntaught"] = f"{kinds['fact']['untaught']}/{kinds['fact']['asks']}"
    macros["TeachProcedureLow"] = f"{wilson(kinds['procedure']['taught'], kinds['procedure']['asks'])[0]:.2f}"

    # The pre-registered hypotheses of the teaching test, each with its verdict.
    low = lambda k: wilson(kinds[k]["taught"], kinds[k]["asks"])[0]
    rate = lambda k, arm: kinds[k][arm] / kinds[k]["asks"]
    def verdict(ok):
        return "met" if ok else "not met"
    hyp = [
        ("H1 fact", f"taught {kinds['fact']['taught']}/{kinds['fact']['asks']}, lower bound {low('fact'):.2f}; untaught {kinds['fact']['untaught']}/{kinds['fact']['asks']}", low("fact") >= 0.80 and rate("fact", "untaught") < 0.20),
        ("H1 corrected fact", f"taught {kinds['correction']['taught']}/{kinds['correction']['asks']}, lower bound {low('correction'):.2f}; untaught {kinds['correction']['untaught']}/{kinds['correction']['asks']}", low("correction") >= 0.80 and rate("correction", "untaught") < 0.20),
        ("H1 procedure", f"taught {kinds['procedure']['taught']}/{kinds['procedure']['asks']}, lower bound {low('procedure'):.2f}", low("procedure") >= 0.80 and rate("procedure", "untaught") < 0.20),
        ("H2 forgotten stays forgotten", f"{kinds['retract']['taught']}/{kinds['retract']['asks']} (untaught {kinds['retract']['untaught']})", rate("retract", "taught") >= 0.95 and kinds["retract"]["taught"] >= kinds["retract"]["untaught"] - 1),
        ("H2 unrelated answers unchanged", f"{kinds['unrelated']['taught']}/{kinds['unrelated']['asks']} (untaught {kinds['unrelated']['untaught']})", rate("unrelated", "taught") >= 0.95 and kinds["unrelated"]["taught"] >= kinds["unrelated"]["untaught"] - 1),
        ("H3 restated for confirmation", f"{sum(k['understood'] for k in kinds.values())}/{sum(k['teachings'] for k in kinds.values())}", sum(k["understood"] for k in kinds.values()) / sum(k["teachings"] for k in kinds.values()) >= 0.95),
    ]
    lines = ["\\begin{tabular}{@{}l l l@{}}", "\\toprule", "Pre-registered hypothesis & Result & Verdict\\\\", "\\midrule"]
    lines += [f"{h} & {r} & {verdict(ok)}\\\\" for h, r, ok in hyp]
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "hypotheses_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    macros["TeachFactScenarioLow"] = f"{wilson(kinds['fact']['scenarios'], kinds['fact']['scenarios'])[0]:.2f}"
    macros["TeachRulesAsks"] = str(kinds["rule"]["asks"])
    macros["TeachProcedureAsks"] = str(kinds["procedure"]["asks"])
    macros["TeachRulesTwoTaught"] = f"{kinds['rules2']['taught']}/{kinds['rules2']['asks']}"

    # The same test split on three sizes of the model.
    def kind_rates(report):
        out = {}
        for r in report["results"]:
            k = out.setdefault(r["kind"], [0, 0])
            for o in r["outcomes"]:
                k[0] += o["taught_passes"]
                k[1] += 1
        return out
    sizes = [("Qwen3-4B", "teaching-test-4b-20261010"), ("Qwen3-8B", "teaching-test-20261010"), ("Qwen3-14B", "teaching-test-14b-20261010")]
    rates = {name: kind_rates(json.loads((ROOT / runs[rid]["file"]).read_text(encoding="utf-8"))) for name, rid in sizes}
    cols = [("fact", "fact"), ("correction", "corrected fact"), ("indirect", "used fact"), ("scale", "16 facts"), ("procedure", "procedure"), ("rule", "rule"), ("rules2", "two rules"), ("retract", "forgotten"), ("unrelated", "unrelated")]
    lines = ["\\begin{tabular}{@{}l" + " r" * len(cols) + "@{}}", "\\toprule", "Model & " + " & ".join(c[1] for c in cols) + "\\\\", "\\midrule"]
    for name, _ in sizes:
        lines.append(name + " & " + " & ".join(f"{rates[name][k][0]}/{rates[name][k][1]}" for k, _ in cols) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "models_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # The same test split with every user turn said aloud and recognised, four voices.
    scale = json.loads((ROOT / runs["teaching-spoken-scale-20261010"]["file"]).read_text(encoding="utf-8"))
    sk = {}
    for r in scale["results"]:
        k = sk.setdefault(r["kind"], {"asks": 0, "taught": 0, "exact_asks": 0, "exact_taught": 0, "teachings": 0, "misheard": 0})
        k["teachings"] += r["teachings"]
        k["misheard"] += len(r["misheard"])
        for o in r["outcomes"]:
            k["asks"] += 1
            k["taught"] += o["taught_passes"]
            if not r["misheard"]:
                k["exact_asks"] += 1
                k["exact_taught"] += o["taught_passes"]
    lines = ["\\begin{tabular}{@{}l r r r r@{}}", "\\toprule", "Taught & Asks & Passes, text (one run) & Passes, spoken (four voices) & of which, teaching heard as said\\\\", "\\midrule"]
    for key, label in labels.items():
        k = sk[key]
        exact = f"{k['exact_taught']}/{k['exact_asks']}" if k["exact_asks"] else "none"
        lines.append(f"{label} & {k['asks']} & {kinds[key]['taught']}/{kinds[key]['asks']} & {k['taught']}/{k['asks']} & {exact}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "spoken_scale_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    macros["SpokenFactPass"] = f"{sk['fact']['taught']}/{sk['fact']['asks']}"
    macros["SpokenFactLow"] = f"{wilson(sk['fact']['taught'], sk['fact']['asks'])[1]:.2f}"
    macros["SpokenFactMisheard"] = f"{sk['fact']['misheard']}/{sk['fact']['teachings']}"
    macros["SpokenRulePass"] = f"{sk['rule']['taught']}/{sk['rule']['asks']}"
    macros["SpokenRuleExact"] = f"{sk['rule']['exact_taught']}/{sk['rule']['exact_asks']}"
    macros["SpokenRetractPass"] = f"{sk['retract']['taught']}/{sk['retract']['asks']}"
    macros["SpokenUnrelatedPass"] = f"{sk['unrelated']['taught']}/{sk['unrelated']['asks']}"

    # Procedures and rules, the second set, with enough asks to meet the gate.
    more = json.loads((ROOT / runs["teaching-more-20261010"]["file"]).read_text(encoding="utf-8"))
    scen = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "data/teaching/scenarios-more-test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    groups = {"procedures": [0, 0, 0], "rules to do something": [0, 0, 0], "rules to avoid a word": [0, 0, 0]}
    for r in more["results"]:
        if r["kind"] == "procedure":
            g = groups["procedures"]
        elif scen[r["id"]]["tests"][0]["check"].get("none_of"):
            g = groups["rules to avoid a word"]
        else:
            g = groups["rules to do something"]
        for o in r["outcomes"]:
            g[0] += 1
            g[1] += o["untaught_passes"]
            g[2] += o["taught_passes"]
    lines = ["\\begin{tabular}{@{}l r r l l@{}}", "\\toprule", "Taught & Asks & Untaught passes & Taught passes (95\\% interval) & Gate\\\\", "\\midrule"]
    for label, (n, un, tp) in groups.items():
        lo, hi = wilson(tp, n)
        gate = "met" if lo >= 0.80 and un / n < 0.20 else "not met"
        lines.append(f"{label} & {n} & {un} & {tp} ({lo:.2f}--{hi:.2f}) & {gate}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "more_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pooled_rules = [groups["rules to do something"][i] + groups["rules to avoid a word"][i] for i in range(3)]
    macros["MoreRulesPooled"] = f"{pooled_rules[2]}/{pooled_rules[0]}"
    macros["MoreRulesPooledLow"] = f"{wilson(pooled_rules[2], pooled_rules[0])[0]:.2f}"
    macros["MoreProcedures"] = f"{groups['procedures'][2]}/{groups['procedures'][0]}"
    macros["MoreProceduresLow"] = f"{wilson(groups['procedures'][2], groups['procedures'][0])[0]:.2f}"
    macros["MoreDoRules"] = f"{groups['rules to do something'][2]}/{groups['rules to do something'][0]}"
    macros["MoreAvoidRules"] = f"{groups['rules to avoid a word'][2]}/{groups['rules to avoid a word'][0]}"

    # The spoken test split again, with a user who refuses a wrong restatement and says it again.
    rec = json.loads((ROOT / runs["teaching-recovery-20261010"]["file"]).read_text(encoding="utf-8"))
    rk = {}
    for r in rec["results"]:
        k = rk.setdefault(r["kind"], {"asks": 0, "taught": 0, "retries": 0})
        k["retries"] += r["retried"]
        for o in r["outcomes"]:
            k["asks"] += 1
            k["taught"] += o["taught_passes"]
    macros["RecoveryFactPass"] = f"{rk['fact']['taught']}/{rk['fact']['asks']}"
    macros["RecoveryCorrectionPass"] = f"{rk['correction']['taught']}/{rk['correction']['asks']}"
    macros["RecoveryFactRetries"] = str(rk["fact"]["retries"])
    macros["RecoveryCorrectionRetries"] = str(rk["correction"]["retries"])
    macros["RecoveryFactLow"] = f"{wilson(rk['fact']['taught'], rk['fact']['asks'])[1]:.2f}"
    macros["SpokenCorrectionPass"] = f"{sk['correction']['taught']}/{sk['correction']['asks']}"

    # The readers of what a user says, on the test split of the authored utterances.
    gold = [json.loads(l) for l in (ROOT / "data/teaching/utterances.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    readings = {r["id"]: r for r in json.loads((ROOT / runs["teaching-read-20261010"]["file"]).read_text(encoding="utf-8"))}

    def right(g, t):
        if t is None:
            return g["kind"] == "none"
        (name, body), = t.items()
        text = json.dumps(body).lower()
        return name.lower() == g["kind"] and any(w in text for w in g["key"].lower().split("|"))

    readers = {"plain phrases": lambda r: r["plain"], "model": lambda r: r["model"]}
    lines = ["\\begin{tabular}{@{}l r r r@{}}", "\\toprule", "Reader & Teachings found & Fired on a non-teaching & Misread a teaching\\\\", "\\midrule"]
    test = [g for g in gold if g["split"] == "test"]
    pos = [g for g in test if g["kind"] != "none"]
    neg = [g for g in test if g["kind"] == "none"]
    for name, pick in readers.items():
        found = sum(right(g, pick(readings[g["id"]])) for g in pos)
        wrong = sum(pick(readings[g["id"]]) is not None and not right(g, pick(readings[g["id"]])) for g in pos)
        fired = sum(pick(readings[g["id"]]) is not None for g in neg)
        lines.append(f"{name} & {found}/{len(pos)} & {fired}/{len(neg)} & {wrong}\\\\")
        key = {"plain phrases": "Plain", "model": "Model"}[name]
        macros[f"Read{key}Found"] = f"{found}/{len(pos)}"
        macros[f"Read{key}Fired"] = f"{fired}/{len(neg)}"
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "reading_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Teaching by voice: the pre-registered pass conditions of three scripted conversations.
    spoken = json.loads((ROOT / runs["teaching-spoken-20261010"]["file"]).read_text(encoding="utf-8"))
    ends = lambda text, tail: text.lower().rstrip(" .!?").endswith(tail)
    c1, c2, c3 = (spoken[c]["answers"] for c in ("c1", "c2", "c3"))
    passes = {
        "c1": "atlas" in c1[-1].lower() and c1[0].endswith("Is that right?"),
        "c2": ends(c2[2], "over") and c2[3] == "Forgotten." and not ends(c2[4], "over"),
        "c3": "harbor" in c3[-1].lower() and "seal" in c3[-1].lower(),
    }
    macros["SpokenPassed"] = f"{sum(passes.values())} of {len(passes)}"
    macros["SpokenHeardServer"] = re.search(r"called (.+?)\.", c1[0]).group(1)

    # Taking a teaching back inside the conversation that heard it.
    retract = json.loads((ROOT / runs["teaching-retract-20261010"]["file"]).read_text(encoding="utf-8"))
    asks = [o for r in retract["results"] for o in r["outcomes"]]
    macros["RetractPassed"] = f"{sum(o['taught_passes'] for o in asks)}/{len(asks)}"
    macros["RetractLow"] = f"{wilson(sum(o['taught_passes'] for o in asks), len(asks))[0]:.2f}"

    # Speech in without a transcript: the projector against the cascade, on the same held-out spoken questions.
    speech_in = json.loads((ROOT / runs["speech-in-20261009"]["file"]).read_text(encoding="utf-8"))
    macros["SpeechInTurns"] = str(speech_in["listen-adams.json"]["turns"])
    macros["SpeechInF"] = f"{speech_in['listen-adams.json']['mean_word_f1_vs_text_path_answer']:.3f}"
    macros["CascadeF"] = f"{speech_in['cascade-adams.json']['mean_word_f1_vs_text_path_answer']:.3f}"
    macros["SpeechInTopic"] = speech_in["listen-adams.json"]["answers_mentioning_a_topic_word"]
    macros["CascadeTopic"] = speech_in["cascade-adams.json"]["answers_mentioning_a_topic_word"]

    # The same readers on the independently written set.
    gold_b = [json.loads(l) for l in (ROOT / "data/teaching/utterances-b.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    brief = set((ROOT / "data/teaching/utterances-b.brief.txt").read_text(encoding="utf-8").split())
    readings_b = {r["id"]: r for r in json.loads((ROOT / runs["teaching-read-b-20261010"]["file"]).read_text(encoding="utf-8"))}
    lines = ["\\begin{tabular}{@{}l l r r r@{}}", "\\toprule", "Utterances & Reader & Teachings found & Fired on a non-teaching & Misread\\\\", "\\midrule"]
    for label, rows_b in (("all 90", gold_b), ("the 76 not from the brief", [g for g in gold_b if g["id"] not in brief])):
        pos_b = [g for g in rows_b if g["kind"] != "none"]
        neg_b = [g for g in rows_b if g["kind"] == "none"]
        for name, field in (("plain phrases", "plain"), ("model", "model")):
            pick = lambda g: readings_b[g["id"]][field]
            found = sum(right(g, pick(g)) for g in pos_b)
            fired = sum(pick(g) is not None for g in neg_b)
            wrong = sum(pick(g) is not None and not right(g, pick(g)) for g in pos_b)
            lines.append(f"{label} & {name} & {found}/{len(pos_b)} & {fired}/{len(neg_b)} & {wrong}\\\\")
            if label == "all 90":
                macros[f"ReadB{field.capitalize()}Found"] = f"{found}/{len(pos_b)}"
                macros[f"ReadB{field.capitalize()}Fired"] = f"{fired}/{len(neg_b)}"
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "reading_b_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # The revised readers on the third set.
    gold_c = [json.loads(l) for l in (ROOT / "data/teaching/utterances-c.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    readings_c = {r["id"]: r for r in json.loads((ROOT / runs["teaching-read-c-20261010"]["file"]).read_text(encoding="utf-8"))}
    pos_c = [g for g in gold_c if g["kind"] != "none"]
    neg_c = [g for g in gold_c if g["kind"] == "none"]
    lines = ["\\begin{tabular}{@{}l r r r@{}}", "\\toprule", "Reader & Teachings found & Fired on a non-teaching & Misread\\\\", "\\midrule"]
    for name, field in (("plain phrases", "plain"), ("model", "model")):
        found = sum(right(g, readings_c[g["id"]][field]) for g in pos_c)
        fired = sum(readings_c[g["id"]][field] is not None for g in neg_c)
        wrong = sum(readings_c[g["id"]][field] is not None and not right(g, readings_c[g["id"]][field]) for g in pos_c)
        lines.append(f"{name} & {found}/{len(pos_c)} & {fired}/{len(neg_c)} & {wrong}\\\\")
        macros[f"ReadC{field.capitalize()}Found"] = f"{found}/{len(pos_c)}"
        macros[f"ReadC{field.capitalize()}Fired"] = f"{fired}/{len(neg_c)}"
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "reading_c_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Extraction study: the rule-based readers of a user's words, on the frozen test split.
    study = runs["extraction-rules-20261010"]
    readers = json.loads((ROOT / study["file"]).read_text(encoding="utf-8"))["readers"]
    names = {"respell": "pronounce X as Y", "imitate": "talk as follows", "reject": "that sounded wrong", "approve": "that sounded good",
             "retract": "forget that", "confirm_yes": "confirm", "confirm_no": "refuse", "compare_new": "second take better", "compare_old": "first take better"}
    lines = ["\\begin{tabular}{@{}l r r r@{}}", "\\toprule", "Reader & Utterances & Precision & Recall\\\\", "\\midrule"]
    for key, label in names.items():
        t = readers[key]["test"]
        precision = "n/a" if t["precision"] is None else f"{t['precision']:.2f}"
        recall = "n/a" if t["recall"] is None else f"{t['recall']:.2f}"
        lines.append(f"{label} & {t['positives']} & {precision} & {recall}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "extraction_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    macros["ExtractFalsePositives"] = str(sum(r["test"]["false_positives"] for r in readers.values()))
    macros["ExtractTestSize"] = study["n"]

    (OUT / "macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()), encoding="utf-8")


if __name__ == "__main__":
    main()
