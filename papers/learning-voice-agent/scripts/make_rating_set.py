#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements evaluation of synthetic speech that rests on
# blind human judgement, for its clients. If your team needs expertise in
# designing listening tests whose results can be trusted, you can procure
# our services by sending an email to info@swedishembedded.com.

"""Builds one listening session from the takes `voice-loop takes` wrote.

A session is blind A/B trials of two takes of the same sentence in the same
voice. Besides the real pairs it holds repeats (a real pair shown again with the
sides swapped, to measure how consistent the rater is) and controls (a take
against the same take with noise added, which a rater paying attention
prefers cleanly). The audio is copied under anonymous names; the key that says
which take is which, and which trial is a repeat or a control, stays in
key.json and is never put on the page. Standard library only.

    make_rating_set.py TAKES_DIR OUT_DIR --pairs 80 --repeats 10 --controls 10 --seed 1
"""

import argparse
import array
import json
import random
import shutil
import wave
from pathlib import Path


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1, "16-bit mono only"
        return w.getframerate(), array.array("h", w.readframes(w.getnframes()))


def write_wav(path, rate, samples):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())


def noisy(samples, snr_db, rng):
    """`samples` with white noise at `snr_db` below their power."""
    power = sum(s * s for s in samples) / max(1, len(samples))
    sigma = (power / (10 ** (snr_db / 10))) ** 0.5
    out = array.array("h")
    for s in samples:
        v = int(s + rng.gauss(0, sigma))
        out.append(max(-32768, min(32767, v)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("takes")
    ap.add_argument("out")
    ap.add_argument("--pairs", type=int, default=80)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--controls", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--per-sentence", type=int, default=2, help="most pairs drawn from one sentence")
    ap.add_argument("--skip", help="a key.json of an earlier session whose pairs are not drawn again")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    takes_dir, out = Path(args.takes), Path(args.out)
    rows = [json.loads(line) for line in (takes_dir / "manifest.jsonl").read_text().splitlines() if line.strip()]
    by_sentence = {}
    for row in rows:
        if row["agreed"]:
            by_sentence.setdefault(row["sentence"], []).append(row)
    used = set()
    if args.skip:
        for entry in json.loads(Path(args.skip).read_text()):
            used.add(frozenset((entry["a"], entry["b"])))
    sentences = [s for s, t in by_sentence.items() if len(t) >= 2]
    rng.shuffle(sentences)
    capacity = sum(min(args.per_sentence, len(by_sentence[s]) // 2) for s in sentences)
    if capacity < args.pairs:
        raise SystemExit(f"the takes allow {capacity} pairs ({len(sentences)} sentences with two agreed takes); {args.pairs} asked")

    trials = []
    chosen = set()
    for round_ in range(args.per_sentence):
        for s in sentences:
            if len(trials) == args.pairs:
                break
            takes = [t for t in by_sentence[s] if not any(t["id"] in pair for pair in chosen if pair[2] == s)]
            if len(takes) < 2:
                continue
            a, b = rng.sample(takes, 2)
            if frozenset((a["id"], b["id"])) in used:
                continue
            chosen.add((a["id"], b["id"], s))
            trials.append({"kind": "pair", "text": a["text"], "a": a["id"], "b": b["id"], "dir": str(takes_dir)})
    # Repeats: the same pair again with the sides swapped.
    for t in rng.sample(trials, min(args.repeats, len(trials))):
        trials.append({"kind": "repeat", "text": t["text"], "a": t["b"], "b": t["a"], "dir": t["dir"]})
    # Controls: a take against itself with noise added; the noisy side is hidden by position.
    for s in rng.sample(sentences, args.controls):
        t = rng.choice(by_sentence[s])
        trials.append({"kind": "control", "text": t["text"], "a": t["id"], "b": t["id"] + "+noise", "dir": str(takes_dir)})
    rng.shuffle(trials)
    for t in trials:
        if rng.random() < 0.5 and t["kind"] == "control":
            t["a"], t["b"] = t["b"], t["a"]

    (out / "audio").mkdir(parents=True, exist_ok=True)
    public, key = [], []
    for i, t in enumerate(trials, 1):
        trial = f"t{i:03d}"
        files = {}
        for side in ("a", "b"):
            take = t[side]
            name = f"{trial}-{side}.wav"
            if take.endswith("+noise"):
                rate, samples = read_wav(Path(t["dir"]) / f"{take[:-6]}.wav")
                write_wav(out / "audio" / name, rate, noisy(samples, 12.0, rng))
            else:
                shutil.copyfile(Path(t["dir"]) / f"{take}.wav", out / "audio" / name)
            files[side] = f"audio/{name}"
        public.append({"id": trial, "text": t["text"], "a": files["a"], "b": files["b"]})
        key.append({"id": trial, "kind": t["kind"], "a": t["a"], "b": t["b"]})
    (out / "pairs.json").write_text(json.dumps(public, indent=1) + "\n")
    (out / "key.json").write_text(json.dumps(key, indent=1) + "\n")
    print(f"{len(public)} trials ({args.pairs} pairs, {min(args.repeats, args.pairs)} repeats, {args.controls} controls)")


if __name__ == "__main__":
    main()
