# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for knowing exactly which passages a retrieval dataset trains on, for its
# clients. If your team needs expertise in validating contrastive training data for search then you can procure our
# services by sending an email to info@swedishembedded.com.
"""Oracle `data/retrieval-triples`: a second, deliberately plain implementation of the retrieval view (plan 6B),
written from its documentation (crates/data/src/views/retrieval.rs and the shared source-text rule of
crates/data/src/views/mod.rs), never from its code.

A case is the view's minimum strength, the contents a source store holds (or does not: a span may name content it
was never given), and experiences in corpus order. Each has an instruction, a verdict (pass, fail, or none) and the
strength it was decided at, and evidence spans: a content and a byte range of it. The expected value is the records
the view writes, each (query, positive, negatives), in order, and what it leaves out, by reason.

The rules (each a sentence of the documentation):
  * an experience without a verdict is left out as Undecided, a failed one as Failed, a pass decided below the
    minimum strength as TooWeak; a decided pass with no evidence as NoEvidence;
  * otherwise each evidence span, in order, is a candidate record whose query is the instruction and whose positive is
    the span's bytes: a span over content the store lacks is left out as SourceMissing, one whose bytes are not UTF-8
    (a range that cuts a character) as NotText;
  * its negatives are the evidence spans of every experience in the corpus, in corpus order and whatever its verdict,
    over the same content and overlapping none of this experience's own spans, resolved (one that cannot be is
    skipped), each text once and never the positive's text.

Usage: python -I retrieval_triples.py --out FILE [--seed N --cases M]
"""

import argparse
import json
import random

NAME = "data/retrieval-triples"
REFERENCE = "reference-projector"
RANK = {"judged": 0, "executable": 1}
TEXTS = [
    "Experiences are immutable. Annotations are appended. Views project them.",
    "The pump runs at 40 L/min. The valve opens at 3.5 bar. The log is kept.",
    "Gr\u00f6\u00dfe and caf\u00e9: a span may cut a character in two.",
]


def draw_case(rng):
    contents = [{"text": t, "stored": rng.random() < 0.85} for t in rng.sample(TEXTS, rng.randint(1, 3))]
    experiences = []
    for e in range(rng.randint(1, 6)):
        spans = []
        for _ in range(rng.choice([0, 1, 1, 2, 3])):
            c = rng.randrange(len(contents))
            size = len(contents[c]["text"].encode("utf-8"))
            start = rng.randrange(size - 1)
            # Short and long ranges at any byte, so spans repeat, overlap and sometimes cut a character.
            end = min(size, start + rng.choice([1, 5, 12, 25, 40]))
            spans.append({"content": c, "start": start, "end": end})
        experiences.append({
            "instruction": f"question {e}",
            "verdict": rng.choice(["pass", "pass", "pass", "fail", "none"]),
            "strength": rng.choice(["executable", "executable", "judged"]),
            "evidence": spans,
        })
    return {"minimum": rng.choice(["executable", "judged"]), "contents": contents, "experiences": experiences}


def resolve(case, span):
    """The span's text, or the reason it has none."""
    content = case["contents"][span["content"]]
    if not content["stored"]:
        return None, "SourceMissing"
    data = content["text"].encode("utf-8")[span["start"]:span["end"]]
    try:
        return data.decode("utf-8"), None
    except UnicodeDecodeError:
        return None, "NotText"


def overlaps(a, b):
    return a["content"] == b["content"] and a["start"] < b["end"] and b["start"] < a["end"]


def project(case):
    excluded = {}

    def leave_out(reason):
        excluded[reason] = excluded.get(reason, 0) + 1

    records = []
    for experience in case["experiences"]:
        if experience["verdict"] == "none":
            leave_out("Undecided")
            continue
        if experience["verdict"] == "fail":
            leave_out("Failed")
            continue
        if RANK[experience["strength"]] < RANK[case["minimum"]]:
            leave_out("TooWeak")
            continue
        own = experience["evidence"]
        if not own:
            leave_out("NoEvidence")
            continue
        for span in own:
            positive, reason = resolve(case, span)
            if positive is None:
                leave_out(reason)
                continue
            negatives = []
            for other_experience in case["experiences"]:
                for other in other_experience["evidence"]:
                    if other["content"] != span["content"] or any(overlaps(mine, other) for mine in own):
                        continue
                    text, _ = resolve(case, other)
                    if text is not None and text != positive and text not in negatives:
                        negatives.append(text)
            records.append({"query": experience["instruction"], "positive": positive, "negatives": negatives})
    return {"records": records, "excluded": excluded}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cases", type=int, default=300)
    args = parser.parse_args()
    # seed: frozen: the golden cases are drawn once, from a fixed seed, and reviewed.
    rng = random.Random(20261125 if args.seed is None else args.seed)
    cases = []
    for _ in range(args.cases):
        case = draw_case(rng)
        cases.append({**case, "view": project(case)})
    fixture = {"fixture": NAME, "references": [REFERENCE], "cases": cases}
    if args.seed is not None:
        fixture["differential"] = True
    with open(args.out, "w") as f:
        json.dump(fixture, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
