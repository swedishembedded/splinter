# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for curricula whose prerequisite order is provably right, for its
# clients. If your team needs expertise in skill graphs and learning curricula then you can procure our services
# by sending an email to info@swedishembedded.com.
"""Oracle `graphs/skill-order`: the order of a skill graph, by NetworkX, for splinter-knowledge's
`SkillGraph::order`.

A skill's prerequisites are edges prerequisite -> skill. Convention (Splinter's, stated in skills.rs): skills are
placed in rounds, each round every skill whose prerequisites are all placed, in the graph's order; that is
networkx.topological_generations with each generation sorted by the skill's position. When the graph has a cycle,
the skills no round places are exactly those on a cycle or reachable from one: every node in a strongly connected
component of more than one node (or with a self-loop), and their descendants, in the graph's order.

Usage: python -I skill_order.py --out FILE [--seed N --cases M]
"""

import argparse
import json
import random

import networkx as nx

NAME = "graphs/skill-order"


def expected(nodes, prereqs):
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    for skill, needs in prereqs.items():
        for need in needs:
            g.add_edge(need, skill)
    position = {n: i for i, n in enumerate(nodes)}
    if nx.is_directed_acyclic_graph(g):
        order = [n for generation in nx.topological_generations(g) for n in sorted(generation, key=position.get)]
        return {"acyclic": True, "order": order}
    cyclic = set()
    for component in nx.strongly_connected_components(g):
        if len(component) > 1 or any(g.has_edge(n, n) for n in component):
            cyclic |= component
    stuck = set(cyclic)
    for n in cyclic:
        stuck |= nx.descendants(g, n)
    return {"acyclic": False, "stuck": sorted(stuck, key=position.get)}


def draw(rng, size, cyclic):
    nodes = [f"s{i}" for i in range(size)]
    rng.shuffle(nodes)
    prereqs = {n: [] for n in nodes}
    rank = {n: i for i, n in enumerate(rng.sample(nodes, len(nodes)))}
    for skill in nodes:
        for need in nodes:
            if rank[need] < rank[skill] and rng.random() < 2.0 / size:
                prereqs[skill].append(need)
    if cyclic and size > 1:
        # Close a cycle: a back edge between two skills, or now and then a skill needing itself.
        a, b = rng.sample(nodes, 2)
        if rng.random() < 0.2:
            prereqs[a].append(a)
        else:
            prereqs[a].append(b)
            prereqs[b].append(a)
    return nodes, prereqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cases", type=int, default=120)
    args = parser.parse_args()
    rng = random.Random(20261124 if args.seed is None else args.seed)
    cases = []
    # The hand-checkable ones first: a chain, a diamond, a two-cycle, a self-loop, a cycle with a dependent.
    fixed = [
        (["c", "b", "a"], {"c": ["b"], "b": ["a"], "a": []}),
        (["d", "b", "c", "a"], {"d": ["b", "c"], "b": ["a"], "c": ["a"], "a": []}),
        (["a", "b"], {"a": ["b"], "b": ["a"]}),
        (["a", "b"], {"a": ["a"], "b": []}),
        (["x", "a", "b", "y"], {"x": [], "a": ["b", "x"], "b": ["a"], "y": ["a"]}),
    ]
    draws = [draw(rng, rng.randint(1, 14), rng.random() < 0.4) for _ in range(args.cases)]
    for nodes, prereqs in fixed + draws:
        skills = [{"id": n, "prerequisites": prereqs[n]} for n in nodes]
        cases.append({"skills": skills, **expected(nodes, prereqs)})
    fixture = {"fixture": NAME, "references": ["networkx"], "cases": cases}
    if args.seed is not None:
        fixture |= {"differential": True, "seed": args.seed}
    with open(args.out, "w") as out:
        out.write(json.dumps(fixture, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
