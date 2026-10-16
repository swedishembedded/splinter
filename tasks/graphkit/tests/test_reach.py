# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.graphkithost import LuaCase

CHAIN = "G('d', 'a>b b>c c>d a>d')"
LOOP = "G('d', 'a>b b>a b>c')"
LATE_HUB = "G('d', '+u +x u>w w>x')"


class T(LuaCase):
    modules = {"reach": "reach"}

    def test_closure_follows_vertex_order_not_arc_order(self):
        self.check(
            "return reach.gkreach_closure(G('d', 'c>d b>c a>b'))", "{a=[c,d,b],b=[c,d],c=[d],d=[]}")

    def test_pairs_flatten_the_closure(self):
        self.check(
            f"return reach.gkreach_pairs({CHAIN})",
            "[{from=a,to=b},{from=a,to=c},{from=a,to=d},{from=b,to=c},{from=b,to=d},{from=c,to=d}]")

    def test_pair_count_totals_reachable_pairs(self):
        self.check(f"return reach.gkreach_pair_count({CHAIN})", "6")

    def test_pair_count_of_cycle_counts_self_pairs(self):
        self.check(f"return reach.gkreach_pair_count({LOOP})", "6")

    def test_cycle_start_is_in_its_own_reach(self):
        self.check(f"return reach.gkreach_reachable_from({LOOP}, 'a')", "[a,b,c]")

    def test_reachable_from_unknown_vertex_fails(self):
        self.check(f"return fail(reach.gkreach_reachable_from, {CHAIN}, 'q')", "unknown vertex: q")

    def test_reachable_from_matches_closure_row(self):
        self.check(
            "local g = G('d', 'a>b a>c b>d c>d d>e f>a')\n"
            "local closure, ok = reach.gkreach_closure(g), true\n"
            "for _, v in ipairs(g.order) do\n"
            "  ok = ok and show(reach.gkreach_reachable_from(g, v)) == show(closure[v])\n"
            "end\n"
            "return ok", "true")

    def test_can_reach_forward_but_not_back(self):
        self.check(
            f"return {{reach.gkreach_can_reach({CHAIN}, 'a', 'd'), reach.gkreach_can_reach({CHAIN}, 'd', 'a')}}",
            "[true,false]")

    def test_vertex_on_cycle_reaches_itself(self):
        self.check(f"return reach.gkreach_can_reach({LOOP}, 'a', 'a')", "true")

    def test_can_reach_rejects_unknown_target(self):
        self.check(f"return fail(reach.gkreach_can_reach, {CHAIN}, 'a', 'q')", "unknown vertex: q")

    def test_can_reach_rejects_unknown_origin(self):
        self.check(f"return fail(reach.gkreach_can_reach, {CHAIN}, 'p', 'a')", "unknown vertex: p")

    def test_ancestors_list_everything_that_leads_here(self):
        self.check(f"return reach.gkreach_ancestors({CHAIN}, 'd')", "[a,b,c]")

    def test_ancestors_include_the_vertex_on_a_cycle(self):
        self.check(f"return reach.gkreach_ancestors({LOOP}, 'b')", "[a,b]")

    def test_ancestors_of_unknown_vertex_fail(self):
        self.check(f"return fail(reach.gkreach_ancestors, {CHAIN}, 'q')", "unknown vertex: q")

    def test_one_way_reachability_is_not_mutual(self):
        self.check(f"return reach.gkreach_mutual({LOOP}, 'a', 'c')", "false")

    def test_reverse_one_way_reachability_is_not_mutual(self):
        self.check(f"return reach.gkreach_mutual({LOOP}, 'c', 'a')", "false")

    def test_reduction_drops_implied_arcs(self):
        self.check(f"return reach.gkreach_reduction({CHAIN})", "[{from=a,to=b},{from=b,to=c},{from=c,to=d}]")

    def test_reduction_keeps_both_sides_of_a_diamond(self):
        self.check(
            "return reach.gkreach_reduction(G('d', 'a>b a>c b>d c>d a>d'))",
            "[{from=a,to=b},{from=a,to=c},{from=b,to=d},{from=c,to=d}]")

    def test_reduction_keeps_unrelated_arcs(self):
        self.check("return reach.gkreach_reduction(G('d', 'a>b c>d'))", "[{from=a,to=b},{from=c,to=d}]")

    def test_reduction_rejects_cyclic_graph_input(self):
        self.check(f"return fail(reach.gkreach_reduction, {LOOP})", "reduction needs an acyclic graph")

    def test_reduction_rejects_self_loop(self):
        self.check("return fail(reach.gkreach_reduction, G('d', 'a>a'))", "reduction needs an acyclic graph")
