# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.graphkithost import LuaCase

SQUARE = "G('u', 'a>b b>c c>d d>a')"
TRIANGLE = "G('u', 'a>b b>c c>a')"
NEEDS_UNDIRECTED = "bipartite check needs an undirected graph"


def ring(size):
    return (
        f"local parts = {{}}\nfor i = 1, {size} do parts[#parts + 1] = 'v' .. i .. '>v' .. (i % {size} + 1) end\n"
        "local g = G('u', table.concat(parts, ' '))\n"
    )


class T(LuaCase):
    modules = {"bip": "bipartite"}

    def test_colouring_alternates_along_edges(self):
        self.check(f"return bip.gkbip_colouring({SQUARE})", "{a=1,b=2,c=1,d=2}")

    def test_colouring_starts_each_component_with_one(self):
        self.check("return bip.gkbip_colouring(G('u', 'a>b c>d +e'))", "{a=1,b=2,c=1,d=2,e=1}")

    def test_colouring_of_odd_cycle_is_nil(self):
        self.check(f"return tostring(bip.gkbip_colouring({TRIANGLE}))", "nil")

    def test_colouring_refuses_directed_graph(self):
        self.check("return fail(bip.gkbip_colouring, G('d', 'a>b'))", NEEDS_UNDIRECTED)

    def test_long_odd_ring_is_not_bipartite(self):
        self.check(ring(201) + "return bip.gkbip_is_bipartite(g)", "false")

    def test_self_loop_breaks_bipartiteness(self):
        self.check("return bip.gkbip_is_bipartite(G('u', 'a>b b>b'))", "false")

    def test_bipartite_check_refuses_directed_graph(self):
        self.check("return fail(bip.gkbip_is_bipartite, G('d', 'a>b'))", NEEDS_UNDIRECTED)

    def test_conflict_names_the_clashing_edge(self):
        self.check(f"return bip.gkbip_conflict({TRIANGLE})", "[b,c]")

    def test_conflict_of_self_loop_repeats_vertex(self):
        self.check("return bip.gkbip_conflict(G('u', 'a>b b>b'))", "[b,b]")

    def test_conflict_refuses_directed_graph(self):
        self.check("return fail(bip.gkbip_conflict, G('d', 'a>b'))", NEEDS_UNDIRECTED)

    def test_sides_split_by_colour(self):
        self.check(f"return {{bip.gkbip_sides({SQUARE})}}", "[[a,c],[b,d]]")

    def test_sides_follow_vertex_order(self):
        self.check("return {bip.gkbip_sides(G('u', 'b>a c>b'))}", "[[b],[a,c]]")

    def test_sides_of_odd_cycle_are_nil(self):
        self.check(f"return tostring(bip.gkbip_sides({TRIANGLE}))", "nil")

    def test_sides_of_empty_graph_are_empty(self):
        self.check("return {bip.gkbip_sides(G('u', ''))}", "[[],[]]")

    def test_matching_pairs_a_path(self):
        self.check(
            "return bip.gkbip_matching(G('u', 'a>b b>c c>d'))", "[{left=a,right=b},{left=c,right=d}]")

    def test_matching_reroutes_earlier_pairs(self):
        self.check(
            "return bip.gkbip_matching(G('u', 'x>p x>q y>p'))", "[{left=y,right=p},{left=x,right=q}]")

    def test_matching_in_complete_bipartite_graph(self):
        self.check(
            "return bip.gkbip_matching(G('u', 'a>x a>y a>z b>x b>y b>z c>x c>y c>z'))",
            "[{left=c,right=x},{left=b,right=y},{left=a,right=z}]")

    def test_matching_rejects_odd_cycle(self):
        self.check(f"return fail(bip.gkbip_matching, {TRIANGLE})", "graph is not bipartite")

    def test_matching_size_of_even_ring_is_half(self):
        self.check(ring(40) + "return bip.gkbip_matching_size(g)", "20")

    def test_matching_size_limited_by_smaller_side(self):
        self.check(
            "return bip.gkbip_matching_size(G('u', 'a>x b>x c>x d>x'))", "1")
