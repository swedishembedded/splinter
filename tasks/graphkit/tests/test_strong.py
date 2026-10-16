# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.graphkithost import LuaCase

MIXED = "G('d', 'a>b b>c c>a c>d d>e e>d f>e')"
NEEDS_DIRECTED = "components need a directed graph"


class T(LuaCase):
    modules = {"scc": "strong"}

    def test_finish_order_ends_with_last_root(self):
        self.check(f"return scc.gkscc_finish_order({MIXED})", "[e,d,c,b,a,f]")

    def test_finish_order_refuses_undirected_graph(self):
        self.check("return fail(scc.gkscc_finish_order, G('u', 'a>b'))", NEEDS_DIRECTED)

    def test_components_group_mutual_reachability(self):
        self.check(f"return scc.gkscc_components({MIXED})", "[[f],[a,b,c],[d,e]]")

    def test_components_refuse_undirected_graph(self):
        self.check("return fail(scc.gkscc_components, G('u', 'a>b'))", NEEDS_DIRECTED)

    def test_long_chain_gives_one_component_per_vertex(self):
        self.check(
            "local parts = {}\n"
            "for i = 1, 3000 do parts[#parts + 1] = 'v' .. i .. '>v' .. (i + 1) end\n"
            "return scc.gkscc_component_count(G('d', table.concat(parts, ' ')))", "3001")

    def test_chain_is_not_strongly_connected(self):
        self.check("return scc.gkscc_is_strongly_connected(G('d', 'a>b'))", "false")

    def test_single_vertex_is_strongly_connected(self):
        self.check("return scc.gkscc_is_strongly_connected(G('d', '+a'))", "true")

    def test_two_loose_vertices_are_not_strongly_connected(self):
        self.check("return scc.gkscc_is_strongly_connected(G('d', '+a +b'))", "false")

    def test_cycle_members_share_a_component(self):
        self.check(f"return scc.gkscc_same_component({MIXED}, 'a', 'c')", "true")

    def test_distinct_components_are_told_apart(self):
        self.check(f"return scc.gkscc_same_component({MIXED}, 'a', 'd')", "false")

    def test_same_component_names_unknown_first_vertex(self):
        self.check(f"return fail(scc.gkscc_same_component, {MIXED}, 'zz', 'a')", "unknown vertex: zz")

    def test_same_component_names_unknown_second_vertex(self):
        self.check(f"return fail(scc.gkscc_same_component, {MIXED}, 'a', 'yy')", "unknown vertex: yy")

    def test_condensation_links_between_components(self):
        self.check(f"return scc.gkscc_condensation({MIXED})", "[{from=2,to=3},{from=1,to=3}]")

    def test_condensation_lists_each_link_once(self):
        self.check(
            "return scc.gkscc_condensation(G('d', 'a>b b>a a>c b>c'))", "[{from=1,to=2}]")

    def test_condensation_of_a_ring_has_no_links(self):
        self.check("return scc.gkscc_condensation(G('d', 'a>b b>a'))", "[]")

    def test_largest_component_has_most_members(self):
        self.check(f"return scc.gkscc_largest({MIXED})", "[a,b,c]")

    def test_largest_prefers_the_first_of_equals(self):
        self.check("return scc.gkscc_largest(G('d', 'a>b b>a c>d d>c'))", "[c,d]")

    def test_cyclic_vertices_exclude_tree_parts(self):
        self.check(f"return scc.gkscc_cyclic_vertices({MIXED})", "[a,b,c,d,e]")

    def test_cyclic_vertices_include_self_loops(self):
        self.check("return scc.gkscc_cyclic_vertices(G('d', 'a>a b>c'))", "[a]")

    def test_cyclic_vertices_ignore_arcs_to_others(self):
        self.check("return scc.gkscc_cyclic_vertices(G('d', 'a>b b>b'))", "[b]")
