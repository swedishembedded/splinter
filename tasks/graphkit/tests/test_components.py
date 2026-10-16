# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.graphkithost import LuaCase

PARTS = "G('u', 'a>b b>c d>e +f')"
NEEDS_UNDIRECTED = "components need an undirected graph"


class T(LuaCase):
    modules = {"comp": "components"}

    def test_components_refuse_directed_graph(self):
        self.check("return fail(comp.gkcomp_components, G('d', 'a>b'))", NEEDS_UNDIRECTED)

    def test_sizes_list_group_lengths(self):
        self.check(f"return comp.gkcomp_sizes({PARTS})", "[3,2,1]")

    def test_component_of_gives_group_index(self):
        self.check(f"return {{comp.gkcomp_component_of({PARTS}, 'a'), comp.gkcomp_component_of({PARTS}, 'e')}}", "[1,2]")

    def test_component_of_last_group(self):
        self.check(f"return comp.gkcomp_component_of({PARTS}, 'f')", "3")

    def test_component_of_unknown_vertex_fails(self):
        self.check(f"return fail(comp.gkcomp_component_of, {PARTS}, 'q')", "unknown vertex: q")

    def test_joined_graph_is_connected(self):
        self.check("return comp.gkcomp_is_connected(G('u', 'a>b b>c'))", "true")

    def test_single_vertex_is_connected(self):
        self.check("return comp.gkcomp_is_connected(G('u', '+a'))", "true")

    def test_two_loose_vertices_are_not_connected(self):
        self.check("return comp.gkcomp_is_connected(G('u', '+a +b'))", "false")

    def test_largest_group_has_most_vertices(self):
        self.check(f"return comp.gkcomp_largest({PARTS})", "[a,b,c]")

    def test_largest_group_may_come_later(self):
        self.check("return comp.gkcomp_largest(G('u', 'a>b c>d d>e'))", "[c,d,e]")

    def test_largest_prefers_first_of_equal_groups(self):
        self.check("return comp.gkcomp_largest(G('u', 'a>b c>d'))", "[a,b]")

    def test_isolated_vertices_have_no_edges(self):
        self.check(f"return comp.gkcomp_isolated({PARTS})", "[f]")

    def test_self_loop_vertex_is_not_isolated(self):
        self.check("return comp.gkcomp_isolated(G('u', 'a>a +b'))", "[b]")

    def test_graph_without_loose_vertices_has_no_isolated(self):
        self.check("return comp.gkcomp_isolated(G('u', 'a>b'))", "[]")

    def test_isolated_refuses_directed_graph(self):
        self.check("return fail(comp.gkcomp_isolated, G('d', '+a'))", NEEDS_UNDIRECTED)

    def test_cycle_has_no_bridge(self):
        self.check("return comp.gkcomp_bridges(G('u', 'a>b b>c c>a'))", "[]")

    def test_tail_on_a_cycle_is_a_bridge(self):
        self.check("return comp.gkcomp_bridges(G('u', 'a>b b>c c>a c>d'))", "[[c,d]]")

    def test_edge_between_two_cycles_is_a_bridge(self):
        self.check(
            "return comp.gkcomp_bridges(G('u', 'a>b b>c c>a c>d d>e e>f f>d'))", "[[c,d]]")

    def test_bridge_search_survives_a_long_path(self):
        self.check(
            "local parts = {}\n"
            "for i = 1, 3000 do parts[#parts + 1] = 'v' .. i .. '>v' .. (i + 1) end\n"
            "return #comp.gkcomp_bridges(G('u', table.concat(parts, ' ')))", "3000")

    def test_bridges_refuse_directed_graph(self):
        self.check("return fail(comp.gkcomp_bridges, G('d', 'a>b'))", NEEDS_UNDIRECTED)

    def test_count_refuses_directed_graph(self):
        self.check("return fail(comp.gkcomp_count, G('d', 'a>b'))", NEEDS_UNDIRECTED)
