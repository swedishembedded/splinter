# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.graphkithost import LuaCase

NET = "G('d', 'a>b a>c b>d c>d d>e c>e')"
LADDER = "G('d', 's>a s>b a>t b>t t>c t>d c>u d>u')"


class T(LuaCase):
    modules = {"hops": "hops"}

    def test_distance_to_self_is_zero(self):
        self.check(f"return hops.gkhops_distance({NET}, 'c', 'c')", "0")

    def test_distance_from_unknown_source_fails(self):
        self.check(f"return fail(hops.gkhops_distance, {NET}, 'q', 'a')", "unknown vertex: q")

    def test_distance_follows_directed_arcs_only(self):
        self.check("return tostring(hops.gkhops_distance(G('d', 'a>b'), 'b', 'a'))", "nil")

    def test_all_maps_every_reachable_vertex(self):
        self.check(f"return hops.gkhops_all({NET}, 'a')", "{a=0,b=1,c=1,d=2,e=2}")

    def test_all_leaves_out_unreachable(self):
        self.check(f"return hops.gkhops_all({NET}, 'd')", "{d=0,e=1}")

    def test_path_lists_vertices_from_source(self):
        self.check(f"return hops.gkhops_path({NET}, 'a', 'e')", "[a,c,e]")

    def test_path_to_unreachable_is_nil(self):
        self.check(f"return tostring(hops.gkhops_path({NET}, 'e', 'a'))", "nil")

    def test_path_prefers_fewer_steps_over_weight(self):
        self.check("return hops.gkhops_path(G('d', 'a>b:1 b>c:1 a>c:50'), 'a', 'c')", "[a,c]")

    def test_path_across_a_long_chain(self):
        self.check(
            "local parts = {}\n"
            "for i = 1, 499 do parts[#parts + 1] = 'v' .. i .. '>v' .. (i + 1) end\n"
            "local route = hops.gkhops_path(G('d', table.concat(parts, ' ')), 'v1', 'v500')\n"
            "return {#route, route[250]}", "[500,v250]")

    def test_farthest_reports_first_discovered_vertex(self):
        self.check(f"return {{hops.gkhops_farthest({NET}, 'a')}}", "[d,2]")

    def test_ring_zero_is_the_source(self):
        self.check(f"return hops.gkhops_ring({NET}, 'a', 0)", "[a]")

    def test_ring_one_lists_direct_neighbours(self):
        self.check(f"return hops.gkhops_ring({NET}, 'a', 1)", "[b,c]")

    def test_ring_two_lists_second_layer(self):
        self.check(f"return hops.gkhops_ring({NET}, 'a', 2)", "[d,e]")

    def test_ring_rejects_negative_steps(self):
        self.check(f"return fail(hops.gkhops_ring, {NET}, 'a', -1)", "steps must be a non-negative number")

    def test_ring_rejects_text_steps(self):
        self.check(f"return fail(hops.gkhops_ring, {NET}, 'a', 'one')", "steps must be a non-negative number")

    def test_reach_count_within_one_step(self):
        self.check(f"return hops.gkhops_reach_count({NET}, 'a', 1)", "2")

    def test_reach_count_includes_exact_boundary(self):
        self.check(f"return hops.gkhops_reach_count({NET}, 'a', 2)", "4")

    def test_reach_count_stops_at_reachable_set(self):
        self.check(f"return hops.gkhops_reach_count({NET}, 'b', 9)", "2")

    def test_path_count_to_direct_neighbour(self):
        self.check(f"return hops.gkhops_count_paths({NET}, 'a', 'b')", "1")

    def test_path_count_along_three_steps(self):
        self.check("return hops.gkhops_count_paths(G('d', 'a>b b>c c>d'), 'a', 'd')", "1")

    def test_path_count_multiplies_through_stages(self):
        self.check(f"return hops.gkhops_count_paths({LADDER}, 's', 'u')", "4")

    def test_path_count_when_unreachable_is_zero(self):
        self.check(f"return hops.gkhops_count_paths({NET}, 'e', 'a')", "0")

    def test_diameter_of_small_network(self):
        self.check(f"return hops.gkhops_diameter({NET})", "2")

    def test_diameter_of_undirected_chain(self):
        self.check("return hops.gkhops_diameter(G('u', 'a>b b>c c>d'))", "3")

    def test_diameter_of_empty_graph_is_zero(self):
        self.check("return hops.gkhops_diameter(G('d', ''))", "0")

    def test_diameter_of_isolated_vertices_is_zero(self):
        self.check("return hops.gkhops_diameter(G('d', '+a +b'))", "0")
