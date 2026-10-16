# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.graphkithost import LuaCase

MAP = "G('d', 'a>b:7 a>c:9 a>f:14 b>c:10 b>d:15 c>d:11 c>f:2 d>e:6 f>e:9')"
TIE = "G('d', 'a>b:1 a>c:1 b>d:1 c>d:1')"


class T(LuaCase):
    modules = {"path": "shortest"}

    def test_zero_weight_arcs_cost_nothing(self):
        self.check("return path.gkpath_distances(G('d', 'a>b:0 b>c:0'), 'a')", "{a=0,b=0,c=0}")

    def test_fractional_weights_add_exactly(self):
        self.check("return path.gkpath_distances(G('d', 'a>b:0.5 b>c:0.25'), 'a')", "{a=0,b=0.5,c=0.75}")

    def test_negative_weight_is_refused(self):
        self.check(
            "return fail(path.gkpath_distances, G('d', 'a>b:-1'), 'a')", "negative weight on edge a>b")

    def test_negative_weight_on_deeper_arc_is_found(self):
        self.check(
            "return fail(path.gkpath_distances, G('d', 'a>b:1 b>c:-2'), 'a')", "negative weight on edge b>c")

    def test_unknown_source_is_refused(self):
        self.check(f"return fail(path.gkpath_distances, {MAP}, 'zz')", "unknown vertex: zz")

    def test_settle_order_is_nearest_first(self):
        self.check(f"return path.gkpath_settle_order({MAP}, 'a')", "[a,b,c,f,d,e]")

    def test_settle_order_skips_unreachable(self):
        self.check("return path.gkpath_settle_order(G('d', 'a>b:1 c>a:1'), 'a')", "[a,b]")

    def test_parents_record_the_cheapest_predecessor(self):
        self.check(f"return path.gkpath_parents({MAP}, 'a')", "{b=a,c=a,d=c,e=f,f=c}")

    def test_equal_cost_keeps_first_parent(self):
        self.check(f"return path.gkpath_parents({TIE}, 'a')", "{b=a,c=a,d=b}")

    def test_cost_reads_one_distance(self):
        self.check(f"return path.gkpath_cost({MAP}, 'a', 'e')", "20")

    def test_cost_of_unreachable_vertex_is_nil(self):
        self.check("return tostring(path.gkpath_cost(G('d', 'a>b:1 +z'), 'a', 'z'))", "nil")

    def test_cost_to_unknown_target_is_refused(self):
        self.check(f"return fail(path.gkpath_cost, {MAP}, 'a', 'nope')", "unknown vertex: nope")

    def test_route_lists_vertices_and_cost(self):
        self.check(f"return {{path.gkpath_route({MAP}, 'a', 'e')}}", "[[a,c,f,e],20]")

    def test_route_breaks_ties_toward_first_parent(self):
        self.check(f"return {{path.gkpath_route({TIE}, 'a', 'd')}}", "[[a,b,d],2]")

    def test_route_to_unreachable_vertex_is_nil(self):
        self.check("return tostring(path.gkpath_route(G('d', 'a>b:1 +z'), 'a', 'z'))", "nil")

    def test_route_to_unknown_target_is_refused(self):
        self.check(f"return fail(path.gkpath_route, {MAP}, 'a', 'nope')", "unknown vertex: nope")

    def test_route_through_a_long_chain(self):
        self.check(
            "local parts = {}\n"
            "for i = 1, 299 do parts[#parts + 1] = 'v' .. i .. '>v' .. (i + 1) .. ':2' end\n"
            "local g = G('d', table.concat(parts, ' '))\n"
            "local route, cost = path.gkpath_route(g, 'v1', 'v300')\n"
            "return {#route, cost}", "[300,598]")

    def test_within_includes_the_limit_itself(self):
        self.check(f"return path.gkpath_within({MAP}, 'a', 11)", "[a,b,c,f]")

    def test_within_zero_keeps_the_source(self):
        self.check(f"return path.gkpath_within({MAP}, 'a', 0)", "[a]")

    def test_within_large_limit_keeps_all_reachable(self):
        self.check(f"return path.gkpath_within({MAP}, 'a', 1000)", "[a,b,c,f,d,e]")

    def test_nearest_takes_closest_others(self):
        self.check(f"return path.gkpath_nearest({MAP}, 'a', 2)", "[b,c]")

    def test_nearest_exactly_all_others(self):
        self.check(f"return path.gkpath_nearest({MAP}, 'a', 5)", "[b,c,f,d,e]")

    def test_nearest_one_below_all_others(self):
        self.check(f"return path.gkpath_nearest({MAP}, 'a', 4)", "[b,c,f,d]")

    def test_farthest_is_last_settled(self):
        self.check(f"return {{path.gkpath_farthest({MAP}, 'a')}}", "[e,20]")

    def test_farthest_of_lonely_source_is_itself(self):
        self.check("return {path.gkpath_farthest(G('d', '+a'), 'a')}", "[a,0]")
