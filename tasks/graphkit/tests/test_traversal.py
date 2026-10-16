# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.graphkithost import LuaCase

DIAMOND = "G('d', 'a>b a>c b>d c>d d>e')"
RING = "G('d', 'a>b a>c b>c c>a')"
SQUARE = "G('u', 'a>b a>c b>d c>d')"


class T(LuaCase):
    modules = {"walk": "traversal"}

    def test_breadth_follows_arc_insertion_order(self):
        self.check("return walk.gkwalk_breadth_order(G('d', 'a>c a>b c>e b>d'), 'a')", "[a,c,b,e,d]")

    def test_breadth_never_repeats_a_vertex(self):
        self.check(f"return walk.gkwalk_breadth_order({RING}, 'b')", "[b,c,a]")

    def test_breadth_crosses_undirected_edges(self):
        self.check(f"return walk.gkwalk_breadth_order({SQUARE}, 'd')", "[d,b,c,a]")

    def test_breadth_rejects_unknown_start(self):
        self.check(f"return fail(walk.gkwalk_breadth_order, {DIAMOND}, 'z')", "unknown vertex: z")

    def test_preorder_honours_arc_insertion_order(self):
        self.check("return walk.gkwalk_depth_preorder(G('d', 'a>c a>b c>d'), 'a')", "[a,c,d,b]")

    def test_preorder_on_undirected_square(self):
        self.check(f"return walk.gkwalk_depth_preorder({SQUARE}, 'a')", "[a,b,d,c]")

    def test_preorder_rejects_unknown_start(self):
        self.check(f"return fail(walk.gkwalk_depth_preorder, {DIAMOND}, 'q')", "unknown vertex: q")

    def test_postorder_finishes_deepest_first(self):
        self.check(f"return walk.gkwalk_depth_postorder({DIAMOND}, 'a')", "[e,d,b,c,a]")

    def test_postorder_in_a_cycle(self):
        self.check(f"return walk.gkwalk_depth_postorder({RING}, 'a')", "[c,b,a]")

    def test_postorder_of_self_loop_vertex(self):
        self.check("return walk.gkwalk_depth_postorder(G('d', 'a>a a>b'), 'a')", "[b,a]")

    def test_postorder_rejects_unknown_start(self):
        self.check(f"return fail(walk.gkwalk_depth_postorder, {DIAMOND}, 'q')", "unknown vertex: q")

    def test_levels_stop_at_visited_vertices(self):
        self.check(f"return walk.gkwalk_breadth_levels({RING}, 'a')", "[[a],[b,c]]")

    def test_levels_reject_unknown_start(self):
        self.check(f"return fail(walk.gkwalk_breadth_levels, {DIAMOND}, 'z')", "unknown vertex: z")

    def test_within_zero_keeps_only_start(self):
        self.check(f"return walk.gkwalk_breadth_within({DIAMOND}, 'a', 0)", "[a]")

    def test_within_two_reaches_third_level(self):
        self.check(f"return walk.gkwalk_breadth_within({DIAMOND}, 'a', 2)", "[a,b,c,d]")

    def test_within_large_limit_returns_everything(self):
        self.check(f"return walk.gkwalk_breadth_within({DIAMOND}, 'a', 50)", "[a,b,c,d,e]")

    def test_within_rejects_negative_limit(self):
        self.check(
            f"return fail(walk.gkwalk_breadth_within, {DIAMOND}, 'a', -1)",
            "depth limit must be a non-negative number")

    def test_within_rejects_text_limit(self):
        self.check(
            f"return fail(walk.gkwalk_breadth_within, {DIAMOND}, 'a', 'two')",
            "depth limit must be a non-negative number")

    def test_all_breadth_covers_disconnected_parts(self):
        self.check("return walk.gkwalk_breadth_order_all(G('d', 'a>b c>d +e'))", "[a,b,c,d,e]")

    def test_all_breadth_starts_in_vertex_order(self):
        self.check("return walk.gkwalk_breadth_order_all(G('d', 'b>a c>b d>c'))", "[b,a,c,d]")

    def test_all_depth_covers_disconnected_parts(self):
        self.check("return walk.gkwalk_depth_preorder_all(G('d', 'a>b c>d +e'))", "[a,b,c,d,e]")

    def test_all_depth_skips_visited_roots(self):
        self.check("return walk.gkwalk_depth_preorder_all(G('d', 'a>c a>b b>d'))", "[a,c,b,d]")
