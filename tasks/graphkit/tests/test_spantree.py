# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.graphkithost import LuaCase

NET = "G('u', 'a>b:4 a>c:1 b>c:2 b>d:5 c>d:8 c>e:10 d>e:2 d>f:6 e>f:3')"
SPLIT = "G('u', 'a>b:1 c>d:2 +z')"
TRIANGLE = "G('u', 'a>b:1 b>c:1 a>c:1')"
NEEDS_UNDIRECTED = "spanning tree needs an undirected graph"


class T(LuaCase):
    modules = {"mst": "spantree"}

    def test_forest_takes_lightest_joining_edges(self):
        self.check(
            f"return {{mst.gkmst_forest({NET})}}",
            "[[{from=a,to=c,w=1},{from=b,to=c,w=2},{from=d,to=e,w=2},{from=e,to=f,w=3},{from=b,to=d,w=5}],13]")

    def test_total_weight_adds_chosen_edges(self):
        self.check(f"return mst.gkmst_total_weight({NET})", "13")

    def test_forest_refuses_directed_graph(self):
        self.check("return fail(mst.gkmst_forest, G('d', 'a>b'))", NEEDS_UNDIRECTED)

    def test_maximum_forest_takes_heaviest_edges(self):
        self.check(
            f"return {{mst.gkmst_maximum_forest({NET})}}",
            "[[{from=c,to=e,w=10},{from=c,to=d,w=8},{from=d,to=f,w=6},{from=b,to=d,w=5},{from=a,to=b,w=4}],33]")

    def test_maximum_forest_ties_prefer_storage_order(self):
        self.check(
            "return {mst.gkmst_maximum_forest(G('u', 'a>b:5 b>c:5 a>c:5'))}",
            "[[{from=a,to=b,w=5},{from=a,to=c,w=5}],10]")

    def test_maximum_forest_refuses_directed_graph(self):
        self.check("return fail(mst.gkmst_maximum_forest, G('d', 'a>b'))", NEEDS_UNDIRECTED)

    def test_tree_of_connected_graph(self):
        self.check(f"local edges, total = mst.gkmst_tree({NET})\nreturn {{#edges, total}}", "[5,13]")

    def test_tree_rejects_split_graph(self):
        self.check(f"return fail(mst.gkmst_tree, {SPLIT})", "graph is not connected: 3 trees")

    def test_tree_rejects_one_stray_vertex(self):
        self.check("return fail(mst.gkmst_tree, G('u', 'a>b:1 +z'))", "graph is not connected: 2 trees")

    def test_tree_accepts_two_vertex_graph(self):
        self.check("return {mst.gkmst_tree(G('u', 'a>b:7'))}", "[[{from=a,to=b,w=7}],7]")

    def test_connected_graph_is_a_spanning_tree(self):
        self.check(f"return mst.gkmst_is_spanning_tree({NET})", "true")

    def test_split_graph_is_not_a_spanning_tree(self):
        self.check(f"return mst.gkmst_is_spanning_tree({SPLIT})", "false")

    def test_lone_vertex_is_a_spanning_tree(self):
        self.check("return mst.gkmst_is_spanning_tree(G('u', '+a'))", "true")

    def test_containment_ignores_endpoint_order(self):
        self.check(f"return mst.gkmst_contains_edge({NET}, 'c', 'a')", "true")

    def test_containment_matches_second_endpoint_order(self):
        self.check(f"return mst.gkmst_contains_edge({NET}, 'd', 'b')", "true")

    def test_skipped_edge_is_not_contained(self):
        self.check(f"return mst.gkmst_contains_edge({NET}, 'a', 'b')", "false")

    def test_edge_sharing_one_endpoint_is_not_contained(self):
        self.check(f"return mst.gkmst_contains_edge({NET}, 'a', 'd')", "false")

    def test_bottleneck_finds_lightest_heaviest_edge(self):
        self.check(f"return mst.gkmst_bottleneck({NET}, 'a', 'f')", "5")

    def test_bottleneck_between_adjacent_vertices(self):
        self.check(f"return mst.gkmst_bottleneck({NET}, 'a', 'c')", "1")

    def test_bottleneck_to_self_is_zero(self):
        self.check(f"return mst.gkmst_bottleneck({NET}, 'e', 'e')", "0")

    def test_bottleneck_across_split_is_nil(self):
        self.check(f"return tostring(mst.gkmst_bottleneck({SPLIT}, 'a', 'c'))", "nil")
