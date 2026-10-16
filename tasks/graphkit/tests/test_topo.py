# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.graphkithost import LuaCase

DIAMOND = "G('d', 'a>b a>c b>d c>d')"
SCRAMBLED = "G('d', 'c>d b>d a>b')"
RING = "G('d', 'a>b b>c c>a')"
UNDIRECTED = "G('u', 'a>b')"
NEEDS_DIRECTED = "topological order needs a directed graph"


class T(LuaCase):
    modules = {"topo": "topo"}

    def test_order_reports_cycle_behind_a_prefix(self):
        self.check(
            "local o, c = topo.gktopo_order(G('d', 'x>a a>b b>a'))\nreturn {tostring(o), c}", "[nil,[a,b]]")

    def test_order_refuses_undirected_graph(self):
        self.check(f"return fail(topo.gktopo_order, {UNDIRECTED})", NEEDS_DIRECTED)

    def test_sort_names_the_cycle_in_its_error(self):
        self.check(f"return fail(topo.gktopo_sort, G('d', 'x>a a>b b>a'))", "cycle detected: a>b>a")

    def test_sort_error_for_self_loop(self):
        self.check("return fail(topo.gktopo_sort, G('d', 'a>a'))", "cycle detected: a>a")

    def test_find_cycle_lists_vertices_in_arc_order(self):
        self.check(f"return topo.gktopo_find_cycle({RING})", "[a,b,c]")

    def test_find_cycle_skips_finished_branches(self):
        self.check("return topo.gktopo_find_cycle(G('d', 'a>b a>c c>d d>c'))", "[c,d]")

    def test_find_cycle_refuses_undirected_graph(self):
        self.check(f"return fail(topo.gktopo_find_cycle, {UNDIRECTED})", NEEDS_DIRECTED)

    def test_acyclic_graph_is_recognised(self):
        self.check(f"return topo.gktopo_is_acyclic({DIAMOND})", "true")

    def test_cyclic_graph_is_recognised(self):
        self.check(f"return topo.gktopo_is_acyclic({RING})", "false")

    def test_reverse_finish_is_a_different_valid_order(self):
        self.check(f"return topo.gktopo_reverse_finish({DIAMOND})", "[a,c,b,d]")

    def test_reverse_finish_covers_every_root(self):
        self.check(f"return topo.gktopo_reverse_finish({SCRAMBLED})", "[a,b,c,d]")

    def test_reverse_finish_names_cycle_in_error(self):
        self.check(f"return fail(topo.gktopo_reverse_finish, {RING})", "cycle detected: a>b>c>a")

    def test_reverse_finish_refuses_undirected_graph(self):
        self.check(f"return fail(topo.gktopo_reverse_finish, {UNDIRECTED})", NEEDS_DIRECTED)

    def test_layers_follow_the_longest_chain(self):
        self.check("return topo.gktopo_layers(G('d', 'a>b b>c a>c'))", "[[a],[b],[c]]")

    def test_layers_put_roots_together(self):
        self.check(f"return topo.gktopo_layers({SCRAMBLED})", "[[c,a],[b],[d]]")

    def test_chain_length_counts_vertices(self):
        self.check(f"return topo.gktopo_chain_length({DIAMOND})", "3")

    def test_chain_length_of_loose_vertices_is_one(self):
        self.check("return topo.gktopo_chain_length(G('d', '+a +b'))", "1")

    def test_rank_map_numbers_the_order_from_one(self):
        self.check(f"return topo.gktopo_rank_map({SCRAMBLED})", "{a=2,b=3,c=1,d=4}")

    def test_sources_have_no_incoming_arcs(self):
        self.check("return topo.gktopo_sources(G('d', 'a>b +z'))", "[a,z]")

    def test_cycle_has_neither_sources_nor_sinks(self):
        self.check(f"return {{topo.gktopo_sources({RING}), topo.gktopo_sinks({RING})}}", "[[],[]]")

    def test_sources_follow_vertex_order(self):
        self.check(f"return topo.gktopo_sources({SCRAMBLED})", "[c,a]")

    def test_source_and_sink_queries_refuse_undirected(self):
        self.check(
            f"return {{fail(topo.gktopo_sources, {UNDIRECTED}), fail(topo.gktopo_sinks, {UNDIRECTED})}}",
            f"[{NEEDS_DIRECTED},{NEEDS_DIRECTED}]")

    def test_sinks_have_no_outgoing_arcs(self):
        self.check("return topo.gktopo_sinks(G('d', 'a>b +z'))", "[b,z]")



    def test_sort_refuses_undirected_graph(self):
        self.check(f"return fail(topo.gktopo_sort, {UNDIRECTED})", NEEDS_DIRECTED)
