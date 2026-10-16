# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.graphkithost import LuaCase


class T(LuaCase):
    modules = {"adj": "adjacency"}


    def test_vertex_add_reports_whether_new(self):
        self.check(
            "local g = adj.gkadj_new(true)\n"
            "return {adj.gkadj_add_vertex(g, 'a'), adj.gkadj_add_vertex(g, 'a')}", "[true,false]")

    def test_nil_vertex_is_rejected(self):
        self.check("return fail(adj.gkadj_add_vertex, adj.gkadj_new(true), nil)", "vertex must not be nil")

    def test_has_vertex_reports_membership(self):
        self.check(
            "local g = adj.gkadj_new(true); adj.gkadj_add_vertex(g, 'a')\n"
            "return {adj.gkadj_has_vertex(g, 'a'), adj.gkadj_has_vertex(g, 'b')}", "[true,false]")

    def test_text_weight_is_rejected(self):
        self.check(
            "return fail(adj.gkadj_add_edge, adj.gkadj_new(true), 'a', 'b', 'heavy')",
            "weight must be a number")

    def test_nil_endpoint_leaves_graph_untouched(self):
        self.check(
            "local g = adj.gkadj_new(true)\n"
            "return {fail(adj.gkadj_add_edge, g, 'a', nil), fail(adj.gkadj_add_edge, g, nil, 'b'),"
            " adj.gkadj_vertices(g)}", "[vertex must not be nil,vertex must not be nil,[]]")


    def test_undirected_edge_goes_both_ways(self):
        self.check(
            "local g = adj.gkadj_new(false); adj.gkadj_add_edge(g, 'a', 'b', 4)\n"
            "return {adj.gkadj_has_edge(g, 'a', 'b'), adj.gkadj_has_edge(g, 'b', 'a'),"
            " adj.gkadj_weight(g, 'b', 'a')}", "[true,true,4]")

    def test_unknown_vertices_have_no_edge(self):
        self.check("return adj.gkadj_has_edge(adj.gkadj_new(true), 'p', 'q')", "false")

    def test_second_add_replaces_weight_without_new_edge(self):
        self.check(
            "local g = adj.gkadj_new(true)\n"
            "adj.gkadj_add_edge(g, 'a', 'b', 3); adj.gkadj_add_edge(g, 'a', 'b', 5)\n"
            "return {adj.gkadj_weight(g, 'a', 'b'), adj.gkadj_edge_count(g), adj.gkadj_degree_in(g, 'b')}",
            "[5,1,1]")

    def test_undirected_readd_from_other_end(self):
        self.check(
            "local g = adj.gkadj_new(false)\n"
            "adj.gkadj_add_edge(g, 'a', 'b', 1); adj.gkadj_add_edge(g, 'b', 'a', 7)\n"
            "return {adj.gkadj_weight(g, 'a', 'b'), adj.gkadj_weight(g, 'b', 'a'), adj.gkadj_edge_count(g)}",
            "[7,7,1]")

    def test_remove_edge_reports_and_updates(self):
        self.check(
            "local g = adj.gkadj_new(true); adj.gkadj_add_edge(g, 'a', 'b')\n"
            "return {adj.gkadj_remove_edge(g, 'a', 'b'), adj.gkadj_edge_count(g), adj.gkadj_has_edge(g, 'a', 'b')}",
            "[true,0,false]")

    def test_removing_absent_edge_changes_nothing(self):
        self.check(
            "local g = adj.gkadj_new(true); adj.gkadj_add_edge(g, 'a', 'b')\n"
            "return {adj.gkadj_remove_edge(g, 'b', 'a'), adj.gkadj_edge_count(g)}", "[false,1]")

    def test_undirected_removal_drops_both_arcs(self):
        self.check(
            "local g = adj.gkadj_new(false); adj.gkadj_add_edge(g, 'a', 'b')\n"
            "adj.gkadj_remove_edge(g, 'b', 'a')\n"
            "return {adj.gkadj_has_edge(g, 'a', 'b'), adj.gkadj_has_edge(g, 'b', 'a'),"
            " adj.gkadj_degree_in(g, 'a'), adj.gkadj_degree_in(g, 'b')}", "[false,false,0,0]")

    def test_undirected_self_loop_removal(self):
        self.check(
            "local g = adj.gkadj_new(false); adj.gkadj_add_edge(g, 'a', 'a')\n"
            "return {adj.gkadj_remove_edge(g, 'a', 'a'), adj.gkadj_edge_count(g), adj.gkadj_degree_in(g, 'a')}",
            "[true,0,0]")

    def test_removed_edge_lowers_target_degree(self):
        self.check(
            "local g = adj.gkadj_new(true)\n"
            "adj.gkadj_add_edge(g, 'a', 'b'); adj.gkadj_add_edge(g, 'c', 'b'); adj.gkadj_remove_edge(g, 'a', 'b')\n"
            "return adj.gkadj_degree_in(g, 'b')", "1")

    def test_degree_of_unknown_vertex_fails(self):
        self.check(
            "local g = adj.gkadj_new(true)\n"
            "return {fail(adj.gkadj_degree_out, g, 'q'), fail(adj.gkadj_degree_in, g, 'r')}",
            "[unknown vertex: q,unknown vertex: r]")


    def test_neighbors_list_targets_and_weights(self):
        self.check(
            "local g = adj.gkadj_new(true)\n"
            "adj.gkadj_add_edge(g, 'a', 'b', 2); adj.gkadj_add_edge(g, 'a', 'c')\n"
            "return adj.gkadj_neighbors(g, 'a')", "[{to=b,w=2},{to=c,w=1}]")

    def test_neighbors_result_is_a_copy(self):
        self.check(
            "local g = adj.gkadj_new(true); adj.gkadj_add_edge(g, 'a', 'b', 2)\n"
            "local list = adj.gkadj_neighbors(g, 'a'); list[1].w = 99; list[1].to = 'z'\n"
            "return adj.gkadj_neighbors(g, 'a')", "[{to=b,w=2}]")

    def test_neighbors_of_unknown_vertex_fail(self):
        self.check("return fail(adj.gkadj_neighbors, adj.gkadj_new(true), 7)", "unknown vertex: 7")

    def test_vertex_removal_drops_incident_arcs(self):
        self.check(
            "local h = adj.gkadj_new(true)\n"
            "adj.gkadj_add_edge(h, 'a', 'b'); adj.gkadj_add_edge(h, 'b', 'c'); adj.gkadj_add_edge(h, 'c', 'a')\n"
            "adj.gkadj_remove_vertex(h, 'b')\n"
            "return {adj.gkadj_vertices(h), adj.gkadj_edge_count(h), adj.gkadj_has_edge(h, 'c', 'a'),"
            " adj.gkadj_neighbors(h, 'a')}", "[[a,c],1,true,[]]")

    def test_vertex_removal_lowers_target_degrees(self):
        self.check(
            "local g = adj.gkadj_new(true)\n"
            "adj.gkadj_add_edge(g, 'a', 'b'); adj.gkadj_add_edge(g, 'a', 'c'); adj.gkadj_remove_vertex(g, 'a')\n"
            "return {adj.gkadj_degree_in(g, 'b'), adj.gkadj_degree_in(g, 'c'), adj.gkadj_edge_count(g)}",
            "[0,0,0]")

    def test_undirected_vertex_removal_counts_edges(self):
        self.check(
            "local g = adj.gkadj_new(false)\n"
            "adj.gkadj_add_edge(g, 'a', 'b'); adj.gkadj_add_edge(g, 'b', 'c'); adj.gkadj_add_edge(g, 'a', 'c')\n"
            "adj.gkadj_remove_vertex(g, 'a')\n"
            "return {adj.gkadj_edge_count(g), adj.gkadj_degree_out(g, 'b'), adj.gkadj_degree_in(g, 'c')}",
            "[1,1,1]")

    def test_directed_self_loop_vanishes_with_vertex(self):
        self.check(
            "local g = adj.gkadj_new(true)\n"
            "adj.gkadj_add_edge(g, 'a', 'a'); adj.gkadj_add_edge(g, 'a', 'b'); adj.gkadj_remove_vertex(g, 'a')\n"
            "return {adj.gkadj_edge_count(g), adj.gkadj_degree_in(g, 'b')}", "[0,0]")

    def test_undirected_self_loop_vanishes_with_vertex(self):
        self.check(
            "local g = adj.gkadj_new(false)\n"
            "adj.gkadj_add_edge(g, 'a', 'a'); adj.gkadj_add_edge(g, 'a', 'b'); adj.gkadj_remove_vertex(g, 'a')\n"
            "return {adj.gkadj_edge_count(g), adj.gkadj_degree_out(g, 'b')}", "[0,0]")


    def test_removing_unknown_vertex_fails(self):
        self.check("return fail(adj.gkadj_remove_vertex, adj.gkadj_new(true), 'q')", "unknown vertex: q")

    def test_directed_edge_listing_keeps_arc_order(self):
        self.check(
            "local g = adj.gkadj_new(true)\n"
            "adj.gkadj_add_edge(g, 'a', 'b', 2); adj.gkadj_add_edge(g, 'a', 'c'); adj.gkadj_add_edge(g, 'b', 'a')\n"
            "return adj.gkadj_edges(g)",
            "[{from=a,to=b,w=2},{from=a,to=c,w=1},{from=b,to=a,w=1}]")

    def test_undirected_edge_listing_names_each_once(self):
        self.check(
            "local g = adj.gkadj_new(false)\n"
            "adj.gkadj_add_edge(g, 'b', 'a', 5); adj.gkadj_add_edge(g, 'a', 'c')\n"
            "return adj.gkadj_edges(g)", "[{from=b,to=a,w=5},{from=a,to=c,w=1}]")

    def test_undirected_listing_includes_self_loop(self):
        self.check(
            "local g = adj.gkadj_new(false); adj.gkadj_add_edge(g, 'a', 'a', 3)\n"
            "return adj.gkadj_edges(g)", "[{from=a,to=a,w=3}]")
