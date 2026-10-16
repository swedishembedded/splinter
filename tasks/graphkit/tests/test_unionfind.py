# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.graphkithost import LuaCase

EIGHT = (
    "local d = dsu.gkdsu_from_list({1, 2, 3, 4, 5, 6, 7, 8})\n"
    "for _, p in ipairs({{1, 2}, {3, 4}, {5, 6}, {7, 8}, {1, 3}, {5, 7}, {1, 5}}) do\n"
    "  dsu.gkdsu_union(d, p[1], p[2])\n"
    "end\n"
)


class T(LuaCase):
    modules = {"dsu": "unionfind"}

    def test_adding_an_element_opens_a_set(self):
        self.check(
            "local d = dsu.gkdsu_new()\n"
            "return {dsu.gkdsu_add(d, 'a'), dsu.gkdsu_set_count(d), dsu.gkdsu_element_count(d)}", "[true,1,1]")

    def test_adding_twice_changes_nothing(self):
        self.check(
            "local d = dsu.gkdsu_new(); dsu.gkdsu_add(d, 'a')\n"
            "return {dsu.gkdsu_add(d, 'a'), dsu.gkdsu_set_count(d), dsu.gkdsu_element_count(d)}", "[false,1,1]")

    def test_nil_element_is_rejected(self):
        self.check("return fail(dsu.gkdsu_add, dsu.gkdsu_new(), nil)", "element must not be nil")

    def test_from_list_adds_each_element_once(self):
        self.check(
            "local d = dsu.gkdsu_from_list({'a', 'b', 'a', 'c'})\n"
            "return {dsu.gkdsu_set_count(d), dsu.gkdsu_element_count(d)}", "[3,3]")

    def test_find_rejects_unknown_element(self):
        self.check("return fail(dsu.gkdsu_find, dsu.gkdsu_new(), 'q')", "unknown element: q")

    def test_larger_second_set_becomes_root(self):
        self.check(
            "local d = dsu.gkdsu_from_list({'a', 'b', 'c'}); dsu.gkdsu_union(d, 'a', 'b')\n"
            "dsu.gkdsu_union(d, 'c', 'a')\n"
            "return {dsu.gkdsu_find(d, 'c'), dsu.gkdsu_find(d, 'b')}", "[a,a]")

    def test_union_rejects_unknown_second_element(self):
        self.check("return fail(dsu.gkdsu_union, dsu.gkdsu_from_list({'a'}), 'a', 'y')", "unknown element: y")

    def test_size_of_counts_set_members(self):
        self.check(
            "local d = dsu.gkdsu_from_list({'a', 'b', 'c', 'd'})\n"
            "dsu.gkdsu_union(d, 'a', 'b'); dsu.gkdsu_union(d, 'c', 'b')\n"
            "return {dsu.gkdsu_size_of(d, 'a'), dsu.gkdsu_size_of(d, 'c'), dsu.gkdsu_size_of(d, 'd')}", "[3,3,1]")

    def test_merged_element_hangs_below_root(self):
        self.check(
            "local d = dsu.gkdsu_from_list({'a', 'b'}); dsu.gkdsu_union(d, 'a', 'b')\n"
            "return {dsu.gkdsu_depth_of(d, 'a'), dsu.gkdsu_depth_of(d, 'b')}", "[0,1]")

    def test_depth_rejects_unknown_element(self):
        self.check("return fail(dsu.gkdsu_depth_of, dsu.gkdsu_new(), 'q')", "unknown element: q")

    def test_depth_query_does_not_compress(self):
        self.check(EIGHT + "dsu.gkdsu_depth_of(d, 8)\nreturn dsu.gkdsu_depth_of(d, 8)", "3")

    def test_find_halves_the_path_it_walks(self):
        self.check(
            EIGHT + "dsu.gkdsu_find(d, 8)\n"
            "return {dsu.gkdsu_depth_of(d, 8), dsu.gkdsu_depth_of(d, 7), dsu.gkdsu_depth_of(d, 4)}", "[2,2,2]")

    def test_find_returns_the_overall_root(self):
        self.check(EIGHT + "return {dsu.gkdsu_find(d, 8), dsu.gkdsu_find(d, 4), dsu.gkdsu_size_of(d, 6)}", "[1,1,8]")

    def test_groups_follow_first_member_order(self):
        self.check(
            "local d = dsu.gkdsu_from_list({'a', 'b', 'c', 'd', 'e'})\n"
            "dsu.gkdsu_union(d, 'c', 'a'); dsu.gkdsu_union(d, 'd', 'e')\n"
            "return dsu.gkdsu_groups(d)", "[[a,c],[b],[d,e]]")

    def test_groups_of_empty_structure_are_empty(self):
        self.check("return dsu.gkdsu_groups(dsu.gkdsu_new())", "[]")

    def test_members_list_the_whole_set(self):
        self.check(
            "local d = dsu.gkdsu_from_list({'a', 'b', 'c', 'd'})\n"
            "dsu.gkdsu_union(d, 'd', 'b'); dsu.gkdsu_union(d, 'a', 'd')\n"
            "return {dsu.gkdsu_members(d, 'b'), dsu.gkdsu_members(d, 'c')}", "[[a,b,d],[c]]")

    def test_numeric_elements_are_supported(self):
        self.check(
            "local d = dsu.gkdsu_from_list({10, 20, 30}); dsu.gkdsu_union(d, 30, 10)\n"
            "return {dsu.gkdsu_connected(d, 10, 30), dsu.gkdsu_members(d, 30)}", "[true,[10,30]]")
