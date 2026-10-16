# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.graphkithost import LuaCase

FILL = (
    "local h = heap.gkheap_new()\n"
    "for i, p in ipairs({5, 3, 8, 1, 9, 2, 7}) do heap.gkheap_push(h, 'v' .. p, p) end\n"
)


class T(LuaCase):
    modules = {"heap": "heap"}

    def test_push_returns_new_size(self):
        self.check(
            "local h = heap.gkheap_new()\n"
            "return {heap.gkheap_push(h, 'a', 4), heap.gkheap_push(h, 'b', 2)}", "[1,2]")

    def test_popping_an_empty_heap_fails(self):
        self.check("return fail(heap.gkheap_pop, heap.gkheap_new())", "heap is empty")

    def test_peeking_an_empty_heap_fails(self):
        self.check("return fail(heap.gkheap_peek, heap.gkheap_new())", "heap is empty")

    def test_peek_shows_minimum_and_keeps_it(self):
        self.check(FILL + "return {heap.gkheap_peek(h), heap.gkheap_size(h)}", "[v1,7]")

    def test_peek_reports_priority_too(self):
        self.check(FILL + "return {heap.gkheap_peek(h)}", "[v1,1]")

    def test_peek_prefers_earliest_of_equal_priority(self):
        self.check(
            "local h = heap.gkheap_new(); heap.gkheap_push(h, 'one', 3); heap.gkheap_push(h, 'two', 3)\n"
            "return {heap.gkheap_peek(h)}", "[one,3]")

    def test_nil_value_is_rejected(self):
        self.check("return fail(heap.gkheap_push, heap.gkheap_new(), nil, 1)", "value must not be nil")

    def test_text_priority_is_rejected(self):
        self.check("return fail(heap.gkheap_push, heap.gkheap_new(), 'a', 'high')", "priority must be a number")

    def test_missing_priority_is_rejected(self):
        self.check("return fail(heap.gkheap_push, heap.gkheap_new(), 'a')", "priority must be a number")

    def test_repeated_priorities_match_a_stable_sort(self):
        self.check(
            "local h, items = heap.gkheap_new(), {}\n"
            "for i = 1, 60 do local p = (i * 7) % 10; heap.gkheap_push(h, i, p); items[i] = {i, p} end\n"
            "table.sort(items, function(a, b) if a[2] ~= b[2] then return a[2] < b[2] end return a[1] < b[1] end)\n"
            "local expected = {}\n"
            "for i, item in ipairs(items) do expected[i] = item[1] end\n"
            "return show(heap.gkheap_drain(h)) == show(expected)", "true")

    def test_pop_many_takes_the_first_few(self):
        self.check(FILL + "return {heap.gkheap_pop_many(h, 3), heap.gkheap_size(h)}", "[[v1,v2,v3],4]")

    def test_pop_many_stops_when_heap_runs_out(self):
        self.check(FILL + "return heap.gkheap_pop_many(h, 99)", "[v1,v2,v3,v5,v7,v8,v9]")

    def test_pop_many_zero_takes_nothing(self):
        self.check(FILL + "return {heap.gkheap_pop_many(h, 0), heap.gkheap_size(h)}", "[[],7]")

    def test_pop_many_rejects_negative_count(self):
        self.check("return fail(heap.gkheap_pop_many, heap.gkheap_new(), -1)", "count must be a non-negative number")

    def test_pop_many_rejects_text_count(self):
        self.check("return fail(heap.gkheap_pop_many, heap.gkheap_new(), 'all')", "count must be a non-negative number")

    def test_drain_of_empty_heap_is_empty_list(self):
        self.check("return heap.gkheap_drain(heap.gkheap_new())", "[]")

    def test_from_pairs_builds_ordered_heap(self):
        self.check(
            "local h = heap.gkheap_from_pairs({{'late', 9}, {'early', 1}, {'mid', 4}})\n"
            "return {heap.gkheap_size(h), heap.gkheap_drain(h)}", "[3,[early,mid,late]]")

    def test_from_pairs_of_nothing_is_empty(self):
        self.check("return heap.gkheap_is_empty(heap.gkheap_from_pairs({}))", "true")
