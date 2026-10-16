# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from ttlcache.memoizer import TtlMemoizer, memoize_key, ttl_memoize


class FakeClock:
    def __init__(self, now=0):
        self.now = now

    def __call__(self):
        return self.now


class Counting:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return len(self.calls)


class T(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock(0)
        self.target = Counting()

    def wrap(self, ttl=10, **options):
        return TtlMemoizer(self.target, self.clock, ttl, **options)

    def test_key_separates_positional_and_keyword_arguments(self):
        self.assertNotEqual(memoize_key((1, "a", 2), {}), memoize_key((1,), {"a": 2}))

    def test_key_ignores_keyword_order(self):
        self.assertEqual(memoize_key((), {"a": 1, "b": 2}), memoize_key((), {"b": 2, "a": 1}))

    def test_decorator_is_untyped_by_default(self):
        cached = self.wrap()
        cached(1)
        cached(1.0)
        self.assertEqual(cached.info()["hits"], 1)

    def test_untyped_key_treats_equal_numbers_alike(self):
        self.assertEqual(memoize_key((1,), {}), memoize_key((1.0,), {}))

    def test_typed_key_separates_argument_types(self):
        self.assertNotEqual(memoize_key((1,), {}, typed=True), memoize_key((1.0,), {}, typed=True))

    def test_typed_key_separates_keyword_value_types(self):
        self.assertNotEqual(
            memoize_key((), {"x": 1}, typed=True), memoize_key((), {"x": True}, typed=True)
        )

    def test_unhashable_argument_is_rejected(self):
        with self.assertRaises(TypeError) as caught:
            memoize_key(([1],), {})
        self.assertEqual(str(caught.exception), "arguments must be hashable")

    def test_first_call_runs_the_function(self):
        self.assertEqual(self.wrap()(5), 1)
        self.assertEqual(self.target.calls, [((5,), {})])

    def test_repeat_call_reuses_cached_result(self):
        cached = self.wrap()
        cached(5)
        self.assertEqual(cached(5), 1)
        self.assertEqual(len(self.target.calls), 1)

    def test_keyword_arguments_take_part_in_the_key(self):
        cached = self.wrap()
        cached(a=1)
        cached(a=2)
        cached(a=1)
        self.assertEqual(len(self.target.calls), 2)

    def test_result_is_fresh_just_before_expiry(self):
        cached = self.wrap(ttl=10)
        cached(1)
        self.clock.now = 9
        cached(1)
        self.assertEqual(len(self.target.calls), 1)

    def test_result_expires_exactly_at_ttl(self):
        cached = self.wrap(ttl=10)
        cached(1)
        self.clock.now = 10
        self.assertEqual(cached(1), 2)

    def test_recompute_restarts_the_lifetime(self):
        cached = self.wrap(ttl=10)
        cached(1)
        self.clock.now = 10
        cached(1)
        self.clock.now = 19
        self.assertEqual(cached(1), 2)

    def test_hit_does_not_extend_the_lifetime(self):
        cached = self.wrap(ttl=10)
        cached(1)
        self.clock.now = 5
        cached(1)
        self.clock.now = 10
        self.assertEqual(cached(1), 2)

    def test_none_results_are_cached(self):
        calls = []

        def returns_none():
            calls.append(1)

        cached = TtlMemoizer(returns_none, self.clock, 10)
        cached()
        cached()
        self.assertEqual(len(calls), 1)

    def test_raised_exceptions_are_not_cached(self):
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("boom")
            return "ok"

        cached = TtlMemoizer(flaky, self.clock, 10)
        with self.assertRaises(RuntimeError):
            cached()
        self.assertEqual(cached(), "ok")
        self.assertEqual(cached.info()["misses"], 2)

    def test_info_counts_hits_misses_and_size(self):
        cached = self.wrap()
        cached(1)
        cached(1)
        cached(2)
        self.assertEqual(cached.info(), {"hits": 1, "misses": 2, "size": 2})

    def test_expired_lookup_counts_as_miss(self):
        cached = self.wrap(ttl=5)
        cached(1)
        self.clock.now = 5
        cached(1)
        self.assertEqual(cached.info(), {"hits": 0, "misses": 2, "size": 1})

    def test_maxsize_evicts_oldest_result(self):
        cached = self.wrap(maxsize=2)
        cached(1)
        cached(2)
        cached(3)
        self.assertEqual(cached.info()["size"], 2)
        cached(1)
        self.assertEqual(len(self.target.calls), 4)

    def test_maxsize_keeps_recently_used_result(self):
        cached = self.wrap(maxsize=2)
        cached(1)
        cached(2)
        cached(1)
        cached(3)
        cached(1)
        self.assertEqual(len(self.target.calls), 3)

    def test_results_fill_up_to_maxsize_without_loss(self):
        cached = self.wrap(maxsize=3)
        for n in (1, 2, 3):
            cached(n)
        for n in (1, 2, 3):
            cached(n)
        self.assertEqual(cached.info(), {"hits": 3, "misses": 3, "size": 3})

    def test_invalidate_forces_fresh_recomputation_next(self):
        cached = self.wrap()
        cached(1)
        self.assertIs(cached.invalidate(1), True)
        self.assertEqual(cached(1), 2)

    def test_invalidate_unknown_arguments_reports_false(self):
        self.assertIs(self.wrap().invalidate(99), False)

    def test_invalidate_honours_keyword_arguments(self):
        cached = self.wrap()
        cached(a=1)
        self.assertTrue(cached.invalidate(a=1))
        self.assertFalse(cached.invalidate(a=1))

    def test_invalidate_of_cached_none_reports_true(self):
        cached = TtlMemoizer(lambda: None, self.clock, 10)
        cached()
        self.assertIs(cached.invalidate(), True)

    def test_decorator_preserves_function_name(self):
        @ttl_memoize(self.clock, 10)
        def compute_total(x):
            return x * 2

        self.assertEqual(compute_total.__name__, "compute_total")
        self.assertEqual(compute_total(4), 8)

    def test_factory_defaults_to_untyped_unbounded(self):
        same = ttl_memoize(self.clock, 10)(lambda x: x)
        same(1)
        same(1.0)
        self.assertEqual(same.info(), {"hits": 1, "misses": 1, "size": 1})

    def test_decorator_passes_options_through(self):
        @ttl_memoize(self.clock, 10, maxsize=1, typed=True)
        def same(x):
            return x

        same(1)
        same(1.0)
        self.assertEqual(same.info(), {"hits": 0, "misses": 2, "size": 1})

    def test_zero_ttl_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            self.wrap(ttl=0)
        self.assertEqual(str(caught.exception), "ttl must be positive")

    def test_zero_maxsize_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            self.wrap(maxsize=0)
        self.assertEqual(str(caught.exception), "maxsize must be at least 1")

    def test_non_callable_clock_is_rejected(self):
        with self.assertRaises(TypeError) as caught:
            TtlMemoizer(self.target, "noon", 5)
        self.assertEqual(str(caught.exception), "clock must be callable")


if __name__ == "__main__":
    unittest.main()
