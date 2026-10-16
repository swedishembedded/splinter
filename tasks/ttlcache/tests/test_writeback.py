# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from ttlcache.writeback import WriteBackBuffer


class FakeClock:
    def __init__(self, now=0):
        self.now = now

    def __call__(self):
        return self.now


class Sink:
    def __init__(self):
        self.batches = []
        self.fail = False

    def __call__(self, writes, removals):
        if self.fail:
            raise OSError("disk full")
        self.batches.append((writes, removals))


class T(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock(50)
        self.sink = Sink()
        self.buf = WriteBackBuffer(self.sink, self.clock, max_batch=3, max_age=10)

    def test_zero_batch_size_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            WriteBackBuffer(self.sink, self.clock, 0, 10)
        self.assertEqual(str(caught.exception), "max_batch must be at least 1")

    def test_zero_max_age_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            WriteBackBuffer(self.sink, self.clock, 3, 0)
        self.assertEqual(str(caught.exception), "max_age must be positive")

    def test_batch_size_of_one_flushes_every_write(self):
        buf = WriteBackBuffer(self.sink, self.clock, 1, 10)
        self.assertIs(buf.write("a", 1), True)
        self.assertEqual(self.sink.batches, [([("a", 1)], [])])

    def test_max_age_of_one_is_accepted(self):
        buf = WriteBackBuffer(self.sink, self.clock, 3, 1)
        buf.write("a", 1)
        self.clock.now = 51
        self.assertIs(buf.due(), True)

    def test_write_below_batch_limit_stays_buffered(self):
        self.assertIs(self.buf.write("a", 1), False)
        self.assertEqual(self.sink.batches, [])
        self.assertEqual(self.buf.pending_count(), 1)

    def test_reaching_batch_limit_flushes_in_order(self):
        self.buf.write("a", 1)
        self.buf.write("b", 2)
        self.assertIs(self.buf.write("c", 3), True)
        self.assertEqual(self.sink.batches, [([("a", 1), ("b", 2), ("c", 3)], [])])
        self.assertEqual(self.buf.pending_count(), 0)

    def test_rewriting_a_key_coalesces_into_one_entry(self):
        self.buf.write("a", 1)
        self.buf.write("a", 2)
        self.assertEqual(self.buf.pending_count(), 1)
        self.buf.flush()
        self.assertEqual(self.sink.batches, [([("a", 2)], [])])

    def test_coalesced_key_keeps_its_first_position(self):
        self.buf.write("a", 1)
        self.buf.write("b", 2)
        self.buf.write("a", 3)
        self.buf.flush()
        self.assertEqual(self.sink.batches[0][0], [("a", 3), ("b", 2)])

    def test_coalescing_does_not_count_toward_batch_limit(self):
        for value in range(5):
            self.assertIs(self.buf.write("a", value), False)

    def test_removals_travel_in_their_own_list(self):
        self.buf.write("a", 1)
        self.buf.remove("b")
        self.buf.flush()
        self.assertEqual(self.sink.batches, [([("a", 1)], ["b"])])

    def test_removal_counts_toward_batch_limit(self):
        self.buf.remove("a")
        self.buf.remove("b")
        self.assertIs(self.buf.remove("c"), True)
        self.assertEqual(self.sink.batches, [([], ["a", "b", "c"])])

    def test_write_after_removal_cancels_the_removal(self):
        self.buf.remove("a")
        self.buf.write("a", 7)
        self.buf.flush()
        self.assertEqual(self.sink.batches, [([("a", 7)], [])])

    def test_removal_after_write_replaces_the_write(self):
        self.buf.write("a", 7)
        self.buf.remove("a")
        self.buf.flush()
        self.assertEqual(self.sink.batches, [([], ["a"])])

    def test_read_sees_pending_value(self):
        self.buf.write("a", 5)
        self.assertEqual(self.buf.read("a"), 5)

    def test_read_of_unknown_key_gives_default(self):
        self.assertEqual(self.buf.read("a", "dflt"), "dflt")
        self.assertIsNone(self.buf.read("a"))

    def test_read_of_pending_removal_gives_default(self):
        self.buf.remove("a")
        self.assertEqual(self.buf.read("a", "gone"), "gone")

    def test_is_pending_covers_writes_and_removals(self):
        self.buf.write("a", 1)
        self.buf.remove("b")
        self.assertTrue(self.buf.is_pending("a"))
        self.assertTrue(self.buf.is_pending("b"))
        self.assertFalse(self.buf.is_pending("c"))

    def test_flush_returns_number_of_entries_sent(self):
        self.buf.write("a", 1)
        self.buf.remove("b")
        self.assertEqual(self.buf.flush(), 2)

    def test_flush_of_empty_buffer_skips_the_sink(self):
        self.assertEqual(self.buf.flush(), 0)
        self.assertEqual(self.sink.batches, [])

    def test_buffer_is_not_due_before_max_age(self):
        self.buf.write("a", 1)
        self.clock.now = 59
        self.assertIs(self.buf.due(), False)
        self.assertIs(self.buf.tick(), False)
        self.assertEqual(self.sink.batches, [])

    def test_buffer_is_due_exactly_at_max_age(self):
        self.buf.write("a", 1)
        self.clock.now = 60
        self.assertIs(self.buf.due(), True)

    def test_tick_flushes_an_aged_buffer(self):
        self.buf.write("a", 1)
        self.clock.now = 60
        self.assertIs(self.buf.tick(), True)
        self.assertEqual(self.sink.batches, [([("a", 1)], [])])

    def test_age_is_measured_from_first_pending_entry(self):
        self.buf.write("a", 1)
        self.clock.now = 58
        self.buf.write("b", 2)
        self.clock.now = 60
        self.assertIs(self.buf.due(), True)

    def test_late_write_triggers_age_flush(self):
        self.buf.write("a", 1)
        self.clock.now = 65
        self.assertIs(self.buf.write("b", 2), True)
        self.assertEqual(self.sink.batches, [([("a", 1), ("b", 2)], [])])

    def test_age_restarts_after_a_flush(self):
        self.buf.write("a", 1)
        self.buf.flush()
        self.clock.now = 200
        self.buf.write("b", 2)
        self.clock.now = 205
        self.assertIs(self.buf.due(), False)

    def test_failed_flush_keeps_entries_pending(self):
        self.buf.write("a", 1)
        self.sink.fail = True
        with self.assertRaises(OSError) as caught:
            self.buf.flush()
        self.assertEqual(str(caught.exception), "disk full")
        self.assertEqual(self.buf.pending_count(), 1)
        self.assertEqual(self.buf.read("a"), 1)

    def test_retry_after_failure_delivers_everything(self):
        self.buf.write("a", 1)
        self.sink.fail = True
        with self.assertRaises(OSError):
            self.buf.flush()
        self.sink.fail = False
        self.assertEqual(self.buf.flush(), 1)
        self.assertEqual(self.sink.batches, [([("a", 1)], [])])

    def test_failure_during_triggered_flush_propagates_from_write(self):
        self.sink.fail = True
        self.buf.write("a", 1)
        self.buf.write("b", 2)
        with self.assertRaises(OSError):
            self.buf.write("c", 3)
        self.assertEqual(self.buf.pending_count(), 3)

    def test_failed_flush_keeps_original_age(self):
        self.buf.write("a", 1)
        self.sink.fail = True
        self.clock.now = 70
        with self.assertRaises(OSError):
            self.buf.flush()
        self.assertIs(self.buf.due(), True)

    def test_stats_track_flushes_failures_and_deliveries(self):
        self.buf.write("a", 1)
        self.sink.fail = True
        with self.assertRaises(OSError):
            self.buf.flush()
        self.sink.fail = False
        self.buf.flush()
        self.buf.write("b", 2)
        self.buf.remove("c")
        self.buf.flush()
        self.buf.write("d", 4)
        self.assertEqual(self.buf.stats(), {"flushes": 2, "failures": 1, "delivered": 3, "pending": 1})


if __name__ == "__main__":
    unittest.main()
