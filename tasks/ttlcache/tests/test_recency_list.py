# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from ttlcache.recency_list import RecencyChain, RecencyLink


def build(*keys):
    chain = RecencyChain()
    links = {}
    for key in keys:
        links[key] = RecencyLink(key, key.upper())
        chain.push_back(links[key])
    return chain, links


class T(unittest.TestCase):
    def test_new_link_has_no_neighbours_or_owner(self):
        link = RecencyLink("k", 5)
        self.assertEqual((link.prev, link.next, link.owner), (None, None, None))

    def test_push_front_makes_newest_first(self):
        chain = RecencyChain()
        for key in "abc":
            chain.push_front(RecencyLink(key, 0))
        self.assertEqual(chain.keys_front_to_back(), ["c", "b", "a"])

    def test_push_back_appends_in_order(self):
        chain, _ = build("a", "b", "c")
        self.assertEqual(chain.keys_front_to_back(), ["a", "b", "c"])

    def test_length_counts_pushed_links(self):
        chain, _ = build("a", "b", "c")
        self.assertEqual(len(chain), 3)

    def test_push_sets_owner_to_chain(self):
        chain, links = build("a")
        self.assertIs(links["a"].owner, chain)

    def test_push_front_on_empty_sets_both_ends(self):
        chain = RecencyChain()
        link = RecencyLink("a", 1)
        chain.push_front(link)
        self.assertIs(chain.peek_front(), link)
        self.assertIs(chain.peek_back(), link)

    def test_push_back_on_empty_sets_both_ends(self):
        chain = RecencyChain()
        link = RecencyLink("a", 1)
        chain.push_back(link)
        self.assertIs(chain.peek_front(), link)
        self.assertIs(chain.peek_back(), link)

    def test_push_front_links_old_head_backwards(self):
        chain, links = build("a")
        chain.push_front(links_b := RecencyLink("b", 0))
        self.assertIs(links["a"].prev, links_b)
        self.assertIs(links_b.next, links["a"])

    def test_push_back_links_old_tail_forwards(self):
        chain, links = build("a")
        extra = RecencyLink("b", 0)
        chain.push_back(extra)
        self.assertIs(links["a"].next, extra)
        self.assertIs(extra.prev, links["a"])

    def test_pushing_owned_link_is_rejected(self):
        chain, links = build("a")
        with self.assertRaises(ValueError) as caught:
            chain.push_front(links["a"])
        self.assertEqual(str(caught.exception), "link already in a chain")

    def test_pushing_link_of_other_chain_is_rejected(self):
        _, links = build("a")
        other = RecencyChain()
        with self.assertRaises(ValueError) as caught:
            other.push_back(links["a"])
        self.assertEqual(str(caught.exception), "link already in a chain")
        self.assertEqual(len(other), 0)

    def test_unlink_middle_joins_neighbours(self):
        chain, links = build("a", "b", "c")
        chain.unlink(links["b"])
        self.assertEqual(chain.keys_front_to_back(), ["a", "c"])
        self.assertIs(links["a"].next, links["c"])
        self.assertIs(links["c"].prev, links["a"])

    def test_unlink_head_promotes_next_link(self):
        chain, links = build("a", "b", "c")
        chain.unlink(links["a"])
        self.assertIs(chain.peek_front(), links["b"])
        self.assertIsNone(links["b"].prev)

    def test_unlink_tail_promotes_previous_link(self):
        chain, links = build("a", "b", "c")
        chain.unlink(links["c"])
        self.assertIs(chain.peek_back(), links["b"])
        self.assertIsNone(links["b"].next)

    def test_unlink_only_link_empties_chain(self):
        chain, links = build("a")
        chain.unlink(links["a"])
        self.assertEqual(len(chain), 0)
        self.assertIsNone(chain.peek_front())
        self.assertIsNone(chain.peek_back())

    def test_unlink_clears_link_pointers(self):
        chain, links = build("a", "b", "c")
        chain.unlink(links["b"])
        self.assertEqual((links["b"].prev, links["b"].next, links["b"].owner), (None, None, None))

    def test_unlink_foreign_link_is_rejected(self):
        chain, _ = build("a")
        stray = RecencyLink("z", 0)
        with self.assertRaises(ValueError) as caught:
            chain.unlink(stray)
        self.assertEqual(str(caught.exception), "link is not in this chain")

    def test_move_to_front_reorders_tail_link(self):
        chain, links = build("a", "b", "c")
        chain.move_to_front(links["c"])
        self.assertEqual(chain.keys_front_to_back(), ["c", "a", "b"])

    def test_move_to_front_keeps_head_in_place(self):
        chain, links = build("a", "b")
        chain.move_to_front(links["a"])
        self.assertEqual(chain.keys_front_to_back(), ["a", "b"])
        self.assertEqual(len(chain), 2)

    def test_move_to_front_of_foreign_link_fails(self):
        chain, _ = build("a")
        with self.assertRaises(ValueError):
            chain.move_to_front(RecencyLink("q", 0))

    def test_pop_back_returns_least_recent_link(self):
        chain, links = build("a", "b", "c")
        self.assertIs(chain.pop_back(), links["c"])
        self.assertEqual(chain.keys_front_to_back(), ["a", "b"])

    def test_pop_back_on_empty_chain_raises(self):
        with self.assertRaises(IndexError) as caught:
            RecencyChain().pop_back()
        self.assertEqual(str(caught.exception), "chain is empty")

    def test_peeking_empty_chain_gives_none(self):
        chain = RecencyChain()
        self.assertIsNone(chain.peek_front())
        self.assertIsNone(chain.peek_back())

    def test_keys_back_to_front_reverses_order(self):
        chain, _ = build("a", "b", "c")
        self.assertEqual(chain.keys_back_to_front(), ["c", "b", "a"])



if __name__ == "__main__":
    unittest.main()
