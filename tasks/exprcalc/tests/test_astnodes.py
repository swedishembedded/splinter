# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import dataclasses
import unittest
from fractions import Fraction

from exprcalc.astnodes import (
    XprBinary,
    XprCall,
    XprNum,
    XprUnary,
    XprVar,
    xpr_child_nodes,
    xpr_is_number,
    xpr_literal_value,
    xpr_map_postorder,
    xpr_node_count,
    xpr_number_node,
    xpr_rebuild_node,
    xpr_tree_depth,
    xpr_walk_nodes,
)
from .sexpr import sx


class ZA(unittest.TestCase):
    def test_equal_nodes_compare_equal(self):
        self.assertEqual(sx("(+ 1 x)"), XprBinary("+", XprNum(1), XprVar("x")))
        self.assertEqual(XprVar("x", 3), XprVar("x", 9))

    def test_position_defaults_to_none(self):
        self.assertIsNone(XprNum(1).position)
        self.assertNotEqual(sx("(+ 1 2)"), sx("(- 1 2)"))

    def test_assigning_to_a_node_field_is_forbidden(self):
        for node in (XprNum(1), XprVar("x"), sx("(- x)"), sx("(+ 1 x)"), sx("(call f x)")):
            with self.assertRaises(dataclasses.FrozenInstanceError):
                node.position = 3
        self.assertEqual(len({sx("(+ 1 x)"), sx("(+ 1 x)"), sx("x")}), 2)


class ZB(unittest.TestCase):
    def test_positive_value_is_a_literal(self):
        self.assertEqual(xpr_number_node(5), XprNum(5))
        self.assertEqual(xpr_number_node(0), XprNum(0))

    def test_negative_value_is_a_negation(self):
        self.assertEqual(xpr_number_node(-5), XprUnary("-", XprNum(5)))
        self.assertEqual(xpr_number_node(Fraction(-1, 2)), XprUnary("-", XprNum(Fraction(1, 2))))

    def test_position_is_passed_to_both_nodes(self):
        node = xpr_number_node(-2, 7)
        self.assertEqual((node.position, node.operand.position), (7, 7))


class ZC(unittest.TestCase):
    def test_is_number_for_literal(self):
        self.assertTrue(xpr_is_number(XprNum(3)))
        self.assertFalse(xpr_is_number(XprVar("x")))
        self.assertFalse(xpr_is_number(sx("(- 3)")))

    def test_is_number_checks_the_value(self):
        self.assertTrue(xpr_is_number(XprNum(3), 3))
        self.assertFalse(xpr_is_number(XprNum(3), 4))
        self.assertTrue(xpr_is_number(XprNum(0), 0))
        self.assertFalse(xpr_is_number(XprNum(1), 0))

    def test_literal_value_of_number(self):
        self.assertEqual(xpr_literal_value(XprNum(Fraction(1, 2))), Fraction(1, 2))
        self.assertEqual(xpr_literal_value(sx("(- 4)")), -4)

    def test_literal_value_ignores_plus_sign(self):
        self.assertIsNone(xpr_literal_value(sx("(+ 4)")))
        self.assertIsNone(xpr_literal_value(sx("(- x)")))

    def test_literal_value_of_variable_is_none(self):
        self.assertIsNone(xpr_literal_value(sx("x")))
        self.assertEqual(xpr_literal_value(XprNum(0)), 0)


class ZD(unittest.TestCase):
    def test_leaves_have_no_children(self):
        self.assertEqual(xpr_child_nodes(XprNum(1)), ())
        self.assertEqual(xpr_child_nodes(XprVar("x")), ())
        self.assertEqual(xpr_child_nodes(sx("(- x)")), (XprVar("x"),))

    def test_binary_children_keep_order(self):
        self.assertEqual(xpr_child_nodes(sx("(- 1 x)")), (XprNum(1), XprVar("x")))
        self.assertEqual(xpr_child_nodes(sx("(call f 1 2 3)")), (XprNum(1), XprNum(2), XprNum(3)))

    def test_call_without_arguments_has_none(self):
        self.assertEqual(xpr_child_nodes(XprCall("f", ())), ())


class ZE(unittest.TestCase):
    def test_leaf_is_returned_unchanged(self):
        leaf = XprVar("x", 4)
        self.assertIs(xpr_rebuild_node(leaf, ()), leaf)
        rebuilt = xpr_rebuild_node(XprUnary("!", XprNum(1), 5), (XprNum(2),))
        self.assertEqual(rebuilt, sx("(! 2)"))
        self.assertEqual(rebuilt.position, 5)

    def test_binary_gets_new_children_and_keeps_position(self):
        rebuilt = xpr_rebuild_node(XprBinary("*", XprNum(1), XprNum(2), 6), (XprNum(3), XprNum(4)))
        self.assertEqual(rebuilt, sx("(* 3 4)"))
        self.assertEqual(rebuilt.position, 6)
        rebuilt = xpr_rebuild_node(XprCall("g", (XprNum(1),), 2), [XprNum(8), XprNum(9)])
        self.assertEqual(rebuilt, sx("(call g 8 9)"))
        self.assertIsInstance(rebuilt.args, tuple)
        self.assertEqual(rebuilt.position, 2)


class ZF(unittest.TestCase):
    def test_walk_visits_parents_before_children(self):
        tree = sx("(+ (* 1 2) (- x))")
        self.assertEqual(
            xpr_walk_nodes(tree),
            [tree, sx("(* 1 2)"), sx("1"), sx("2"), sx("(- x)"), sx("x")],
        )
        self.assertEqual(xpr_walk_nodes(XprVar("x")), [XprVar("x")])

    def test_walk_includes_call_arguments_in_order(self):
        names = [n.name for n in xpr_walk_nodes(sx("(call f a (call g b) c)")) if isinstance(n, XprVar)]
        self.assertEqual(names, ["a", "b", "c"])
        self.assertEqual(xpr_node_count(XprNum(1)), 1)

    def test_node_count_of_tree(self):
        self.assertEqual(xpr_node_count(sx("(+ (* 1 2) (- x))")), 6)


class ZG(unittest.TestCase):
    def test_depth_of_leaf_is_one(self):
        self.assertEqual(xpr_tree_depth(XprVar("x")), 1)
        self.assertEqual(xpr_tree_depth(sx("(+ 1 (* 2 (- x)))")), 4)

    def test_empty_call_has_tree_depth_one(self):
        self.assertEqual(xpr_tree_depth(XprCall("f", ())), 1)
        seen = []

        def record(node):
            seen.append(node)
            return node

        xpr_map_postorder(sx("(+ 1 (- x))"), record)
        self.assertEqual(seen, [sx("1"), sx("x"), sx("(- x)"), sx("(+ 1 (- x))")])

    def test_map_applies_transform_to_every_node(self):
        def bump(node):
            return XprNum(node.value + 1) if isinstance(node, XprNum) else node

        self.assertEqual(xpr_map_postorder(sx("(+ 1 (* 2 x))"), bump), sx("(+ 2 (* 3 x))"))
        def collapse(node):
            return XprNum(0) if isinstance(node, XprVar) else node

        result = xpr_map_postorder(sx("(+ x 1)"), lambda n: n if not isinstance(n, XprBinary) else XprNum(len(xpr_child_nodes(n))))
        self.assertEqual(result, XprNum(2))
        self.assertEqual(xpr_map_postorder(sx("(- x)"), collapse), sx("(- 0)"))
