# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements executable evaluation of code changes
# against a project's own test suite, for its clients. If your team needs
# expertise in test-driven benchmarks for code-editing agents, you can
# procure our services by sending an email to info@swedishembedded.com.
"""Takes unittest cases out of a test module, leaving the rest of it.

usage: python3 -I -c <this script> FILE TEST_ID...

Each TEST_ID is what unittest calls a test (`pkg.module.Class.test_name`);
the case is found by the longest `Class.test_name` the id ends with. The
definition of each case is removed from FILE with the decorators and the
comment lines directly above it, by the line range the parser gives, so
nothing else in the module changes; a class left with no statement gets
`pass`. FILE is rewritten only when the result still parses and defines none
of the cases. Exit status 2, and FILE untouched, when an id names no method
of the module or the result is not valid.
"""
import ast
import sys


def methods_by_name(tree):
    """Maps `Outer.Inner.name` to the (function, owning class) pairs."""
    found = {}

    def walk(body, prefix, owner):
        for node in body:
            if isinstance(node, ast.ClassDef):
                walk(node.body, prefix + node.name + ".", node)
            elif owner is not None and isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                found.setdefault(prefix + node.name, []).append((node, owner))

    walk(tree.body, "", None)
    return found


def first_line(node, lines):
    """The 1-based first line of `node`: its decorators and the comment
    lines directly above it belong to it."""
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    while start > 1 and lines[start - 2].lstrip().startswith("#"):
        start -= 1
    return start


def cut(text, ids):
    tree = ast.parse(text)
    methods = methods_by_name(tree)
    doomed = []
    for test_id in ids:
        names = [q for q in methods if test_id == q or test_id.endswith("." + q)]
        if not names:
            raise ValueError("no method of the module is named by " + test_id)
        doomed.extend(methods[max(names, key=len)])
    lines = text.split("\n")
    ranges = {}
    for node, owner in doomed:
        ranges[(first_line(node, lines), node.end_lineno)] = (node, owner)
    gone = {id(node) for node, _ in ranges.values()}
    emptied = {
        id(owner): owner
        for _, owner in ranges.values()
        if all(id(stmt) in gone for stmt in owner.body)
    }
    # From the bottom up, so that a removal does not move those above it.
    placeholder_at = {
        min(
            start
            for (start, _), (node, owner) in ranges.items()
            if owner is emptied[key]
        ): " " * emptied[key].body[0].col_offset + "pass"
        for key in emptied
    }
    for (start, end), _ in sorted(ranges.items(), reverse=True):
        lines[start - 1 : end] = [placeholder_at[start]] if start in placeholder_at else []
    result = "\n".join(lines)
    left = methods_by_name(ast.parse(result))
    for test_id in ids:
        if any(test_id.endswith("." + q) for q in left):
            raise ValueError("a case named by " + test_id + " is still defined")
    return result


def main(argv):
    path, ids = argv[0], argv[1:]
    with open(path, encoding="utf-8", newline="") as source:
        text = source.read()
    try:
        result = cut(text, ids)
    except (SyntaxError, ValueError) as error:
        print(path + ": " + str(error), file=sys.stderr)
        return 2
    with open(path, "w", encoding="utf-8", newline="") as target:
        target.write(result)
    return 0


sys.exit(main(sys.argv[1:]))
