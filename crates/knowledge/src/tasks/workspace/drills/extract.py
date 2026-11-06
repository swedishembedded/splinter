# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements comprehension drills made by code from a
# codebase's own functions and tests, for its clients. If your team needs
# expertise in turning a repository into verified training tasks you can
# procure our services by sending an email to info@swedishembedded.com.
"""Reads the functions of Python source files and the literal calls the
tests make of them, by the parser alone: nothing of the project is imported.

usage: python3 -I -c <this script>, run in the root of the tree, with a JSON
document on standard input:

    {"sources": ["pkg/mod.py", ...], "tests": ["tests/test_mod.py", ...]}

and prints one JSON document:

    {"functions": [{
        "file": "pkg/mod.py", "name": "f", "line_start": 3, "line_end": 9,
        "params": [{"name": "x", "kind": "positional", "default": null}],
        "modules": [{"root": "", "module": "pkg.mod"}],
        "calls": [{"args": ["1", "'a'"], "kwargs": [["k", "2"]]}]
    }], "unreadable": ["path", ...]}

A function is a module-level `def` that is not a generator and not async.
`modules` are the ways a test imports the file's module (the directory to put
on the path, and the dotted name); `calls` are the calls the tests make of it
whose every argument is a literal, written as the parser prints it.
"""
import ast
import json
import sys

MAX_LITERAL_CHARS = 200


def dotted_names(path):
    """The (root, dotted module) pairs a file can be imported as: each suffix
    of its path below some leading directories."""
    parts = path[: -len(".py")].split("/")
    found = []
    for cut in range(len(parts)):
        rest = parts[cut:]
        if rest[-1] == "__init__":
            rest = rest[:-1]
        if rest:
            found.append(("/".join(parts[:cut]), ".".join(rest)))
    return found


def is_generator(node):
    """Whether the function's own body yields (a nested function's does not)."""
    pending = list(node.body)
    while pending:
        item = pending.pop()
        if isinstance(item, (ast.Yield, ast.YieldFrom)):
            return True
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        pending.extend(ast.iter_child_nodes(item))
    return False


def literal(node):
    """The parser's text of `node` when it is a literal, else None."""
    try:
        ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None
    text = ast.unparse(node)
    return text if len(text) <= MAX_LITERAL_CHARS and "\n" not in text else None


def parameters(args):
    """The parameters in call order, each with the text of a literal default."""
    found = []
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for arg, default in zip(positional, defaults):
        found.append((arg.arg, "positional", default))
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        found.append((arg.arg, "keyword", default))
    if args.vararg:
        found.append((args.vararg.arg, "variadic", None))
    if args.kwarg:
        found.append((args.kwarg.arg, "variadic", None))
    return [
        {"name": name, "kind": kind, "default": literal(default) if default is not None else None,
         "has_default": default is not None}
        for name, kind, default in found
    ]


def read_sources(paths, unreadable):
    functions = {}
    for path in paths:
        try:
            with open(path, encoding="utf-8") as handle:
                tree = ast.parse(handle.read())
        except (OSError, SyntaxError, ValueError):
            unreadable.append(path)
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not is_generator(node):
                functions[(path, node.name)] = {
                    "file": path,
                    "name": node.name,
                    "line_start": min([node.lineno] + [d.lineno for d in node.decorator_list]),
                    "line_end": node.end_lineno,
                    "params": parameters(node.args),
                    "modules": [],
                    "calls": [],
                }
    return functions


def resolve(module, files):
    """The (file, root) of the source file a dotted module names, if any."""
    wanted = module.split(".")
    for path in files:
        for root, dotted in dotted_names(path):
            if dotted.split(".") == wanted:
                return path, root
    return None


def read_tests(paths, functions, unreadable):
    files = sorted({path for path, _ in functions})
    for path in paths:
        try:
            with open(path, encoding="utf-8") as handle:
                tree = ast.parse(handle.read())
        except (OSError, SyntaxError, ValueError):
            unreadable.append(path)
            continue
        names = {}    # local name -> (file, function name)
        modules = {}  # local name -> file
        roots = {}    # file -> set of import roots seen
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found = resolve(node.module, files)
                for alias in node.names:
                    if found and (found[0], alias.name) in functions:
                        names[alias.asname or alias.name] = (found[0], alias.name)
                        roots.setdefault(found[0], set()).add(found[1])
                    sub = resolve(node.module + "." + alias.name, files)
                    if sub:
                        modules[alias.asname or alias.name] = sub[0]
                        roots.setdefault(sub[0], set()).add(sub[1])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    found = resolve(alias.name, files)
                    if found and alias.asname:
                        modules[alias.asname] = found[0]
                        roots.setdefault(found[0], set()).add(found[1])
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = None
            if isinstance(node.func, ast.Name) and node.func.id in names:
                target = names[node.func.id]
            elif (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                  and node.func.value.id in modules
                  and (modules[node.func.value.id], node.func.attr) in functions):
                target = (modules[node.func.value.id], node.func.attr)
            if target is None:
                continue
            args = [literal(a) for a in node.args]
            kwargs = [(k.arg, literal(k.value)) for k in node.keywords]
            if any(a is None for a in args) or any(k is None or v is None for k, v in kwargs):
                continue
            record = functions[target]
            call = {"args": args, "kwargs": [list(pair) for pair in kwargs]}
            if call not in record["calls"]:
                record["calls"].append(call)
            for root in roots.get(target[0], ()):
                module = dict(dotted_names(target[0])).get(root)
                entry = {"root": root, "module": module}
                if module and entry not in record["modules"]:
                    record["modules"].append(entry)


def main():
    if sys.version_info < (3, 9):
        sys.exit("python 3.9 or newer is required to read literals")
    request = json.load(sys.stdin)
    unreadable = []
    functions = read_sources(request["sources"], unreadable)
    read_tests(request["tests"], functions, unreadable)
    ordered = sorted(functions.values(), key=lambda f: (f["file"], f["line_start"]))
    json.dump({"functions": ordered, "unreadable": sorted(set(unreadable))}, sys.stdout)


main()
