// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable, database-backed tool-use
// environments for agent training, for its clients. If your team needs
// expertise in synthesising API and workflow environments with state-diff
// verification, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The sources the database workflow environment is built from: the shop's
//! schema, the tool the solver works with, and the verifier.
//!
//! The tool is one Python file that both the solver (through its command
//! line) and the verifier (as a module, on its own copy) use, so the rules
//! the solver meets are the rules the expected state was computed with.

/// The tables of the shop. `returns` is keyed by order: one return per
/// order, so no generated key depends on the order a solver acts in.
pub(super) const SCHEMA_SQL: &str = "\
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, credit INTEGER NOT NULL);
CREATE TABLE products (sku TEXT PRIMARY KEY, name TEXT NOT NULL, stock INTEGER NOT NULL, price INTEGER NOT NULL);
CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(id), status TEXT NOT NULL);
CREATE TABLE order_items (order_id INTEGER NOT NULL REFERENCES orders(id), sku TEXT NOT NULL REFERENCES products(sku), qty INTEGER NOT NULL);
CREATE TABLE returns (order_id INTEGER PRIMARY KEY REFERENCES orders(id), reason TEXT NOT NULL);
";

/// The shop's tool and its rules. Run with no arguments it lists its
/// commands; every refusal exits with status 2 and changes nothing.
pub(super) const TOOLS_PY: &str = r#"#!/usr/bin/env python3
"""The shop's tool: the only supported way to change the order database.

Commands:
  init                      rebuild shop.db from schema.sql and seed.sql
  tables                    list the tables
  show TABLE                print every row of a table
  cancel_order ID           cancel a pending order and put its items back in stock
  ship_order ID             ship a pending order if every line is in stock
  refund ID                 refund a shipped order to the customer's credit
  restock SKU N             add N units to a product's stock
  apply_credit ID           pay a pending order from the customer's credit
  create_return ID REASON   record the return of a refunded order
"""
import os
import sqlite3
import sys


class Refused(Exception):
    """The shop's rules do not allow the request."""


def init(conn, schema, seed):
    conn.executescript(schema)
    conn.executescript(seed)
    conn.commit()


def _order(conn, order_id, status):
    row = conn.execute("SELECT customer_id, status FROM orders WHERE id = ?", (order_id,)).fetchone()
    if row is None:
        raise Refused("no order %s" % order_id)
    if row[1] != status:
        raise Refused("order %s is %s, not %s" % (order_id, row[1], status))
    return row[0]


def _items(conn, order_id):
    return conn.execute("SELECT sku, qty FROM order_items WHERE order_id = ?", (order_id,)).fetchall()


def _total(conn, order_id):
    return conn.execute(
        "SELECT COALESCE(SUM(p.price * i.qty), 0) FROM order_items i JOIN products p ON p.sku = i.sku WHERE i.order_id = ?",
        (order_id,),
    ).fetchone()[0]


def cancel_order(conn, order_id):
    _order(conn, order_id, "pending")
    for sku, qty in _items(conn, order_id):
        conn.execute("UPDATE products SET stock = stock + ? WHERE sku = ?", (qty, sku))
    conn.execute("UPDATE orders SET status = 'cancelled' WHERE id = ?", (order_id,))


def ship_order(conn, order_id):
    _order(conn, order_id, "pending")
    for sku, qty in _items(conn, order_id):
        have = conn.execute("SELECT stock FROM products WHERE sku = ?", (sku,)).fetchone()[0]
        if have < qty:
            raise Refused("%s is short: %s in stock, %s needed" % (sku, have, qty))
    for sku, qty in _items(conn, order_id):
        conn.execute("UPDATE products SET stock = stock - ? WHERE sku = ?", (qty, sku))
    conn.execute("UPDATE orders SET status = 'shipped' WHERE id = ?", (order_id,))


def refund(conn, order_id):
    customer = _order(conn, order_id, "shipped")
    conn.execute("UPDATE customers SET credit = credit + ? WHERE id = ?", (_total(conn, order_id), customer))
    conn.execute("UPDATE orders SET status = 'refunded' WHERE id = ?", (order_id,))


def restock(conn, sku, n):
    n = int(n)
    if n <= 0:
        raise Refused("restock needs a positive number")
    cur = conn.execute("UPDATE products SET stock = stock + ? WHERE sku = ?", (n, sku))
    if cur.rowcount != 1:
        raise Refused("no product %s" % sku)


def apply_credit(conn, order_id):
    customer = _order(conn, order_id, "pending")
    total = _total(conn, order_id)
    credit = conn.execute("SELECT credit FROM customers WHERE id = ?", (customer,)).fetchone()[0]
    if credit < total:
        raise Refused("credit %s does not cover %s" % (credit, total))
    conn.execute("UPDATE customers SET credit = credit - ? WHERE id = ?", (total, customer))
    conn.execute("UPDATE orders SET status = 'paid' WHERE id = ?", (order_id,))


def create_return(conn, order_id, reason):
    _order(conn, order_id, "refunded")
    if conn.execute("SELECT 1 FROM returns WHERE order_id = ?", (order_id,)).fetchone():
        raise Refused("order %s already has a return" % order_id)
    conn.execute("INSERT INTO returns (order_id, reason) VALUES (?, ?)", (order_id, reason))


COMMANDS = {
    "cancel_order": (cancel_order, int),
    "ship_order": (ship_order, int),
    "refund": (refund, int),
    "restock": (restock, str, int),
    "apply_credit": (apply_credit, int),
    "create_return": (create_return, int, str),
}


def apply(conn, op):
    """Apply one operation given as [name, arg, ...]."""
    function, *types = COMMANDS[op[0]]
    function(conn, *[kind(arg) for kind, arg in zip(types, op[1:])])


def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    database = os.path.join(here, "shop.db")
    if len(argv) < 2:
        print(__doc__)
        return 0
    command = argv[1]
    if command == "init":
        schema = open(os.path.join(here, "schema.sql")).read()
        seed = open(os.path.join(here, "seed.sql")).read()
        init(sqlite3.connect(database), schema, seed)
        return 0
    conn = sqlite3.connect(database)
    if command == "tables":
        for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"):
            print(name)
        return 0
    if command == "show" and len(argv) == 3:
        names = [n for (n,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        if argv[2] not in names:
            print("refused: no table %s" % argv[2])
            return 2
        for row in conn.execute("SELECT * FROM %s" % argv[2]):
            print("|".join(str(v) for v in row))
        return 0
    if command not in COMMANDS:
        print("refused: unknown command %s" % command)
        return 2
    try:
        apply(conn, argv[1:])
        conn.commit()
    except Refused as why:
        print("refused: %s" % why)
        return 2
    except (ValueError, IndexError, TypeError):
        print("refused: bad arguments for %s" % command)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
"#;

/// The verifier: builds the expected state from the pinned schema, seed and
/// reference operations with the pinned tool, and compares every table of the
/// solver's database to it. It reads the database read-only, looks only at
/// real tables, and grades a database it cannot read as zero.
pub(super) const VERIFY_PY: &str = r#"import json
import os
import sqlite3
import sys

tests = os.environ.get("TESTS_DIR", "/tests")
app = os.environ.get("APP_DIR", "/app")
logs = os.environ.get("LOGS_DIR", "/logs/verifier")
sys.path.insert(0, tests)
import tools_ref  # the pinned copy of the shop's tool


def quoted(name):
    # A table name comes from the solver's database: quote it, never splice it.
    return '"' + name.replace('"', '""') + '"'


def tables(conn):
    names = [n for (n,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    state = {}
    for name in names:
        columns = [c[1] for c in conn.execute("PRAGMA table_info(%s)" % quoted(name))]
        rows = [list(r) for r in conn.execute("SELECT * FROM %s" % quoted(name))]
        state[name] = {"columns": columns, "rows": sorted(rows, key=lambda r: json.dumps(r))}
    return state


def write(vector):
    os.makedirs(logs, exist_ok=True)
    with open(os.path.join(logs, "reward.json"), "w") as out:
        json.dump(vector, out)


expected_conn = sqlite3.connect(":memory:")
tools_ref.init(expected_conn, open(os.path.join(tests, "schema.sql")).read(), open(os.path.join(tests, "seed.sql")).read())
for op in json.load(open(os.path.join(tests, "ops.json"))):
    tools_ref.apply(expected_conn, op)
expected = tables(expected_conn)

try:
    got_conn = sqlite3.connect("file:%s?mode=ro" % os.path.join(app, "shop.db"), uri=True)
    got = tables(got_conn)
except sqlite3.Error as why:
    print("the database cannot be read: %s" % why)
    write({"reward": 0.0, "state": 0.0, "exact": 0.0})
    sys.exit(0)

matching = 0
for name in sorted(set(expected) | set(got)):
    if expected.get(name) == got.get(name):
        matching += 1
        continue
    print("table %s differs" % name)
    want = expected.get(name, {"columns": None, "rows": []})
    have = got.get(name, {"columns": None, "rows": []})
    if want["columns"] != have["columns"]:
        print("  columns: expected %s, found %s" % (want["columns"], have["columns"]))
    for row in want["rows"]:
        if row not in have["rows"]:
            print("  missing  %s" % row)
    for row in have["rows"]:
        if row not in want["rows"]:
            print("  unwanted %s" % row)
total = len(set(expected) | set(got))
exact = 1.0 if matching == total and total > 0 else 0.0
write({"reward": exact, "exact": exact, "state": round(matching / total, 4) if total else 0.0})
"#;
