// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifiable training environments whose
// verifiers are held to the tools they stand for, for its clients. If your
// team needs expertise in reinforcement-learning environments, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec, against the real interpreters (plan 4B): the program-output
//! environments, Python 3 and Lua 5.1, at every feature level over many
//! seeds. Each instance's pinned program is run here directly, by the
//! interpreter and nothing of the plugin's, and the plugin's own verifier,
//! run in its host, is held to that output:
//!
//! * the interpreter's output is solved, and so is it with trailing spaces
//!   and blank lines, which the instruction says are ignored;
//! * one line changed, the lines reversed, a line missing, or nothing at
//!   all is not solved (unless the program prints nothing);
//! * a program the interpreter cannot run gives no reward: a broken oracle
//!   is never a wrong answer.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::process::Command;

use splinter_core::verifiability::Split;
use splinter_knowledge::factory::artifact::Instance;
use splinter_knowledge::factory::host::Host;
use splinter_knowledge::factory::plugin::{InstanceRequest, Plugin};
use splinter_knowledge::factory::plugins::program_output::ProgramOutput;

fn request(seed: u64, blocks: u32, features: u32) -> InstanceRequest {
    InstanceRequest {
        split: Split::Train,
        seed,
        difficulty: BTreeMap::from([
            ("blocks".into(), blocks),
            ("features".into(), features),
            ("magnitude".into(), 2),
        ]),
    }
}

/// What `interpreter` prints for `file` of `instance`, run here directly.
fn direct(instance: &Instance, interpreter: &[&str], file: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(file);
    std::fs::write(
        &path,
        instance.folder().text(&format!("tests/{file}")).unwrap(),
    )
    .unwrap();
    let out = Command::new(interpreter[0])
        .args(&interpreter[1..])
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

fn reward(instance: &Instance, answer: &str) -> Option<f64> {
    let session = Host::default().open(instance).unwrap();
    session.write("answer.txt", answer.as_bytes()).unwrap();
    session.verify().unwrap().reward()
}

#[test]
fn the_verifier_rewards_exactly_what_the_interpreter_prints() {
    let languages: [(ProgramOutput, &[&str], &str); 2] = [
        (ProgramOutput::python(), &["python3", "-I"], "program.py"),
        (ProgramOutput::lua(), &["lua5.1"], "program.lua"),
    ];
    let mut wrong = Vec::new();
    for (plugin, interpreter, file) in languages {
        for features in 1..=4 {
            // seed: a spread of generated programs per level, each checked by the interpreter
            for seed in 0..6 {
                let instance = plugin.generate(&request(seed, 3, features)).unwrap();
                let printed = direct(&instance, interpreter, file);
                let lines: Vec<&str> = printed.lines().collect();
                let case = format!("{} features {features} seed {seed}", plugin.id());
                let mut check = |what: &str, answer: String, solved: bool| {
                    let got = reward(&instance, &answer);
                    if got.map(|r| r >= 1.0) != Some(solved) {
                        wrong.push(format!("{case}: {what} got {got:?}"));
                    }
                };
                check("the interpreter's output", printed.clone(), true);
                let padded: String = lines.iter().map(|l| format!("{l}   \n")).collect();
                check(
                    "its output with trailing spaces and lines",
                    format!("{padded}\n\n"),
                    true,
                );
                if lines.is_empty() {
                    continue;
                }
                let mut changed = lines.clone();
                let last = format!("{}x", changed[changed.len() - 1]);
                *changed.last_mut().unwrap() = &last;
                check("a line changed", changed.join("\n"), false);
                check("a line missing", lines[..lines.len() - 1].join("\n"), false);
                check("nothing", String::new(), false);
                let reversed: Vec<&str> = lines.iter().rev().copied().collect();
                if reversed != lines {
                    check("the lines reversed", reversed.join("\n"), false);
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_program_the_interpreter_cannot_run_gives_no_reward() {
    for (plugin, file, broken) in [
        (
            ProgramOutput::python(),
            "program.py",
            "raise SystemExit(3)\n",
        ),
        (ProgramOutput::lua(), "program.lua", "error('broken')\n"),
    ] {
        let instance = plugin.generate(&request(1, 3, 2)).unwrap();
        let mut folder = instance.folder().clone();
        folder.put_text(&format!("tests/{file}"), broken).unwrap();
        let instance = Instance::from_folder(folder).unwrap();
        assert_eq!(reward(&instance, "anything"), None, "{}", plugin.id());
    }
}

/// The shop's rules written again, from the tool's documentation, against
/// SQLite directly: what each request does to the database. Run in the
/// workspace with the reference operations as its argument.
const REFERENCE_SHOP: &str = r#"
import json, os, sqlite3, sys
if os.path.exists("shop.db"):
    os.remove("shop.db")
db = sqlite3.connect("shop.db")
db.executescript(open("schema.sql").read())
db.executescript(open("seed.sql").read())
one = lambda sql, *a: db.execute(sql, a).fetchone()
def total(o):
    return one("SELECT COALESCE(SUM(p.price * i.qty), 0) FROM order_items i, products p WHERE p.sku = i.sku AND i.order_id = ?", o)[0]
for op in json.loads(sys.argv[1]):
    name, args = op[0], op[1:]
    if name == "restock":
        db.execute("UPDATE products SET stock = stock + ? WHERE sku = ?", (int(args[1]), args[0]))
        continue
    o = int(args[0])
    customer, status = one("SELECT customer_id, status FROM orders WHERE id = ?", o)
    items = db.execute("SELECT sku, qty FROM order_items WHERE order_id = ?", (o,)).fetchall()
    if name == "cancel_order":
        for sku, qty in items:
            db.execute("UPDATE products SET stock = stock + ? WHERE sku = ?", (qty, sku))
        db.execute("UPDATE orders SET status = 'cancelled' WHERE id = ?", (o,))
    elif name == "ship_order":
        for sku, qty in items:
            db.execute("UPDATE products SET stock = stock - ? WHERE sku = ?", (qty, sku))
        db.execute("UPDATE orders SET status = 'shipped' WHERE id = ?", (o,))
    elif name == "refund":
        db.execute("UPDATE customers SET credit = credit + ? WHERE id = ?", (total(o), customer))
        db.execute("UPDATE orders SET status = 'refunded' WHERE id = ?", (o,))
    elif name == "apply_credit":
        db.execute("UPDATE customers SET credit = credit - ? WHERE id = ?", (total(o), customer))
        db.execute("UPDATE orders SET status = 'paid' WHERE id = ?", (o,))
    elif name == "create_return":
        db.execute("INSERT INTO returns (order_id, reason) VALUES (?, ?)", (o, args[1]))
    else:
        raise SystemExit("no rule for " + name)
db.commit()
"#;

#[test]
fn the_database_verifier_rewards_exactly_the_state_the_requests_lead_to() {
    use splinter_knowledge::factory::plugins::db_workflow::DbWorkflow;
    let plugin = DbWorkflow::new();
    let mut wrong = Vec::new();
    // seed: a spread of workflows per setting, each rebuilt independently
    for seed in 0..8 {
        for (goals, traps) in [(3, 0), (4, 2)] {
            let instance = plugin
                .generate(&InstanceRequest {
                    split: Split::Train,
                    seed,
                    difficulty: BTreeMap::from([
                        ("goals".into(), goals),
                        ("traps".into(), traps),
                        ("rows".into(), 2),
                        ("distractor_tables".into(), 1),
                    ]),
                })
                .unwrap();
            let ops = instance
                .folder()
                .text("tests/ops.json")
                .unwrap()
                .to_string();
            let case = format!("seed {seed} goals {goals} traps {traps}");
            let run = |extra: &str| {
                let session = Host::default().open(&instance).unwrap();
                session
                    .write("reference_shop.py", REFERENCE_SHOP.as_bytes())
                    .unwrap();
                let built = session
                    .run_in_workspace(&["python3", "-I", "reference_shop.py", &ops])
                    .unwrap();
                assert!(built.succeeded(), "{case}: {built:?}");
                if !extra.is_empty() {
                    session.write("extra.sql", extra.as_bytes()).unwrap();
                    let done = session
                        .run_in_workspace(&[
                            "python3",
                            "-c",
                            "import sqlite3; c = sqlite3.connect('shop.db'); c.executescript(open('extra.sql').read()); c.commit()",
                        ])
                        .unwrap();
                    assert!(done.succeeded(), "{case}: {done:?}");
                }
                session.remove("reference_shop.py");
                session.remove("extra.sql");
                session.verify().unwrap().solved()
            };
            if !run("") {
                wrong.push(format!(
                    "{case}: the state the requests lead to is not solved"
                ));
            }
            if run("UPDATE customers SET credit = credit + 1 WHERE id = (SELECT MIN(id) FROM customers);") {
                wrong.push(format!("{case}: a customer's credit changed besides is solved"));
            }
            if run("UPDATE products SET stock = stock + 1 WHERE sku = (SELECT MIN(sku) FROM products);") {
                wrong.push(format!("{case}: a product's stock changed besides is solved"));
            }
            let session = Host::default().open(&instance).unwrap();
            if session.verify().unwrap().solved() {
                wrong.push(format!("{case}: the untouched database is solved"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
