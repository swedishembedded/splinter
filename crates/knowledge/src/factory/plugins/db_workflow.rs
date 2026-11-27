// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable, database-backed tool-use
// environments for agent training, for its clients. If your team needs
// expertise in synthesising API and workflow environments with state-diff
// verification, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Stateful tool use: a workflow carried out through a tool over a database,
//! verified by the final state.
//!
//! The solver is given an order database (`shop.db`), the shop's tool
//! (`tools.py`, which enforces the shop's rules and refuses what they do not
//! allow) and a list of requests in plain language. Some requests depend on
//! the state of the database ("ship it if every line is in stock"), and the
//! right answer to some of them is to do nothing. The verifier builds the
//! expected database from the pinned schema, the pinned seed and the
//! reference operations with its own copy of the tool, and compares every
//! table of the solver's database to it: the final-state diff. Any way of
//! reaching the state counts, and a change anywhere else (collateral damage)
//! fails.
//!
//! The environment's reset is `reset.sh`, which rebuilds `shop.db` from the
//! schema and the seed; a campaign that restarts an episode runs it.
//!
//! Difficulty knobs: `goals` (how many requests), `traps` (how many of them
//! are conditions whose answer is to leave things alone), `rows` (how much
//! unrelated data the database holds, some of it decoys with the same names
//! as the people in the requests) and `distractor_tables` (tables no request
//! touches).

use std::fmt::Write as _;

use splinter_core::rng::SplitMix64;
use splinter_core::verifiability::VerifierClass;

use crate::factory::artifact::{
    environment_id, EnvMeta, Instance, Parts, SolverTools, SCHEMA_VERSION,
};
use crate::factory::attack::{Attack, AttackKind, Edit};
use crate::factory::folder::Folder;
use crate::factory::plugin::{
    resolve_difficulty, InstanceRequest, Knob, Plugin, PluginError, Reference,
};

use super::db_tools::{SCHEMA_SQL, TOOLS_PY, VERIFY_PY};
use super::DB_WORKFLOW;

/// The plugin's version. Any change to the schema, the tool, the verifier,
/// the generator or the instruction text is a new version.
pub const VERSION: &str = "2";

const KNOBS: &[Knob] = &[
    Knob {
        name: "goals",
        min: 1,
        max: 6,
        default: 3,
        doc: "how many requests the solver must carry out",
    },
    Knob {
        name: "traps",
        min: 0,
        max: 5,
        default: 1,
        doc: "how many requests are conditions whose answer is to change nothing (at most goals - 1)",
    },
    Knob {
        name: "rows",
        min: 0,
        max: 40,
        default: 6,
        doc: "unrelated customers, products and orders in the database, some with the requests' names",
    },
    Knob {
        name: "distractor_tables",
        min: 0,
        max: 3,
        default: 1,
        doc: "tables no request touches",
    },
];

const FIRST: [&str; 10] = [
    "Ana", "Bo", "Cleo", "Dag", "Elin", "Finn", "Greta", "Hugo", "Ida", "Jon",
];
const LAST: [&str; 10] = [
    "Lund", "Berg", "Strand", "Holm", "Dahl", "Ek", "Lind", "Nyberg", "Sjo", "Vik",
];
const THINGS: [&str; 10] = [
    "lamp", "kettle", "rug", "chair", "mirror", "shelf", "vase", "clock", "basket", "stool",
];
const REASONS: [&str; 3] = ["damaged", "wrong item", "arrived late"];
const STATUSES: [&str; 5] = ["pending", "shipped", "cancelled", "refunded", "paid"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Cancel,
    Ship,
    Refund,
    Restock,
    Pay,
}

impl Kind {
    /// Whether the request can be a condition whose answer is to do nothing.
    fn can_trap(self) -> bool {
        matches!(self, Self::Ship | Self::Restock | Self::Pay)
    }
}

/// The database as it is seeded.
#[derive(Default)]
struct World {
    customers: Vec<(i64, String, i64)>,
    products: Vec<(String, String, i64, i64)>,
    orders: Vec<(i64, i64, &'static str)>,
    items: Vec<(i64, String, i64)>,
}

/// A request and the operations that carry it out.
struct Goal {
    text: String,
    ops: Vec<Vec<String>>,
}

fn between(rng: &mut SplitMix64, low: i64, high: i64) -> i64 {
    i64::try_from(rng.between(low.unsigned_abs(), high.unsigned_abs())).unwrap_or(low)
}

/// Adds the customer, order, products and items of one request.
fn build_goal(
    kind: Kind,
    trap: bool,
    index: usize,
    rng: &mut SplitMix64,
    world: &mut World,
) -> Goal {
    let customer = 10 + index as i64;
    // Distinct across requests and varied across instances: an order number
    // is what tells two small instances apart.
    let order = 100 + 700 * index as i64 + between(rng, 0, 699);
    let name = format!(
        "{} {}",
        FIRST[(index * 3 + rng.below(3)) % FIRST.len()],
        LAST[(index * 7 + rng.below(3)) % LAST.len()]
    );
    let lines = between(rng, 1, 3) as usize;
    let mut total = 0;
    let mut item_rows = Vec::new();
    for line in 0..lines {
        let sku = format!("SKU-{index}{line}");
        let (price, qty) = (between(rng, 5, 60), between(rng, 1, 4));
        total += price * qty;
        let thing = THINGS[rng.below(THINGS.len())];
        world.products.push((
            sku.clone(),
            format!("{thing} {index}{line}"),
            between(rng, 0, 30),
            price,
        ));
        item_rows.push((sku, qty));
    }
    let status = match kind {
        Kind::Refund => "shipped",
        _ => "pending",
    };
    let credit = match (kind, trap) {
        (Kind::Pay, false) => total + between(rng, 0, 20),
        (Kind::Pay, true) => (total - between(rng, 1, total / 2 + 1)).max(0),
        _ => between(rng, 0, 80),
    };
    world.customers.push((customer, name.clone(), credit));
    world.orders.push((order, customer, status));
    for (sku, qty) in &item_rows {
        world.items.push((order, sku.clone(), *qty));
    }
    let op = |parts: &[&str]| parts.iter().map(ToString::to_string).collect::<Vec<_>>();
    match kind {
        Kind::Cancel => Goal {
            text: format!("Customer {name} asked to cancel order #{order}. Cancel it."),
            ops: vec![op(&["cancel_order", &order.to_string()])],
        },
        Kind::Ship => {
            // Stock is set from the lines: enough for every one, or short on one.
            for (position, (sku, qty)) in item_rows.iter().enumerate() {
                let short = trap && position == 0;
                let stock = if short { qty - 1 } else { qty + between(rng, 0, 5) };
                if let Some(product) = world.products.iter_mut().find(|p| &p.0 == sku) {
                    product.2 = stock;
                }
            }
            Goal {
                text: format!("Ship order #{order} for {name} if the warehouse can fill every line from stock. If any line is short, leave the order as it is."),
                ops: if trap { vec![] } else { vec![op(&["ship_order", &order.to_string()])] },
            }
        }
        Kind::Refund => {
            let reason = REASONS[rng.below(REASONS.len())];
            Goal {
                text: format!("Order #{order} from {name} has come back as \"{reason}\". Refund it to {name}'s credit and record its return with that reason."),
                ops: vec![
                    op(&["refund", &order.to_string()]),
                    op(&["create_return", &order.to_string(), reason]),
                ],
            }
        }
        Kind::Restock => {
            let (sku, _) = &item_rows[0];
            if let Some(product) = world.products.iter_mut().find(|p| &p.0 == sku) {
                // Enough stock that "at least" can already hold.
                product.2 = product.2.max(3);
            }
            let stock = world.products.iter().find(|p| &p.0 == sku).map_or(0, |p| p.2);
            let minimum = if trap { stock - between(rng, 0, 2) } else { stock + between(rng, 1, 12) };
            let product = world.products.iter().find(|p| &p.0 == sku).map(|p| p.1.clone()).unwrap_or_default();
            Goal {
                text: format!("Product {sku} ({product}) must have at least {minimum} units in stock. Add only what is missing; if it already has enough, change nothing."),
                ops: if trap { vec![] } else { vec![op(&["restock", sku, &(minimum - stock).to_string()])] },
            }
        }
        Kind::Pay => Goal {
            text: format!("Pay order #{order} from {name}'s account credit if the credit covers the whole order total. If it does not, leave the order alone."),
            ops: if trap { vec![] } else { vec![op(&["apply_credit", &order.to_string()])] },
        },
    }
}

/// Rows no request touches: other customers (some sharing a requester's
/// name), products and orders.
fn add_noise(rows: u32, rng: &mut SplitMix64, world: &mut World) {
    let requesters: Vec<String> = world.customers.iter().map(|c| c.1.clone()).collect();
    for n in 0..i64::from(rows) {
        let customer = 1000 + n;
        let name = if n % 3 == 0 && !requesters.is_empty() {
            requesters[rng.below(requesters.len())].clone()
        } else {
            format!(
                "{} {}",
                FIRST[rng.below(FIRST.len())],
                LAST[rng.below(LAST.len())]
            )
        };
        world.customers.push((customer, name, between(rng, 0, 200)));
        let sku = format!("SKU-9{n:02}");
        world.products.push((
            sku.clone(),
            format!("{} {n}", THINGS[rng.below(THINGS.len())]),
            between(rng, 0, 40),
            between(rng, 5, 90),
        ));
        let order = 5000 + n;
        world
            .orders
            .push((order, customer, STATUSES[rng.below(STATUSES.len())]));
        world.items.push((order, sku, between(rng, 1, 3)));
    }
}

fn quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', "''"))
}

fn seed_sql(world: &World, distractors: u32) -> String {
    let mut sql = String::new();
    for (id, name, credit) in &world.customers {
        let _ = writeln!(
            sql,
            "INSERT INTO customers VALUES ({id}, {}, {credit});",
            quote(name)
        );
    }
    for (sku, name, stock, price) in &world.products {
        let _ = writeln!(
            sql,
            "INSERT INTO products VALUES ({}, {}, {stock}, {price});",
            quote(sku),
            quote(name)
        );
    }
    for (id, customer, status) in &world.orders {
        let _ = writeln!(
            sql,
            "INSERT INTO orders VALUES ({id}, {customer}, {});",
            quote(status)
        );
    }
    for (order, sku, qty) in &world.items {
        let _ = writeln!(
            sql,
            "INSERT INTO order_items VALUES ({order}, {}, {qty});",
            quote(sku)
        );
    }
    for table in 0..distractors {
        for row in 0..3 {
            let _ = writeln!(
                sql,
                "INSERT INTO notes_{table} VALUES ({row}, {});",
                quote(&format!("note {table}.{row}"))
            );
        }
    }
    sql
}

fn schema_sql(distractors: u32) -> String {
    let mut sql = SCHEMA_SQL.to_string();
    for table in 0..distractors {
        let _ = writeln!(
            sql,
            "CREATE TABLE notes_{table} (id INTEGER PRIMARY KEY, body TEXT NOT NULL);"
        );
    }
    sql
}

fn shell_word(word: &str) -> String {
    format!("'{}'", word.replace('\'', "'\\''"))
}

/// The database workflow plugin.
#[derive(Default)]
pub struct DbWorkflow;

impl DbWorkflow {
    /// A plugin value.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Plugin for DbWorkflow {
    fn id(&self) -> &'static str {
        DB_WORKFLOW
    }

    fn version(&self) -> &'static str {
        VERSION
    }

    fn verifier_class(&self) -> VerifierClass {
        VerifierClass::Objective
    }

    fn knobs(&self) -> &'static [Knob] {
        KNOBS
    }

    fn generate(&self, request: &InstanceRequest) -> Result<Instance, PluginError> {
        let difficulty = resolve_difficulty(self.id(), KNOBS, &request.difficulty)?;
        let (goals, rows, distractors) = (
            difficulty["goals"],
            difficulty["rows"],
            difficulty["distractor_tables"],
        );
        let traps = difficulty["traps"].min(goals - 1);
        let mut rng = SplitMix64::new(request.seed);

        let kinds: Vec<Kind> = (0..goals)
            .map(|_| {
                [
                    Kind::Cancel,
                    Kind::Ship,
                    Kind::Refund,
                    Kind::Restock,
                    Kind::Pay,
                ][rng.below(5)]
            })
            .collect();
        // The first `traps` goals able to be a trap are traps.
        let mut remaining = traps;
        let trapped: Vec<bool> = kinds
            .iter()
            .map(|kind| {
                if remaining > 0 && kind.can_trap() {
                    remaining -= 1;
                    true
                } else {
                    false
                }
            })
            .collect();

        let mut world = World::default();
        let built: Vec<Goal> = kinds
            .iter()
            .zip(&trapped)
            .enumerate()
            .map(|(index, (kind, trap))| build_goal(*kind, *trap, index, &mut rng, &mut world))
            .collect();
        add_noise(rows, &mut rng, &mut world);

        let mut instruction = String::from(
            "# Shop operations\n\nThe shop's orders are in the SQLite database `/app/shop.db`. `python3 /app/tools.py` is the shop's tool: run it without arguments to see its commands. It enforces the shop's rules and refuses what they do not allow. Carry out every request below and change nothing else; when you are done the database is compared with the state the requests lead to.\n\nRequests:\n",
        );
        for (n, goal) in built.iter().enumerate() {
            let _ = writeln!(instruction, "{}. {}", n + 1, goal.text);
        }
        let ops: Vec<&Vec<String>> = built.iter().flat_map(|g| &g.ops).collect();
        let ops_json =
            serde_json::to_string(&ops).map_err(|e| PluginError::Generation(e.to_string()))?;
        let (schema, seed) = (schema_sql(distractors), seed_sql(&world, distractors));

        let mut workspace = Folder::new();
        workspace.put_text("tools.py", TOOLS_PY)?;
        workspace.put_text("schema.sql", &schema)?;
        workspace.put_text("seed.sql", &seed)?;
        workspace.put_script(
            "reset.sh",
            "#!/bin/sh\n# The environment's reset: rebuild shop.db from schema.sql and seed.sql.\ncd \"$(dirname \"$0\")\" && rm -f shop.db && python3 -I tools.py init\n",
        )?;
        let mut tests = Folder::new();
        tests.put_script("test.sh", "#!/bin/sh\n# Final-state diff: see verify.py. No reward is written if the verifier itself fails.\npython3 -I \"${TESTS_DIR:-/tests}/verify.py\"\n")?;
        tests.put_text("verify.py", VERIFY_PY)?;
        tests.put_text("tools_ref.py", TOOLS_PY)?;
        tests.put_text("schema.sql", &schema)?;
        tests.put_text("seed.sql", &seed)?;
        tests.put_text("ops.json", &ops_json)?;
        let mut solve = String::from("#!/bin/sh\n# The reference: the requests, carried out through the shop's tool.\nset -e\n");
        for op in &ops {
            let words: Vec<String> = op.iter().map(|w| shell_word(w)).collect();
            let _ = writeln!(
                solve,
                "python3 -I \"${{APP_DIR:-/app}}/tools.py\" {}",
                words.join(" ")
            );
        }
        solve.push_str("exit 0\n");
        let mut solution = Folder::new();
        solution.put_script("solve.sh", &solve)?;

        let meta = EnvMeta {
            schema_version: SCHEMA_VERSION,
            environment: environment_id(self.id(), VERSION, &difficulty),
            plugin: self.id().into(),
            plugin_version: VERSION.into(),
            verifier_class: VerifierClass::Objective,
            verifier: format!("database-state-diff@{VERSION}"),
            difficulty,
            split: request.split,
            seed: request.seed,
            solver_tools: SolverTools::Shell,
        };
        Ok(Instance::assemble(
            meta,
            Parts {
                instruction,
                workspace,
                packages: vec!["python3".into()],
                tests,
                solution,
                category: "tool-use".into(),
            },
        )?)
    }

    fn novelty_text(&self, instance: &Instance) -> String {
        let text = instance.instruction();
        text.split_once("Requests:\n")
            .map_or(text, |(_, goals)| goals)
            .to_string()
    }

    fn attacks(&self, _instance: &Instance, _reference: &Reference) -> Vec<Attack> {
        // Each attack leaves the database in a state a careless verifier
        // might call good, or in one it might crash on.
        let sql = |statements: &str| {
            vec![
                Edit::Write {
                    path: "attack.sql".into(),
                    bytes: statements.as_bytes().to_vec(),
                },
                Edit::Shell(
                    "python3 -c \"import sqlite3; c = sqlite3.connect('shop.db'); c.executescript(open('attack.sql').read()); c.commit()\"".into(),
                ),
            ]
        };
        let attack = |name: &str, edits: Vec<Edit>| Attack {
            name: format!("db:{name}"),
            kind: AttackKind::Tamper,
            edits,
        };
        vec![
            attack("garbage-file", vec![Edit::Write { path: "shop.db".into(), bytes: b"not a database".to_vec() }]),
            attack("empty-file", vec![Edit::Write { path: "shop.db".into(), bytes: Vec::new() }]),
            attack("delete-all-rows", sql("DELETE FROM order_items; DELETE FROM orders; DELETE FROM returns; DELETE FROM customers; DELETE FROM products;")),
            attack("status-sweep", sql("UPDATE orders SET status = 'shipped';")),
            attack("credit-inflation", sql("UPDATE customers SET credit = credit + 1000000;")),
            attack("drop-tables", sql("DROP TABLE returns; DROP TABLE order_items;")),
            attack("views-in-place-of-tables", sql("ALTER TABLE orders RENAME TO orders_real; CREATE VIEW orders AS SELECT * FROM orders_real;")),
        ]
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // a panic is the failure report
mod tests {
    use super::*;
    use splinter_core::verifiability::Split;
    use std::collections::BTreeMap;

    fn request(seed: u64, goals: u32, traps: u32) -> InstanceRequest {
        InstanceRequest {
            split: Split::Dev,
            seed,
            difficulty: BTreeMap::from([
                ("goals".into(), goals),
                ("traps".into(), traps),
                ("rows".into(), 8),
                ("distractor_tables".into(), 1),
            ]),
        }
    }

    #[test]
    fn the_same_request_gives_the_same_folder_and_other_seeds_other_databases() {
        // seed: determinism and difference are what is tested
        let plugin = DbWorkflow::new();
        let a = plugin.generate(&request(1, 4, 1)).unwrap();
        assert_eq!(
            a.version(),
            plugin.generate(&request(1, 4, 1)).unwrap().version()
        );
        assert_ne!(
            a.version(),
            plugin.generate(&request(2, 4, 1)).unwrap().version()
        );
    }

    #[test]
    fn traps_leave_fewer_operations_than_requests_and_one_request_always_acts() {
        // seed: coverage over many seeds
        let plugin = DbWorkflow::new();
        for seed in 0..40 {
            let instance = plugin.generate(&request(seed, 3, 5)).unwrap(); // traps are capped at goals - 1
            let ops: Vec<Vec<String>> =
                serde_json::from_str(instance.folder().text("tests/ops.json").unwrap()).unwrap();
            assert!(!ops.is_empty(), "seed {seed}: the nop would pass");
        }
        let none = plugin.generate(&request(7, 4, 0)).unwrap();
        let ops: Vec<Vec<String>> =
            serde_json::from_str(none.folder().text("tests/ops.json").unwrap()).unwrap();
        assert!(ops.len() >= 4, "without traps every request acts");
    }

    #[test]
    fn the_verifier_and_the_expected_state_are_kept_from_the_solver() {
        let instance = DbWorkflow::new().generate(&request(3, 3, 1)).unwrap();
        let shown: Vec<_> = instance
            .solver_view()
            .paths()
            .map(ToString::to_string)
            .collect();
        assert!(shown.contains(&"environment/tools.py".to_string()));
        assert!(!shown
            .iter()
            .any(|p| p.contains("ops.json") || p.contains("verify.py")));
        assert!(
            !instance.instruction().contains("apply_credit "),
            "the instruction never names the operations"
        );
    }

    #[test]
    fn unknown_knobs_are_refused() {
        let mut bad = request(1, 2, 0);
        bad.difficulty.insert("tables".into(), 9);
        assert!(DbWorkflow::new().generate(&bad).is_err());
    }
}
