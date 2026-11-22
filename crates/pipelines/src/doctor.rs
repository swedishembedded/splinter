// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning pipelines that tell you what is
// wrong before an expensive run starts, for its clients. If your team needs
// expertise in making long-running ML jobs fail early and clearly, you can
// procure our services by sending an email to info@swedishembedded.com.

//! `doctor`: whether a learning run can start here, answered without
//! loading a model, touching the network or changing what the state root
//! holds. Each check is cheap and independent; a failed one says what to do.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use splinter_core::model_ref::ModelRef;
use splinter_core::role::{Role, RoleOverrides};
use splinter_model::device::{devices, GpuCard};
use splinter_model::ModelSelection;
use splinter_orchestrator::cluster::{ClusterMode, ClusterSettings};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::roles;
use splinter_store::space::available_bytes;

/// Free space below which a run cannot be expected to finish: a checkpoint
/// copy, a dataset and an adapter do not fit.
pub const FAIL_BELOW_FREE_BYTES: u64 = 1 << 30;

/// Free space below which a run may run out part-way through. An estimate of
/// what a run on a small base needs, not a bound.
pub const WARN_BELOW_FREE_BYTES: u64 = 10 << 30;

/// How a check came out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Nothing to do.
    Ok,
    /// A run can start, and may meet trouble.
    Warn,
    /// A run cannot start.
    Fail,
}

/// One thing checked.
#[derive(Clone, Debug, Serialize)]
pub struct Check {
    /// What was checked: `state-writable`, `free-space`, `role:teacher`.
    pub name: String,
    /// How it came out.
    pub verdict: Verdict,
    /// What was found.
    pub detail: String,
    /// What to do about it, when it is not `ok`.
    pub fix: Option<String>,
}

impl Check {
    fn ok(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            verdict: Verdict::Ok,
            detail: detail.into(),
            fix: None,
        }
    }

    fn problem(
        verdict: Verdict,
        name: impl Into<String>,
        detail: impl Into<String>,
        fix: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            verdict,
            detail: detail.into(),
            fix: Some(fix.into()),
        }
    }
}

/// What `doctor` reports.
#[derive(Clone, Debug, Serialize)]
pub struct Doctor {
    /// Every check, in the order made.
    pub checks: Vec<Check>,
}

impl Doctor {
    /// Whether a run can start: no check failed. Warnings do not stop one.
    #[must_use]
    pub fn ready(&self) -> bool {
        self.checks.iter().all(|c| c.verdict != Verdict::Fail)
    }
}

/// Checks that a run can start in the state root and with the models `ctx`
/// is configured with.
pub fn doctor(ctx: &Context) -> Doctor {
    let mut checks = storage(ctx);
    checks.push(cluster_check(&ctx.config().cluster));
    checks.extend(distributed_check(ctx));
    checks.extend(gpu_check(ctx));
    checks.extend(roles_resolve(ctx));
    Doctor { checks }
}

/// The checks of the state root alone: that it can be written, and has room.
/// What a run needs before it may start, whatever models it uses.
pub fn storage(ctx: &Context) -> Vec<Check> {
    let root = ctx.root().path();
    // A state root not yet created is where the first run will create it:
    // what matters is the nearest directory that exists, and nothing is
    // created here.
    let anchor = nearest_existing(root);
    let mut checks = vec![state_writable(root, anchor)];
    // Free space needs a directory that exists; without a writable one the
    // first check has already said why.
    if checks[0].verdict == Verdict::Ok {
        checks.push(free_space(anchor));
    }
    checks
}

impl Check {
    /// This failed check as one sentence for a refusal: what was found and
    /// what to change.
    #[must_use]
    pub fn refusal(&self) -> String {
        match &self.fix {
            Some(fix) => format!("{}: {} ({fix})", self.name, self.detail),
            None => format!("{}: {}", self.name, self.detail),
        }
    }
}

/// `path`, else its nearest ancestor that exists.
fn nearest_existing(path: &Path) -> &Path {
    path.ancestors()
        .find(|ancestor| ancestor.exists())
        .unwrap_or(path)
}

fn state_writable(root: &Path, anchor: &Path) -> Check {
    let name = "state-writable";
    let probe = anchor.join(format!(".doctor-{}", std::process::id()));
    let outcome = fs::write(&probe, b"probe").and_then(|()| fs::remove_file(&probe));
    match outcome {
        Ok(()) => Check::ok(name, format!("{} accepts writes", anchor.display())),
        Err(e) => Check::problem(
            Verdict::Fail,
            name,
            format!(
                "cannot write under {} (for the state root {}): {e}",
                anchor.display(),
                root.display()
            ),
            "choose a writable directory with --state",
        ),
    }
}

fn free_space(root: &Path) -> Check {
    let name = "free-space";
    let free = match available_bytes(root) {
        Ok(free) => free,
        Err(e) => {
            return Check::problem(
                Verdict::Warn,
                name,
                format!(
                    "cannot measure the free space under {}: {e}",
                    root.display()
                ),
                "make sure the state root's filesystem is mounted and readable",
            )
        }
    };
    let detail = format!("{} free under {}", gib(free), root.display());
    let fix = "free space or point --state at a larger filesystem";
    if free < FAIL_BELOW_FREE_BYTES {
        Check::problem(Verdict::Fail, name, detail, fix)
    } else if free < WARN_BELOW_FREE_BYTES {
        Check::problem(Verdict::Warn, name, detail, fix)
    } else {
        Check::ok(name, detail)
    }
}

/// The first local weights file or directory `selection` needs that is not
/// there. A selection only names paths; whether they exist is for this check.
fn missing_weights(selection: &ModelSelection) -> Option<PathBuf> {
    let ModelSelection::Local(weights) = selection else {
        return None;
    };
    std::iter::once(&weights.base)
        .chain(weights.adapter.as_ref())
        .find(|path| !path.exists())
        .cloned()
}

/// The roles a model plays to help the policy learn: played by the policy
/// itself, they hold the run to what the policy already does.
fn helps_the_policy(role: Role) -> bool {
    matches!(
        role,
        Role::Teacher | Role::Generator | Role::Planner | Role::Judge | Role::Router
    )
}

/// Whether the model each role would get resolves to weights or an endpoint
/// that exists, without loading any of them.
fn roles_resolve(ctx: &Context) -> Vec<Check> {
    let assignments = match roles::assignments(ctx.config(), &RoleOverrides::new()) {
        Ok(assignments) => assignments,
        Err(e) => {
            return vec![Check::problem(
                Verdict::Fail,
                "roles",
                e.to_string(),
                "fix the model settings named in the message",
            )]
        }
    };
    Role::ALL
        .into_iter()
        .map(|role| {
            let name = format!("role:{role}");
            let reference = assignments.get(role);
            match ctx.selection(reference) {
                Ok(selection) => match missing_weights(&selection) {
                    None if helps_the_policy(role) && *reference == ModelRef::policy_default() => {
                        Check::problem(
                            Verdict::Warn,
                            name,
                            format!(
                                "the policy plays the {role} too ({}): a model rarely teaches, \
                                 judges or writes tasks above its own level",
                                selection.identity()
                            ),
                            "pull a larger chat model (`brain pull Qwen/Qwen3-8B`) for Splinter \
                             to pick, or name one with --assistant-model",
                        )
                    }
                    None => Check::ok(name, format!("{reference} is {}", selection.identity())),
                    Some(path) => Check::problem(
                        Verdict::Fail,
                        name,
                        format!("{reference} needs {}, which does not exist", path.display()),
                        "fetch the base checkpoint, or set the model store (see `splinter status`)",
                    ),
                },
                Err(e) => Check::problem(
                    Verdict::Fail,
                    name,
                    format!("{reference} does not resolve: {e}"),
                    "point the role at a model that exists (see `splinter status`)",
                ),
            }
        })
        .collect()
}

/// What the batch allocation this process runs inside is, and which of its
/// nodes a run uses: models are served on every one of them when it spans
/// nodes. An allocation that does not parse fails the check, by the value.
pub fn cluster_check(cluster: &ClusterSettings) -> Check {
    let name = "cluster";
    if cluster.mode == ClusterMode::Off {
        return Check::ok(name, "--cluster off: everything runs on this machine");
    }
    match cluster.usable() {
        Err(e) => Check::problem(
            Verdict::Fail,
            name,
            format!("the allocation cannot be used: {e}"),
            "fix the value named, or run with --cluster off",
        ),
        Ok(None) => Check::ok(name, "no batch allocation: everything runs on this machine"),
        Ok(Some(usable)) => {
            let allocation = &usable.allocation;
            let this = allocation.this_node.as_deref().unwrap_or("unknown");
            let found = format!(
                "job {}: nodes {} ({} GPUs each), this node {this}",
                allocation.job_id,
                allocation.nodes.join(","),
                allocation.gpus_per_node
            );
            if !usable.spans_nodes() {
                return Check::ok(
                    name,
                    format!("{found}; one node: everything runs on this machine"),
                );
            }
            let how = if usable.chosen {
                "chosen by --cluster-nodes"
            } else {
                "every node of the allocation"
            };
            let using = format!(
                "{found}; using {} ({how}, {} GPUs): a brain server per node",
                usable.nodes.join(","),
                usable.cards()
            );
            let training = match (usable.nodes.len(), allocation.gpus_per_node) {
                (servers, cards) if servers > 1 => format!(
                    "a supervised or policy-gradient run trains as one data-parallel job over \
                     every card of the {servers} servers ({cards} each)"
                ),
                (_, cards) if cards > 1 => format!(
                    "a supervised or policy-gradient run trains on all {cards} cards of the server"
                ),
                _ => "a training run uses one card".to_string(),
            };
            let using =
                format!("{using}; {training}, where the servers' brain/train takes cards and job");
            let staging = cluster.staging();
            let using = match &staging.dir {
                Some(dir) => format!(
                    "{using}; models staged on each node under {} ({}; checked on the nodes when \
                     a step starts)",
                    dir.display(),
                    staging.source
                ),
                None => format!(
                    "{using}; models read from the shared store: {}",
                    staging.source
                ),
            };
            if cluster.srun.is_none() {
                return Check::problem(
                    Verdict::Fail,
                    name,
                    format!("{using}, but no srun is known to start them with"),
                    "put srun on PATH or name it with --slurm-srun, or run with --cluster off",
                );
            }
            Check::ok(name, using)
        }
    }
}

/// Which brain actions can run as the ranks of one job across the
/// allocation, as the `brain` binary's own manifests say; no check outside an
/// allocation that spans nodes, or without a binary to ask.
fn distributed_check(ctx: &Context) -> Option<Check> {
    let spans = ctx
        .config()
        .cluster
        .usable()
        .ok()
        .flatten()
        .is_some_and(|usable| usable.spans_nodes());
    let binary = ctx.config().brain_binary.as_ref().filter(|_| spans)?;
    let name = "distributed-training";
    Some(match splinter_model::distributed::brain_actions(binary) {
        Err(e) => Check::problem(
            Verdict::Warn,
            name,
            format!("brain could not say which actions join a job plan: {e}"),
            "name a `brain` binary that lists its actions with --brain-bin",
        ),
        Ok(actions) => {
            let able: Vec<String> = actions
                .iter()
                .filter(|a| a.joins_plan)
                .map(|a| format!("{} {}", a.model, a.action))
                .collect();
            if able.is_empty() {
                Check::problem(
                    Verdict::Warn,
                    name,
                    "this brain lists no action that joins a job plan: a job runs on one card",
                    "use a brain build whose trainers take --job-plan",
                )
            } else {
                Check::ok(
                    name,
                    format!(
                        "actions that run as the ranks of one job: {}",
                        able.join(", ")
                    ),
                )
            }
        }
    })
}

const GIB: u64 = 1 << 30;

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / GIB as f64)
}

/// Whether some card has room for the policy's weights right now. Brain
/// places a model that no card can hold on the CPU backend without a word,
/// which turns a minutes-long run into hours; this says so first. An
/// estimate: it counts the weights, not the activations or the KV cache a run
/// adds on top.
pub fn gpu_fit(cards: &[GpuCard], weights_bytes: u64) -> Check {
    let name = "gpu";
    let fix = "free GPU memory (stop the processes using it) before the run starts";
    if cards.is_empty() {
        return Check::problem(
            Verdict::Warn,
            name,
            "no GPU found: a run would use the CPU backend, far slower",
            "run on a machine with a GPU",
        );
    }
    if !cards.iter().any(|card| card.measured) {
        return Check::problem(
            Verdict::Warn,
            name,
            "the free memory of the GPUs could not be read, so whether the weights fit is unknown",
            "check the brain server's device list (GET /v1/devices)",
        );
    }
    // With a server on every node, a card is named by its endpoint too.
    let several = cards.iter().any(|card| card.endpoint != cards[0].endpoint);
    let label = |card: &GpuCard| {
        if several {
            format!("endpoint {} gpu{}", card.endpoint, card.index)
        } else {
            format!("gpu{}", card.index)
        }
    };
    let listing = cards
        .iter()
        .filter(|card| card.measured)
        .map(|card| format!("{} {} free", label(card), gib(card.free_bytes)))
        .collect::<Vec<_>>()
        .join(", ");
    match cards
        .iter()
        .filter(|card| card.measured && card.free_bytes >= weights_bytes)
        .max_by_key(|card| card.free_bytes)
    {
        Some(best) => Check::ok(
            name,
            format!(
                "the policy's weights ({}) fit on {} ({listing}); training needs more for activations",
                gib(weights_bytes),
                label(best)
            ),
        ),
        None => Check::problem(
            Verdict::Warn,
            name,
            format!(
                "the policy's weights ({}) fit on no GPU right now ({listing}): brain will run it on the CPU backend",
                gib(weights_bytes)
            ),
            fix,
        ),
    }
}

/// [`gpu_fit`] for the policy this context trains, when it is a local model;
/// no check for one served over a network.
fn gpu_check(ctx: &Context) -> Option<Check> {
    let selection = ctx.selection(&ModelRef::policy_default()).ok()?;
    let ModelSelection::Local(weights) = selection else {
        return None;
    };
    // The files directly in the checkpoint directory; an `adapters`
    // directory beside them is not loaded with the base.
    let bytes: u64 = fs::read_dir(&weights.base)
        .ok()?
        .filter_map(Result::ok)
        .filter_map(|entry| entry.metadata().ok())
        .filter(|meta| meta.is_file())
        .map(|meta| meta.len())
        .sum();
    if bytes == 0 {
        return None;
    }
    // The cards are the brain server's to report: the run's own, started now
    // if it is not running yet.
    let cards =
        match ctx
            .brain()
            .map_err(|e| e.to_string())
            .and_then(|brain| devices(&brain).map_err(|e| e.to_string()))
        {
            Ok(cards) => cards,
            Err(reason) => return Some(Check::problem(
                Verdict::Warn,
                "gpu",
                format!("the brain server could not be asked for its cards: {reason}"),
                "name a running server with --brain-endpoint, or a `brain` binary with --brain-bin",
            )),
        };
    Some(gpu_fit(&cards, bytes))
}
