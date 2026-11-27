// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning loops whose every generation can be
// explained from its own records, for its clients. If your team needs
// expertise in observing and steering agent training loops, you can procure
// our services by sending an email to info@swedishembedded.com.

//! What `observe` prints: each run's state at every generation as tables.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde::Serialize;
use splinter_sdk::factory::observe::capability::Evaluation;
use splinter_sdk::factory::observe::conduct::Conduct;
use splinter_sdk::factory::observe::data::{Data, Policy};
use splinter_sdk::factory::observe::probe::{ProbeReport, Verdict};
use splinter_sdk::factory::observe::training::Training;
use splinter_sdk::factory::observe::{AttemptStage, RunObservation};

use crate::output::Report;

/// The runs observed.
#[derive(Serialize)]
#[serde(transparent)]
pub struct Observed(pub Vec<RunObservation>);

fn num(v: Option<f64>, digits: usize) -> String {
    v.map_or_else(|| "-".to_owned(), |v| format!("{v:.digits$}"))
}

fn int<T: ToString>(v: Option<T>) -> String {
    v.map_or_else(|| "-".to_owned(), |v| v.to_string())
}

fn by_policy(m: &BTreeMap<Policy, usize>) -> String {
    let own = m.get(&Policy::Own).copied().unwrap_or(0);
    let other = m.get(&Policy::Other).copied().unwrap_or(0);
    format!("{} (own {own}, other {other})", own + other)
}

fn data(out: &mut String, d: &Data) {
    let _ = writeln!(
        out,
        "    data {} (target {}): train {} records / {} chars, validation {} / {}, new tasks {}, action sequences {}, supervised tokens {}",
        d.stage,
        d.target,
        d.train.records,
        d.train.supervised_characters,
        d.validation.records,
        d.validation.supervised_characters,
        d.new_tasks,
        d.action_sequences,
        int(d.supervised_tokens),
    );
    let _ = writeln!(
        out,
        "      {:<28} {:<6} {:>9} {:>12} {:>8} {:>11} {:>16} {:>9} {:>6}",
        "source",
        "policy",
        "train rec",
        "train chars",
        "val rec",
        "val chars",
        "pass (new tasks)",
        "task mean",
        "tasks"
    );
    for s in &d.sources {
        let pass = s.pass_rate.map_or("-".to_owned(), |p| {
            format!("{}/{} {:.3}", p.successes, p.total, p.value)
        });
        let _ = writeln!(
            out,
            "      {:<28} {:<6} {:>9} {:>12} {:>8} {:>11} {:>16} {:>9} {:>6}",
            s.stage,
            if s.policy == Policy::Own {
                "own"
            } else {
                "other"
            },
            s.train.records,
            s.train.supervised_characters,
            s.validation.records,
            s.validation.supervised_characters,
            pass,
            num(s.mean_task_pass_rate, 3),
            s.tasks_attempted,
        );
    }
    let _ = writeln!(
        out,
        "      own share: train records {}, train chars {}; validation records {}, validation chars {}",
        num(d.own_share_records, 3),
        num(d.own_share_characters, 3),
        num(d.validation_own_share_records, 3),
        num(d.validation_own_share_characters, 3),
    );
    match &d.frontier {
        Some(f) => {
            let _ = writeln!(
                out,
                "      frontier of {}: tasks failed {} / sometimes {} / passed {}; records from them {} / {} / {}, unattempted {}; frontier share {}",
                f.from.join(", "),
                f.failed_tasks,
                f.sometimes_tasks,
                f.passed_tasks,
                by_policy(&f.records_failed),
                by_policy(&f.records_sometimes),
                by_policy(&f.records_passed),
                f.records_unattempted,
                num(f.frontier_share, 3),
            );
        }
        None => {
            out.push_str("      frontier: the target's own policy attempted none of its tasks\n")
        }
    }
    if !d.replay.is_empty() {
        let replay: Vec<String> = d
            .replay
            .iter()
            .map(|(s, n)| format!("{s} ({})", int(*n)))
            .collect();
        let _ = writeln!(out, "      replay: {}", replay.join(", "));
    }
}

fn training(out: &mut String, t: &Training) {
    let _ = writeln!(
        out,
        "    training {}: epochs asked {}, steps {} (ran {}, selected {}, stopped early {}); {} epochs at the selected step, {} at the last ({} records/step {}, {} draws/epoch); tokens {} at {} tok/s; validation loss base {} -> first {}, selected {}, last {}",
        t.stage,
        num(t.epochs_asked, 3),
        int(t.steps),
        int(t.steps_completed),
        int(t.selected_step),
        int(t.stopped_early),
        num(t.epochs_at_selected, 3),
        num(t.epochs_completed, 3),
        int(t.records_per_step),
        if t.records_per_step_recorded { "recorded" } else { "from grad_accum and ranks" },
        num(t.draws_per_epoch, 0),
        int(t.tokens),
        num(t.tokens_per_second, 0),
        num(t.validation_loss_base, 4),
        num(t.validation_loss_first, 4),
        num(t.validation_loss_selected, 4),
        num(t.validation_loss_last, 4),
    );
}

fn evaluations(out: &mut String, evaluations: &[Evaluation]) {
    if evaluations.is_empty() {
        return;
    }
    let _ = writeln!(
        out,
        "    {:<18} {:<12} {:<26} {:>5} {:>10} {:>14} {:>14} {:>7} {:>7} {:>8} {:>19} {:>9} decision",
        "evaluation",
        "purpose",
        "task set",
        "tasks",
        "witnessed",
        "baseline",
        "candidate",
        "W/L",
        "p",
        "diff",
        "interval (tasks)",
        "uncounted"
    );
    for e in evaluations {
        let arm = |a: &splinter_sdk::factory::observe::capability::Arm| {
            a.pass_rate.map_or("-".to_owned(), |p| {
                format!("{}/{} {:.3}", p.successes, p.total, p.value)
            })
        };
        let interval = e.interval.map_or("-".to_owned(), |i| {
            format!("[{:+.3},{:+.3}]({})", i.low, i.high, e.interval_tasks)
        });
        let s = &e.task_set.solvability;
        let _ = writeln!(
            out,
            "    {:<18} {:<12} {:<26} {:>5} {:>10} {:>14} {:>14} {:>7} {:>7} {:>8} {:>19} {:>9} {}",
            e.stage,
            e.purpose.as_deref().unwrap_or("measurement"),
            e.task_set.label,
            e.task_set.tasks,
            format!("{}/{}/{}", s.witnessed, s.refuted, s.unknown),
            arm(&e.baseline),
            arm(&e.candidate_arm),
            format!("{}/{}", int(e.wins), int(e.losses)),
            num(e.p, 3),
            e.difference.map_or("-".to_owned(), |d| format!("{d:+.3}")),
            interval,
            num(e.uncounted_share, 3),
            e.decision.as_deref().unwrap_or("-"),
        );
    }
}

fn conduct_line(name: &str, c: &Conduct) -> String {
    let classes: Vec<String> = c
        .attribution
        .iter()
        .map(|(k, v)| {
            format!(
                "{} {v}",
                serde_json::to_value(k)
                    .ok()
                    .and_then(|v| v.as_str().map(str::to_owned))
                    .unwrap_or_default()
            )
        })
        .collect();
    let sides: Vec<String> = c.by_side.iter().map(|(k, v)| format!("{k} {v}")).collect();
    let sv = &c.self_verification;
    format!(
        "    {name}: {} attempts, {} counted, {} passed, {} failed [{}] by class [{}]; claims: {} passed, {} undetected failures (rate {}), {} unclaimed failed, {} unclaimed passed, disagreement {}\n",
        c.trials,
        c.counted,
        c.passed,
        c.failed,
        sides.join(", "),
        classes.join(", "),
        sv.claimed_passed,
        sv.undetected_failures,
        num(sv.undetected_failure_rate.map(|s| s.value), 3),
        sv.unclaimed_failed,
        sv.unclaimed_passed,
        num(sv.disagreement.map(|s| s.value), 3),
    )
}

fn attempts(out: &mut String, stages: &[AttemptStage]) {
    out.push_str("  attempt stages\n");
    let _ = writeln!(
        out,
        "    {:<28} {:>7} {:>8} {:>10} {:>12} {:>10} {:<14} uncounted",
        "stage", "trials", "counted", "per hour", "compl tok/s", "witnessed", "models"
    );
    for a in stages {
        let s = &a.solvability;
        let uncounted: Vec<String> = a
            .pace
            .uncounted
            .iter()
            .map(|(k, v)| format!("{k} {v}"))
            .collect();
        let _ = writeln!(
            out,
            "    {:<28} {:>7} {:>8} {:>10} {:>12} {:>10} {:<14} {}",
            a.stage,
            a.pace.trials,
            a.pace.counted,
            num(a.pace.attempts_per_hour, 0),
            num(a.pace.completion_tokens_per_second, 0),
            format!("{}/{}/{}", s.witnessed, s.refuted, s.unknown),
            if a.reference {
                "reference".to_owned()
            } else {
                a.models.len().to_string() + " model(s)"
            },
            uncounted.join(", "),
        );
    }
    out.push_str("  conduct (failures by what fixes them and by class; claims of completion against the grader)\n");
    for a in stages {
        out.push_str(&conduct_line(&a.stage, &a.conduct));
    }
}

impl Report for Observed {
    fn human(&self) -> String {
        let mut out = String::new();
        for run in &self.0 {
            let _ = writeln!(
                out,
                "run {} (interval seed {})",
                run.run.display(),
                run.seed
            );
            for g in &run.generations {
                let r = &g.release;
                let _ = writeln!(
                    out,
                    "  generation {}: {} promoted {}, champion {}, requires [{}]",
                    g.number,
                    r.stage,
                    r.promoted.as_deref().unwrap_or("none"),
                    r.champion.as_deref().unwrap_or("-"),
                    r.requires.join(", "),
                );
                for why in &r.why {
                    let _ = writeln!(out, "    why: {why}");
                }
                for c in &g.candidates {
                    let _ = writeln!(out, "    candidate {}", c.stage);
                    if let Some(d) = &c.data {
                        data(&mut out, d);
                    }
                    if let Some(t) = &c.training {
                        training(&mut out, t);
                    }
                    for card in &c.cards {
                        let _ = writeln!(
                            out,
                            "      gpu {}: sm {}%, mem {}% over {} samples",
                            card.gpu,
                            num(card.sm, 0),
                            num(card.mem, 0),
                            card.samples
                        );
                    }
                }
                evaluations(&mut out, &g.evaluations);
                for e in &g.evaluations {
                    for (arm, a) in [("baseline", &e.baseline), ("candidate", &e.candidate_arm)] {
                        if let Some(c) = &a.conduct {
                            out.push_str(&conduct_line(&format!("{}/{arm}", e.stage), c));
                        }
                    }
                }
            }
            attempts(&mut out, &run.attempts);
            let s = &run.solvability;
            let _ = writeln!(
                out,
                "  tasks pinned: {} witnessed, {} refuted, {} without a reference run",
                s.witnessed, s.refuted, s.unknown
            );
        }
        out
    }
}

/// What a probe of one knob found.
#[derive(Serialize)]
#[serde(transparent)]
pub struct Probed(pub ProbeReport);

impl Report for Probed {
    fn human(&self) -> String {
        let r = &self.0;
        let mut out = String::new();
        let _ = writeln!(
            out,
            "knob {} ({} = {}) moves {}: generation {}, control {}, probe {}, change {} (least {}): {}",
            r.knob,
            r.set,
            r.value,
            r.moves,
            r.generation,
            num(r.control, 4),
            num(r.probed, 4),
            r.change.map_or("-".to_owned(), |c| format!("{c:+.4}")),
            r.min_change,
            match r.verdict {
                Verdict::Moved => "moved",
                Verdict::DidNotMove => "did not move: the knob is a defect",
                Verdict::Unmeasured => "unmeasured",
            }
        );
        let share = |s: Option<splinter_sdk::factory::observe::Share>| {
            s.map_or("-".to_owned(), |p| {
                format!("{}/{} {:.3}", p.successes, p.total, p.value)
            })
        };
        for c in &r.capability {
            let _ = writeln!(
                out,
                "  {:<18} {:<12} {:<26} control {:>14}  probe {:>14}  probe - control {} over {} tasks: {}",
                c.evaluation,
                c.purpose.as_deref().unwrap_or("measurement"),
                c.task_set,
                share(c.control),
                share(c.probed),
                c.interval.map_or("-".to_owned(), |i| format!("{:+.3} [{:+.3},{:+.3}]", i.mean, i.low, i.high)),
                c.tasks,
                match c.followed {
                    Some(true) => "followed",
                    Some(false) => "did not follow",
                    None => "no interval",
                }
            );
        }
        out
    }
}
