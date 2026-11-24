// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements held-out examinations of what a model
// learned from a person's writing, for its clients. If your team needs
// expertise in measuring a fine-tune with enough power to tell a gain from
// luck, you can procure our services by sending an email to
// info@swedishembedded.com.

//! What `exam-set` and the powered `exam` print as text.

use std::fmt::Write as _;

use splinter_sdk::exam_set::{ExamBuilt, ExamSet};
use splinter_sdk::powered::analysis::Comparison;
use splinter_sdk::powered::{Powered, PoweredExam};
use splinter_sdk::reserve::Reservation;

use crate::output::Report;

fn share(value: Option<f64>) -> String {
    value.map_or("-".into(), |v| format!("{v:.3}"))
}

/// One line saying what a reservation took: its families and what was left.
pub fn reservation_line(reserved: &Reservation) -> String {
    format!(
        "{} of {} families reserved ({} could have been); the sources the run reads from here \
         hold none of their text",
        reserved.families.len(),
        reserved.total_families,
        reserved.examinable
    )
}

/// One line saying what an exam set holds.
pub fn exam_set_line(exam: &ExamSet) -> String {
    format!(
        "exam {}: {} task(s) over {} of {} reserved families, frozen",
        exam.id,
        exam.tasks.len(),
        exam.families_examined(),
        exam.families.len()
    )
}

impl Report for ExamSet {
    fn human(&self) -> String {
        format!("{}\n", exam_set_line(self))
    }
}

impl Report for ExamBuilt {
    fn human(&self) -> String {
        let mut out = format!(
            "{}\n{}\n",
            reservation_line(&self.reservation),
            exam_set_line(&self.exam)
        );
        let _ = writeln!(
            out,
            "put a candidate to it: splinter exam CANDIDATE --exam-set {}",
            self.exam.id
        );
        out
    }
}

fn comparison(c: &Comparison) -> String {
    let interval = c.difference.map_or("-".into(), |i| {
        format!("{:+.3} [{:+.3}, {:+.3}]", i.mean, i.low, i.high)
    });
    let score = c.score_difference.map_or("-".into(), |i| {
        format!("{:+.3} [{:+.3}, {:+.3}]", i.mean, i.low, i.high)
    });
    let matched = c.length_matched.map_or("-".into(), |m| {
        format!(
            "{}-{} of {} (p = {:.4})",
            m.first_only, m.second_only, m.tasks, m.p_value
        )
    });
    let corrected = c.holm_p.map_or(
        " (the primary comparison, not corrected)".to_string(),
        |p| format!(" (Holm-corrected over families {p:.4})"),
    );
    format!(
        "  {} vs {}{corrected}: {} tasks, only {} right {}, only {} right {}, tied {}; p over tasks {:.4}, over \
         families {:.4}; share right {interval}; pass rate {score}; like-length answers {matched}; \
         {} family(ies), {} discordant task(s), a difference of {} would be seen at 80% power\n",
        c.first,
        c.second,
        c.tasks,
        c.first,
        c.first_only,
        c.second,
        c.second_only,
        c.both_right + c.both_wrong,
        c.p_tasks,
        c.p_families,
        c.power.families,
        c.power.discordant_tasks,
        share(c.power.minimum_detectable_difference)
    )
}

impl Report for Powered {
    fn human(&self) -> String {
        let mut out = format!(
            "exam {} on candidate {}: {} of {} task(s) over {} family(ies) ({} family(ies) the \
             candidate was trained on left out), {} answer(s) per task per arm\n",
            self.exam,
            self.candidate,
            self.tasks,
            self.tasks_in_exam,
            self.families,
            self.families_trained_on,
            self.resamples
        );
        let _ = writeln!(
            out,
            "judge {}: {} (precision on passes {}, on fails {})",
            self.judge.judge,
            if self.judge.trusted {
                "trusted"
            } else {
                "NOT trusted: no claim is made"
            },
            share(self.judge.calibration.precision_pass),
            share(self.judge.calibration.precision_fail)
        );
        if let Some(hard) = &self.hard_controls {
            let _ = writeln!(
                out,
                "  on {} harder wrong answers it passed {}",
                hard.wrong_answers, hard.passed
            );
        }
        for arm in &self.arms {
            let _ = writeln!(
                out,
                "  {} (asked under {}): right {}/{}, pass rate {}, invented {}/{}, answers {} characters on average",
                arm.arm,
                self.prompts.get(&arm.arm).map_or("?", String::as_str),
                arm.right,
                arm.judged,
                share(arm.mean_score),
                arm.invented,
                arm.checked,
                arm.mean_chars.map_or("-".into(), |c| format!("{c:.0}"))
            );
        }
        for c in &self.comparisons {
            out.push_str(&comparison(c));
        }
        for v in &self.voice {
            let _ = writeln!(
                out,
                "  voice, {}: loss {} (perplexity {}) over {} positions of the writer's own text",
                v.arm,
                share(v.loss.map(f64::from)),
                share(v.perplexity),
                v.positions
            );
        }
        if let Some(why) = &self.voice_error {
            let _ = writeln!(out, "  voice not scored: {why}");
        }
        for w in &self.training_warnings {
            let _ = writeln!(out, "warning: {w}");
        }
        out
    }
}

impl Report for PoweredExam {
    fn human(&self) -> String {
        match self {
            Self::Ran(powered) => powered.human(),
            Self::NotRun(why) => format!("the exam did not run: {why}\n"),
        }
    }
}

impl Report for crate::exam_session::PlannedExam {
    fn human(&self) -> String {
        let (p, r, s) = (&self.plan, &self.planned, &self.sign_test);
        let mut out = format!(
            "{} families of {} tasks ({} tasks), the first arm {:.0} points ahead, {:.0}% of \
             tasks discordant, intraclass correlation {:.2} (design effect {:.2}): the planned test \
             finds it in {:.0}% of {} simulated exams; with no difference it finds one in {:.1}%\n",
            p.families,
            p.tasks_per_family,
            r.tasks,
            p.effect * 100.0,
            p.discordance * 100.0,
            p.icc,
            r.design_effect,
            r.power * 100.0,
            p.replicates,
            r.false_positive_rate * 100.0
        );
        let lead = s.minimum_detectable_effect.map_or_else(
            || "none: the families cannot show a lead at this level".into(),
            |e| format!("{:.0} points", e * 100.0),
        );
        let _ = writeln!(
            out,
            "release gate (exact one-sided sign test, a family counts once, alpha {}): power {:.0}% at \
             the planned lead, smallest lead seen with {:.0}% power: {lead}",
            s.alpha,
            s.power * 100.0,
            s.wanted_power * 100.0
        );
        match s.critical_wins {
            Some(k) => {
                let _ = writeln!(
                    out,
                    "  at the expected discordance, {k} wins among the discordant families are needed"
                );
            }
            None => out.push_str(
                "  not even unanimity among the expected discordant families is significant\n",
            ),
        }
        if !s.adequate {
            out.push_str(
                "  UNDERPOWERED: a miss on this exam is inconclusive, never a pass or a fail\n",
            );
        }
        if !self.pilot.is_empty() {
            let _ = writeln!(out, "  PILOT: {}", self.pilot.join("; "));
        }
        out
    }
}

impl Report for crate::exam_session::LabelsExported {
    fn human(&self) -> String {
        format!(
            "{} answer(s) to label written to {}; the key is {}. Fill each item's label with \
             right or wrong, then: splinter exam-report labels-import {} --key {}\n",
            self.items,
            self.out.display(),
            self.key.display(),
            self.out.display(),
            self.key.display()
        )
    }
}

impl Report for splinter_sdk::powered::audit::Agreement {
    fn human(&self) -> String {
        format!(
            "{} labelled answer(s) the judge also decided ({} left out): agreement {}, kappa {}; \
             of what the judge called right the person called wrong {}, of what it called wrong the \
             person called right {}; agreement on short answers {}, on long {}; length bias {} ({})\n",
            self.items,
            self.left_out,
            share(self.agreement),
            share(self.kappa),
            share(self.false_right_rate),
            share(self.false_wrong_rate),
            share(self.agreement_short),
            share(self.agreement_long),
            share(self.length_bias),
            self.position_bias
        )
    }
}

impl Report for crate::exam_session::Memorisation {
    fn human(&self) -> String {
        let mut out = String::new();
        for a in &self.arms {
            let _ = writeln!(
                out,
                "  {}: {} answer(s); share of 8, 13 and 20 word runs found in the training text {}, {}, {}; \
                 longest run {} words; {} quote(s) of twenty words or more",
                a.arm,
                a.answers,
                share(a.mean_overlap_8),
                share(a.mean_overlap_13),
                share(a.mean_overlap_20),
                a.longest_run,
                a.quoting
            );
        }
        out
    }
}

impl Report for crate::exam_session::Picked {
    fn human(&self) -> String {
        let mut out = format!(
            "candidate {} on dev suite {}\n",
            self.selected.candidate, self.selected.exam
        );
        for s in &self.selected.scored {
            let _ = writeln!(
                out,
                "  step {:>5}: right {}/{}, invented {}/{}, voice loss {}, monitoring loss {:.3}",
                s.step,
                s.right,
                s.judged,
                s.invented,
                s.checked,
                s.voice_loss.map_or("-".into(), |v| format!("{v:.3}")),
                s.monitor_loss
            );
        }
        let chosen = &self.selected.chosen;
        let _ = writeln!(
            out,
            "chosen: step {} ({} in the running{})",
            chosen.step,
            chosen.in_the_running,
            if chosen.gate_passed {
                ""
            } else {
                "; NO checkpoint passed the gate on invented specifics"
            }
        );
        if let Some(adopted) = &self.adopted {
            let _ = writeln!(out, "adopted as candidate {adopted}");
        }
        out
    }
}

/// What the exam found, on one line: the judge's standing, each arm's judged
/// and grounded results, and the paired test.
fn exam_line(summary: &serde_json::Value) -> String {
    if let Some(why) = summary["not_run"].as_str() {
        return format!("not run: {why}");
    }
    let ran = &summary["ran"];
    let arm = |a: &serde_json::Value| {
        format!(
            "{}/{} judged right ({} unanswered), {}/{} invented a specific",
            a["judged_right"], a["judged"], a["unanswered"], a["invented"], a["checked"]
        )
    };
    let judge = &ran["judge"];
    let trust = if judge["trusted"] == true {
        "trusted"
    } else {
        "NOT trusted, so no claim"
    };
    let won = |t: &serde_json::Value, against: &str| {
        t.as_object().map_or(String::new(), |t| {
            format!(
                "; the candidate won {} of {} discordant task(s) against {against}, p = {:.4}",
                t["candidate_wins"],
                t["discordant"],
                t["p_value"].as_f64().unwrap_or(1.0)
            )
        })
    };
    let interval = ran["paired_interval"]
        .as_object()
        .map_or(String::new(), |i| {
            format!(
                " (gain per family {:.2}, interval {:.2} to {:.2})",
                i["mean"].as_f64().unwrap_or(0.0),
                i["low"].as_f64().unwrap_or(0.0),
                i["high"].as_f64().unwrap_or(0.0)
            )
        });
    let test = format!("{}{interval}", won(&ran["paired"], "the base"));
    let prompted = ran["prompted"].as_object().map_or(String::new(), |_| {
        format!(
            "; base asked under the goal {}{}",
            arm(&ran["prompted"]),
            won(&ran["paired_vs_prompted"], "it")
        )
    });
    let retrieval = ran["retrieval"].as_object().map_or(String::new(), |r| {
        let wins = ran["paired_retrieval"].as_object().map_or(String::new(), |t| {
            format!(
                "; retrieval won {} of {} discordant task(s), p = {:.4}",
                t["candidate_wins"],
                t["discordant"],
                t["p_value"].as_f64().unwrap_or(1.0)
            )
        });
        let right = |found: bool, key: &str| {
            r["by_task"].as_array().map_or(0, |tasks| {
                tasks
                    .iter()
                    .filter(|t| t["evidence_found"] == found && t[key] == true)
                    .count()
            })
        };
        format!(
            "; candidate with {} retrieved passage(s) {} (retrieval found the task's evidence for {}/{}; of those, right with the passages {} and alone {}; of the rest, with {} and alone {}){wins}",
            r["passages"],
            arm(&ran["retrieval"]["arm"]),
            r["hits"],
            r["tasks"],
            right(true, "with_passages"),
            right(true, "alone"),
            right(false, "with_passages"),
            right(false, "alone"),
        )
    });
    let warnings: String = ran["training_warnings"]
        .as_array()
        .map(|w| {
            w.iter()
                .filter_map(|w| w.as_str())
                .map(|w| format!("\nwarning: {w}"))
                .collect()
        })
        .unwrap_or_default();
    format!(
        "{} task(s) from {} source family(ies), judge {} {trust}: base {}; candidate {}{test}{prompted}{retrieval}{warnings}",
        ran["tasks"],
        ran["families"],
        judge["judge"].as_str().unwrap_or("?"),
        arm(&ran["base"]),
        arm(&ran["candidate"]),
    )
}

impl Report for splinter_sdk::exam::Exam {
    fn human(&self) -> String {
        exam_line(&serde_json::to_value(self).unwrap_or(serde_json::Value::Null)) + "\n"
    }
}
