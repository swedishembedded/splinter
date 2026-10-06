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

/// One line for the exam stage of a `learn` that reserved its exam.
pub fn powered_line(summary: &serde_json::Value) -> String {
    let ran = &summary["ran"];
    if ran.is_null() {
        return format!("not run: {}", summary["not_run"].as_str().unwrap_or("?"));
    }
    let arms: Vec<String> = ran["arms"]
        .as_array()
        .map(|arms| {
            arms.iter()
                .map(|a| {
                    format!(
                        "{} {}/{}",
                        a["arm"].as_str().unwrap_or("?"),
                        a["right"],
                        a["judged"]
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    let first = ran["comparisons"].get(0).map_or(String::new(), |c| {
        format!(
            "; {} vs {}: p over tasks {:.4}, over families {:.4}",
            c["first"].as_str().unwrap_or("?"),
            c["second"].as_str().unwrap_or("?"),
            c["p_tasks"].as_f64().unwrap_or(1.0),
            c["p_families"].as_f64().unwrap_or(1.0)
        )
    });
    format!(
        "{} task(s) over {} family(ies): {}{first}",
        ran["tasks"],
        ran["families"],
        arms.join(", ")
    )
}

impl Report for crate::exam_session::PlannedExam {
    fn human(&self) -> String {
        let (p, r) = (&self.plan, &self.planned);
        format!(
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
        )
    }
}
