// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a skill is a hypothesis. How far it is believed, and its status,
//! are computed from the evidence for and against it, never stored as truth.
#![allow(clippy::unwrap_used)]

mod common;

use common::{attempt, skill, Scratch};
use splinter_expdb::analyze::SkillStatus;
use splinter_expdb::model::{Body, Outcome, SkillEvidence, Stance};
use splinter_expdb::{RecordId, WriterIdentity};

struct Fixture {
    db: splinter_expdb::Database,
    collector: splinter_expdb::ingest::Collector,
    experiences: Vec<RecordId>,
}

fn fixture(scratch: &Scratch) -> Fixture {
    let db = scratch.open();
    let mut collector = db
        .collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap();
    let (decisions, _) = attempt(&mut collector, 1, "p", 12, Outcome::Pass);
    Fixture {
        db,
        collector,
        experiences: decisions.iter().map(|d| d.id).collect(),
    }
}

impl Fixture {
    fn skill(&mut self, name: &str) -> RecordId {
        self.collector.record(Body::Skill(skill(name))).unwrap()
    }

    fn evidence(&mut self, skill: RecordId, n: usize, stance: Stance, domain: &str, strength: f64) {
        self.collector
            .record(Body::SkillEvidence(SkillEvidence {
                skill,
                target: self.experiences[n],
                stance,
                domain: domain.into(),
                effect_size: Some(0.5),
                strength,
            }))
            .unwrap();
    }

    fn view(&mut self, skill: RecordId) -> splinter_expdb::analyze::SkillView {
        self.collector.flush().unwrap();
        self.db.snapshot().unwrap().skill(skill).unwrap().unwrap()
    }
}

#[test]
fn a_skill_with_no_evidence_is_only_proposed_and_has_no_confidence() {
    let scratch = Scratch::new();
    let mut f = fixture(&scratch);
    let s = f.skill("inspect first");
    let view = f.view(s);
    assert_eq!(view.assessment.status, SkillStatus::Proposed);
    assert_eq!(
        view.assessment.confidence, None,
        "no evidence is not zero confidence"
    );
    assert_eq!(view.assessment.transfer, None);
}

#[test]
fn a_little_supporting_evidence_makes_a_skill_observed() {
    let scratch = Scratch::new();
    let mut f = fixture(&scratch);
    let s = f.skill("inspect first");
    f.evidence(s, 0, Stance::Supports, "embedded", 1.0);
    f.evidence(s, 1, Stance::Supports, "embedded", 1.0);
    let view = f.view(s);
    assert_eq!(view.assessment.status, SkillStatus::Observed);
    // The posterior mean of a Beta(1 + supports, 1 + contradictions).
    assert!((view.assessment.confidence.unwrap() - 3.0 / 4.0).abs() < 1e-12);
}

#[test]
fn enough_verified_support_in_one_domain_validates_but_does_not_make_it_transferable() {
    let scratch = Scratch::new();
    let mut f = fixture(&scratch);
    let s = f.skill("inspect first");
    for n in 0..3 {
        f.evidence(s, n, Stance::Supports, "embedded", 1.0);
    }
    f.evidence(s, 3, Stance::Validates, "embedded", 1.0);
    let view = f.view(s);
    assert_eq!(view.assessment.status, SkillStatus::Validated);
    assert_eq!(view.assessment.domains, ["embedded"]);
}

#[test]
fn support_across_several_domains_makes_a_skill_transferable() {
    let scratch = Scratch::new();
    let mut f = fixture(&scratch);
    let s = f.skill("inspect first");
    for (n, domain) in ["embedded", "networking", "browsers", "databases"]
        .into_iter()
        .enumerate()
    {
        f.evidence(s, n, Stance::Supports, domain, 1.0);
    }
    f.evidence(s, 4, Stance::Validates, "networking", 1.0);
    let view = f.view(s);
    assert_eq!(view.assessment.status, SkillStatus::Transferable);
    assert_eq!(view.assessment.transfer, Some(1.0));
}

#[test]
fn contradicting_evidence_lowers_confidence_and_can_make_a_skill_obsolete() {
    let scratch = Scratch::new();
    let mut f = fixture(&scratch);
    let s = f.skill("retry with a longer timeout");
    f.evidence(s, 0, Stance::Supports, "networking", 1.0);
    f.evidence(s, 1, Stance::Supports, "networking", 1.0);
    let before = f.view(s).assessment.confidence.unwrap();

    for n in 2..5 {
        f.evidence(s, n, Stance::Contradicts, "networking", 1.0);
    }
    let after = f.view(s);
    assert!(after.assessment.confidence.unwrap() < before);
    assert_eq!(after.assessment.status, SkillStatus::Obsolete);
}

#[test]
fn weak_evidence_counts_for_less_than_strong_evidence() {
    let scratch = Scratch::new();
    let mut f = fixture(&scratch);
    let strong = f.skill("strong");
    let weak = f.skill("weak");
    f.evidence(strong, 0, Stance::Supports, "embedded", 1.0);
    f.evidence(weak, 1, Stance::Supports, "embedded", 0.2);
    let (strong, weak) = (f.view(strong), f.view(weak));
    assert!(strong.assessment.confidence.unwrap() > weak.assessment.confidence.unwrap());
}

#[test]
fn skills_with_enough_evidence_to_test_but_little_confidence_are_listed_for_follow_up() {
    let scratch = Scratch::new();
    let mut f = fixture(&scratch);
    let unsure = f.skill("unsure");
    let sure = f.skill("sure");
    let thin = f.skill("thin");
    for n in 0..4 {
        f.evidence(
            unsure,
            n,
            if n < 2 {
                Stance::Supports
            } else {
                Stance::Contradicts
            },
            "embedded",
            1.0,
        );
        f.evidence(sure, n, Stance::Supports, "embedded", 1.0);
    }
    f.evidence(thin, 0, Stance::Contradicts, "embedded", 1.0);
    f.collector.flush().unwrap();

    let wanted = f.db.snapshot().unwrap().uncertain_skills(0.6, 3).unwrap();
    let names: Vec<_> = wanted.iter().map(|v| v.skill.name.as_str()).collect();
    assert_eq!(names, ["unsure"]);
}
