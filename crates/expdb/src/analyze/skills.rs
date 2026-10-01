// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Skills are hypotheses. Their confidence and status are computed from the
//! evidence for and against them, by a policy stated in one place.

use std::collections::BTreeMap;

use crate::error::Result;
use crate::id::RecordId;
use crate::manifest::Snapshot;
use crate::model::{Body, RecordKind, Skill, SkillEvidence, Stance};

/// How far a skill has got.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SkillStatus {
    /// A hypothesis with no evidence yet.
    Proposed,
    /// Seen to work, but not enough to rely on.
    Observed,
    /// Enough strong support, and a deliberate test of it, to rely on.
    Validated,
    /// Validated, and supported across several domains.
    Transferable,
    /// More evidence against than for.
    Obsolete,
}

/// The thresholds that turn evidence into a status.
#[derive(Debug, Clone, Copy)]
pub struct SkillPolicy {
    /// Total supporting strength needed to validate.
    pub validated_support: f64,
    /// Confidence needed to validate.
    pub validated_confidence: f64,
    /// Supporting domains needed to be transferable.
    pub transfer_domains: usize,
    /// Share of domains that must support it to be transferable.
    pub transfer_share: f64,
    /// Pieces of evidence needed before a skill can be called obsolete.
    pub obsolete_min_evidence: usize,
}

impl Default for SkillPolicy {
    fn default() -> Self {
        Self {
            validated_support: 3.0,
            validated_confidence: 0.7,
            transfer_domains: 3,
            transfer_share: 0.6,
            obsolete_min_evidence: 3,
        }
    }
}

/// What the evidence says about a skill.
#[derive(Debug, Clone, PartialEq)]
pub struct SkillAssessment {
    /// Total strength of evidence for the skill.
    pub supports: f64,
    /// Total strength of evidence against it.
    pub contradicts: f64,
    /// Pieces of evidence.
    pub evidence_count: usize,
    /// Domains with supporting evidence, sorted.
    pub domains: Vec<String>,
    /// The share of observed domains in which support outweighs
    /// contradiction. `None` with no evidence.
    pub transfer: Option<f64>,
    /// The posterior mean of a Beta(1 + support, 1 + contradiction). `None`
    /// with no evidence: a prior is not a measurement.
    pub confidence: Option<f64>,
    /// How far the skill has got.
    pub status: SkillStatus,
}

/// Assesses evidence under a policy.
pub fn assess(evidence: &[SkillEvidence], policy: &SkillPolicy) -> SkillAssessment {
    let (mut supports, mut contradicts) = (0.0, 0.0);
    let mut validations = 0;
    let mut per_domain: BTreeMap<&str, (f64, f64)> = BTreeMap::new();
    for e in evidence {
        let entry = per_domain.entry(e.domain.as_str()).or_default();
        match e.stance {
            Stance::Supports | Stance::Validates => {
                supports += e.strength;
                entry.0 += e.strength;
                validations += usize::from(e.stance == Stance::Validates);
            }
            Stance::Contradicts => {
                contradicts += e.strength;
                entry.1 += e.strength;
            }
        }
    }
    let domains: Vec<String> = per_domain
        .iter()
        .filter(|(_, (s, _))| *s > 0.0)
        .map(|(d, _)| (*d).to_owned())
        .collect();
    let supporting = per_domain.values().filter(|(s, c)| s > c).count();
    let (transfer, confidence) = if evidence.is_empty() {
        (None, None)
    } else {
        (
            Some(supporting as f64 / per_domain.len() as f64),
            Some((1.0 + supports) / (2.0 + supports + contradicts)),
        )
    };
    let status = if evidence.is_empty() {
        SkillStatus::Proposed
    } else if contradicts > supports && evidence.len() >= policy.obsolete_min_evidence {
        SkillStatus::Obsolete
    } else if supports >= policy.validated_support
        && validations > 0
        && confidence.is_some_and(|c| c >= policy.validated_confidence)
    {
        if supporting >= policy.transfer_domains
            && transfer.is_some_and(|t| t >= policy.transfer_share)
        {
            SkillStatus::Transferable
        } else {
            SkillStatus::Validated
        }
    } else {
        SkillStatus::Observed
    };
    SkillAssessment {
        supports,
        contradicts,
        evidence_count: evidence.len(),
        domains,
        transfer,
        confidence,
        status,
    }
}

/// A skill with its evidence assessed.
#[derive(Debug, Clone, PartialEq)]
pub struct SkillView {
    /// The skill's record id.
    pub id: RecordId,
    /// The skill.
    pub skill: Skill,
    /// What the evidence says.
    pub assessment: SkillAssessment,
}

impl Snapshot {
    fn skills_with(&self, policy: &SkillPolicy) -> Result<Vec<SkillView>> {
        let index = self.index()?;
        let mut evidence: BTreeMap<RecordId, Vec<SkillEvidence>> = BTreeMap::new();
        for id in index.by_kind(RecordKind::SkillEvidence) {
            if let Some(Body::SkillEvidence(e)) = self.get(id)?.map(|r| r.body) {
                evidence.entry(e.skill).or_default().push(e);
            }
        }
        let mut views = Vec::new();
        for id in index.by_kind(RecordKind::Skill) {
            if let Some(Body::Skill(skill)) = self.get(id)?.map(|r| r.body) {
                let assessment = assess(evidence.get(&id).map_or(&[][..], Vec::as_slice), policy);
                views.push(SkillView {
                    id,
                    skill,
                    assessment,
                });
            }
        }
        Ok(views)
    }

    /// Every skill, assessed under the default policy.
    pub fn skills(&self) -> Result<Vec<SkillView>> {
        self.skills_with(&SkillPolicy::default())
    }

    /// One skill, assessed under the default policy.
    pub fn skill(&self, id: RecordId) -> Result<Option<SkillView>> {
        Ok(self.skills()?.into_iter().find(|v| v.id == id))
    }

    /// Skills with enough evidence to test further but too little
    /// confidence: where a new experiment would teach the most.
    pub fn uncertain_skills(
        &self,
        max_confidence: f64,
        min_evidence: usize,
    ) -> Result<Vec<SkillView>> {
        Ok(self
            .skills()?
            .into_iter()
            .filter(|v| {
                v.assessment.evidence_count >= min_evidence
                    && v.assessment.confidence.is_some_and(|c| c <= max_confidence)
            })
            .collect())
    }
}
