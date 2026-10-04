// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for telling a trained model's real
// gains from what a fixed answer already scores for its clients. If your team
// needs expertise in baselines and statistical power for model evaluation then
// you can procure our services by sending an email to info@swedishembedded.com.

//! The answer that needs no model: what a fixed template scores on the
//! transfer benchmark.
//!
//! A gain means something only above this. The template always says the
//! method applies, says nothing a check can fault, and when passages are
//! shown quotes the first words of the first one, so it is exactly as good
//! as the benchmark's rules are easy to satisfy and the mix of cases lets it
//! be.

use crate::curate::Document;
use crate::datasets::{Mode, TransferTask};

/// Words of the first passage the template quotes: enough to ground a line.
const QUOTE_WORDS: usize = 8;

/// The fixed answer to `task`: APPLIES, one sentence, and a block that quotes
/// the first evidence document when there is one.
#[must_use]
pub fn constant_answer(task: &TransferTask, docs: &[Document]) -> String {
    let mut lines = vec![
        "Applicability: APPLIES".to_string(),
        String::new(),
        "I would act as I have always done: state the matter plainly and let the people be heard."
            .to_string(),
        String::new(),
        "Grounding:".to_string(),
    ];
    let quoted = (task.mode == Mode::Retrieval)
        .then(|| task.evidence_docs.first())
        .flatten()
        .and_then(|id| docs.iter().find(|d| &d.id == id))
        .map(|d| {
            let quote: Vec<&str> = d.body.split_whitespace().take(QUOTE_WORDS).collect();
            format!("- SOURCE_DIRECT: \"{}\" [{}]", quote.join(" "), d.id)
        });
    lines.push(quoted.unwrap_or_else(|| {
        "- SPECULATION: I take this to be like the matters I knew.".to_string()
    }));
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario::Case;
    use crate::testkit::doc;

    fn task(mode: Mode, case: Case) -> TransferTask {
        TransferTask {
            id: "t".into(),
            scenario_id: "s".into(),
            mode,
            prompt: "Situation: five teams disagree.".into(),
            case,
            observations: Vec::new(),
            evidence_docs: if mode == Mode::Retrieval {
                vec!["d1".into()]
            } else {
                Vec::new()
            },
        }
    }

    fn docs() -> Vec<Document> {
        vec![doc(
            "d1",
            1770,
            "A B",
            "Let the Committee write to every Town, that the Sense of the People may be known.",
        )]
    }

    /// The template is right exactly where "it applies" is, and grounded where
    /// there is a passage to quote: that is the score a model must beat.
    #[test]
    fn the_constant_answer_passes_a_fit_and_fails_a_case_where_the_method_does_not_apply() {
        let documents = docs();
        for mode in [Mode::Retrieval, Mode::Internalized] {
            let fit = task(mode, Case::Clear);
            assert!(
                fit.verdict(&constant_answer(&fit, &documents), &documents)
                    .correct,
                "{mode:?}"
            );
            let not_fit = task(mode, Case::MissingPrecondition);
            assert!(
                !not_fit
                    .verdict(&constant_answer(&not_fit, &documents), &documents)
                    .correct,
                "{mode:?}"
            );
        }
        let retrieval = constant_answer(&task(Mode::Retrieval, Case::Clear), &documents);
        assert!(retrieval.contains("Let the Committee write to every Town"));
    }
}
