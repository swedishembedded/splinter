// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements resumable training pipelines whose stages
// recompute exactly when what they depend on changes, for its clients. If
// your team needs expertise in reproducible, incrementally rebuilt training
// runs, you can procure our services by sending an email to
// info@swedishembedded.com.

//! What each stage's key says about code: the version of the stage's own
//! recipe, and the revisions of the services whose behaviour it consumes.
//!
//! A stage is recomputed when its result could differ, and only then. Its
//! key holds its configuration slice, the run seed and the digests of what it
//! reads (the ledger's part), plus two kinds of code identity:
//!
//! - **its recipe version**, a constant below for the sample code that
//!   decides the stage's result (the world generator, the task families and
//!   verifier, the expert and the episode protocol, the record assembly, the
//!   training and probe requests). It is bumped in the change that alters
//!   that behaviour. The pins in this module's tests fail when the
//!   generators' output changes without a bump;
//! - **the revisions of the services it consumes**: sven's for every
//!   episode (the agent loop and its tools decide what the model sees),
//!   the brain binary's digest for every stage brain computes or serves,
//!   bench's for the examination.
//!
//! The revision of this repository as a whole is in no stage key: a change
//! to the report, the terminal output or a comment recomputes nothing. It is
//! recorded in the run manifest, for provenance.

/// A stage's recipe: its name and the version of its behaviour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Recipe {
    /// The stage it versions.
    pub stage: &'static str,
    /// Bumped when the stage's result for the same inputs changes.
    pub version: u32,
}

impl Recipe {
    /// The value a key records for it.
    #[must_use]
    pub fn id(self) -> String {
        format!("{}/{}", self.stage, self.version)
    }
}

/// The world corpus: its generator, the worked trajectories (the expert and
/// the tool results as rendered), the held-out split and the tokenizer text.
pub const WORLD: Recipe = Recipe {
    stage: "world",
    version: 6,
};

/// The task families, their instance draw and the Harbor export.
pub const TASKS: Recipe = Recipe {
    stage: "tasks",
    version: 3,
};

/// One episode: the workspace, the expert's procedure, the agent settings,
/// the verifier and the records an accepted episode writes.
pub const EPISODE: Recipe = Recipe {
    stage: "episode",
    version: 6,
};

/// Gathering accepted episodes into a dataset.
pub const GATHER: Recipe = Recipe {
    stage: "gather",
    version: 3,
};

/// The tokenizer and random-init request.
pub const CHECKPOINT: Recipe = Recipe {
    stage: "checkpoint",
    version: 2,
};

/// A training request (pretraining, fine-tuning, improvement) and the data
/// mix it is given.
pub const TRAIN: Recipe = Recipe {
    stage: "train",
    version: 1,
};

/// A probe's summary and its raw-text continuation.
pub const PROBE: Recipe = Recipe {
    stage: "probe",
    version: 3,
};

/// The examination request.
pub const EXAMINE: Recipe = Recipe {
    stage: "examine",
    version: 4,
};

#[cfg(test)]
mod tests {
    use splinter_sdk::vocabulary::digest::Digest;

    use crate::config::WorldSize;
    use crate::tasks::{instances, Split};
    use crate::template::Template;

    /// Fails when the world generator, the worked trajectories, the template
    /// or the task draw writes other bytes for the same seed: bump the
    /// recipe of the stage that changed (`WORLD`, `TASKS`; `EPISODE` too when
    /// the expert changed) and then the pin.
    #[test]
    fn the_generators_are_pinned_to_their_recipe_versions() {
        // seed: a fixed seed is the pin; any value works once recorded.
        let seed = 41;
        let size = WorldSize {
            towns: 2,
            expressions: 4,
            programs: 2,
            conversations: 4,
            trajectories: 0,
            recall: 4,
        };
        let template = Template::default();
        let tools = crate::collect::tool_sigs();
        let docs = crate::world::corpus(seed, &size, &template, &tools, "system");
        let scratch = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let mut text = crate::world::to_jsonl(&docs);
        for instance in instances(seed, Split::World, 3) {
            let doc = crate::world::trajectory(
                &instance,
                &template,
                &tools,
                "system",
                &scratch.path().join("w"),
            )
            .unwrap_or_else(|e| panic!("{e}"));
            text.push_str(&crate::world::to_jsonl(&[doc]));
        }
        let draw: String = [Split::Train, Split::Explore, Split::Dev]
            .iter()
            .flat_map(|s| instances(seed, *s, 3))
            .map(|i| serde_json::to_string(&i).unwrap_or_default())
            .collect();
        let pins = [
            ("world (WORLD, EPISODE)", Digest::of(text.as_bytes())),
            (
                "template (WORLD, CHECKPOINT)",
                Digest::of(template.jinja().as_bytes()),
            ),
            ("tasks (TASKS)", Digest::of(draw.as_bytes())),
        ];
        let shown: Vec<String> = pins.iter().map(|(n, d)| format!("{n}: {d}")).collect();
        assert_eq!(
            shown, PINNED,
            "a generator changed: bump its recipe, then this pin"
        );
    }

    const PINNED: [&str; 3] = [
        "world (WORLD, EPISODE): blake3:58577f4ec08e1067c09d7bba8680f3fcb565503c6b42fe02fe68e62ada65c0f3",
        "template (WORLD, CHECKPOINT): blake3:354117630131a85c444d649decf8b227f76e57221451ea66e3cd18ede9066edc",
        "tasks (TASKS): blake3:6c61c10760bd20f4818483d942fe77497519be58e12ca723986db413879afa1a",
    ];
}
