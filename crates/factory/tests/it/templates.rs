// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning factories a team can start from a
// working example, for its clients. If your team needs expertise in model
// fine-tuning pipelines, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Templates: each is a task repository that works as written. Written into
//! an empty directory, every recipe of it validates; a template never
//! writes over a file that is already there.

use splinter_factory::resolve::resolve;
use splinter_factory::templates::{init, TEMPLATES};

use crate::repo::connections;

#[test]
fn every_template_is_a_task_repository_whose_recipes_validate() {
    // Adding or removing a template updates this list, and so its run test.
    let names: std::collections::BTreeSet<_> = TEMPLATES.iter().map(|t| t.name).collect();
    assert_eq!(
        names,
        [
            "agent-sft",
            "generations",
            "harness-search",
            "knowledge-learning",
            "repository-practice"
        ]
        .into()
    );
    for template in TEMPLATES {
        let dir = tempfile::tempdir().unwrap();
        let written = init(template.name, dir.path()).unwrap();
        assert!(!written.is_empty(), "{}", template.name);
        let recipes: Vec<_> = written
            .iter()
            .filter(|p| p.starts_with(dir.path().join("recipes")))
            .collect();
        assert!(!recipes.is_empty(), "{} has a recipe", template.name);
        for recipe in recipes {
            let text = std::fs::read_to_string(recipe).unwrap();
            if let Err(e) = resolve(dir.path(), &text, connections()) {
                panic!("{}: {}: {e:?}", template.name, recipe.display());
            }
        }
        // A verifier is a program the task's sandbox runs.
        for test in written.iter().filter(|p| p.ends_with("tests/test.sh")) {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(test).unwrap().permissions().mode();
            assert!(mode & 0o111 != 0, "{} is executable", test.display());
        }
    }
}

#[test]
fn a_template_never_writes_over_a_file_and_an_unknown_one_is_named() {
    let dir = tempfile::tempdir().unwrap();
    let recipe = dir.path().join("recipes/agent-sft.yaml");
    std::fs::create_dir_all(recipe.parent().unwrap()).unwrap();
    std::fs::write(&recipe, "mine").unwrap();
    let refused = init("agent-sft", dir.path()).unwrap_err().to_string();
    assert!(refused.contains("recipes/agent-sft.yaml"), "{refused}");
    assert_eq!(std::fs::read_to_string(&recipe).unwrap(), "mine");
    assert_eq!(
        std::fs::read_dir(dir.path()).unwrap().count(),
        1,
        "nothing else was written"
    );
    let unknown = init("no-such", dir.path()).unwrap_err().to_string();
    assert!(unknown.contains("agent-sft"), "{unknown}");
}
