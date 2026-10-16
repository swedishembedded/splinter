// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: a plan is valid only if its families are domain-disjoint across
//! roles, no sealed family is ever scheduled for training or retention, each
//! generation holds itself to earlier families only, and a run is admitted
//! only for a family the plan allows in its generation.

use super::*;

/// A catalog of `(id, language, domain)` families, each with the tree id
/// `tree-<id>`.
fn catalog(families: &[(&str, &str, &str)]) -> String {
    families
        .iter()
        .map(|(id, language, domain)| {
            format!(
                "[[family]]\nid = \"{id}\"\nlanguage = \"{language}\"\ndomain = \"{domain}\"\n\
                 tree = \"tree-{id}\"\n"
            )
        })
        .collect()
}

const FAMILIES: [(&str, &str, &str); 5] = [
    ("a", "python", "one"),
    ("b", "lua", "two"),
    ("c", "python", "three"),
    ("d", "lua", "four"),
    ("e", "python", "five"),
];

const GOOD: &str = r#"
[plan]
name = "t"
[[generation]]
train = ["a", "b"]
retention = []
[[generation]]
train = ["c"]
retention = ["a", "b"]
[sealed]
families = ["d", "e"]
"#;

fn violations(plan: &str, catalog: &str) -> Vec<Violation> {
    match ExperimentPlan::parse(plan, catalog) {
        Err(PlanError::Invalid(found)) => found,
        other => panic!("expected a refused plan, got {other:?}"),
    }
}

#[test]
fn a_plan_whose_families_are_disjoint_and_sealed_is_valid() {
    let plan = ExperimentPlan::parse(GOOD, &catalog(&FAMILIES)).unwrap();
    assert_eq!(plan.generations(), 2);
    assert_eq!(plan.sealed(), ["d", "e"]);
}

#[test]
fn a_family_sharing_a_domain_with_another_role_is_refused() {
    let shared = [
        ("a", "python", "one"),
        ("b", "lua", "two"),
        ("c", "python", "three"),
        ("d", "lua", "four"),
        ("e", "python", "one"),
    ];
    let found = violations(GOOD, &catalog(&shared));
    assert!(
        found.contains(&Violation::SharedDomain {
            domain: "one".into(),
            families: vec!["a".into(), "e".into()],
        }),
        "{found:?}"
    );
}

#[test]
fn a_sealed_family_may_not_be_trained_or_retained_or_scheduled_twice() {
    let plan = r#"
[plan]
name = "t"
[[generation]]
train = ["a", "d"]
retention = []
[[generation]]
train = ["c", "a"]
retention = ["d"]
[sealed]
families = ["d", "e"]
"#;
    let found = violations(plan, &catalog(&FAMILIES));
    assert!(found.contains(&Violation::SealedScheduled {
        family: "d".into(),
        generation: 1
    }));
    assert!(found.contains(&Violation::SealedScheduled {
        family: "d".into(),
        generation: 2
    }));
    assert!(
        found.contains(&Violation::Repeated { family: "a".into() }),
        "{found:?}"
    );
}

#[test]
fn retention_names_only_families_an_earlier_generation_trained() {
    let plan = r#"
[plan]
name = "t"
[[generation]]
train = ["a"]
retention = ["b"]
[[generation]]
train = ["b"]
retention = ["c"]
[sealed]
families = ["e"]
"#;
    let found = violations(plan, &catalog(&FAMILIES));
    assert!(found.contains(&Violation::RetentionNotYetTrained {
        family: "b".into(),
        generation: 1
    }));
    assert!(found.contains(&Violation::RetentionNotYetTrained {
        family: "c".into(),
        generation: 2
    }));
}

#[test]
fn unknown_families_and_empty_stages_and_a_missing_tree_are_refused() {
    let plan = r#"
[plan]
name = "t"
[[generation]]
train = []
retention = []
[[generation]]
train = ["zz"]
retention = []
[sealed]
families = []
"#;
    let found = violations(plan, &catalog(&FAMILIES));
    assert!(found.contains(&Violation::EmptyGeneration { generation: 1 }));
    assert!(found.contains(&Violation::Unknown {
        family: "zz".into()
    }));
    assert!(found.contains(&Violation::NothingSealed));

    let no_tree = "[[family]]\nid = \"a\"\nlanguage = \"python\"\ndomain = \"one\"\n";
    let plan = "[plan]\nname=\"t\"\n[[generation]]\ntrain=[\"a\"]\nretention=[]\n[sealed]\nfamilies=[\"a\"]\n";
    assert!(violations(plan, no_tree).contains(&Violation::NoTree { family: "a".into() }));
}

#[test]
fn a_run_is_admitted_only_for_a_family_the_plan_allows_in_that_generation() {
    let plan = ExperimentPlan::parse(GOOD, &catalog(&FAMILIES)).unwrap();
    assert_eq!(plan.admit(1, "tree-a").unwrap(), "a");
    assert_eq!(plan.admit(2, "tree-c").unwrap(), "c");
    // Another generation's family, a sealed one, one outside the plan, and a
    // generation the plan does not have are each refused with the reason.
    for (generation, tree, why) in [
        (2, "tree-a", "generation 1"),
        (1, "tree-d", "sealed"),
        (2, "tree-e", "sealed"),
        (1, "tree-zz", "no family"),
        (3, "tree-a", "has 2 generation"),
        (0, "tree-a", "from 1"),
    ] {
        let refused = plan.admit(generation, tree).unwrap_err();
        assert!(refused.to_string().contains(why), "{refused}");
    }
}

#[test]
fn the_plan_the_repository_ships_is_valid_against_its_catalog() {
    let tasks = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tasks");
    let plan = std::fs::read_to_string(tasks.join("e3-plan.toml")).unwrap();
    let catalog = std::fs::read_to_string(tasks.join("catalog.toml")).unwrap();
    let plan = ExperimentPlan::parse(&plan, &catalog).unwrap();
    assert_eq!(plan.generations(), 3);
    assert_eq!(plan.sealed().len(), 2);
}
