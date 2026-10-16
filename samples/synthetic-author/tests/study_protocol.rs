// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the measurement of fine-tuned language
// models for its clients. If your team needs expertise in pre-registering a
// replication study so that no choice can follow its results, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a replication study is run from one protocol frozen before any run.
//! The protocol draws its authors' seeds once, names three different model
//! families, sizes the examination for the comparison it exists to settle,
//! and is refused in any other form once written. The statistics it names are
//! exact.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::collections::BTreeSet;
use std::path::PathBuf;

use synthetic_author::exam::items;
use synthetic_author::persona::Persona;
use synthetic_author::study::power::{derive, power, Observed};
use synthetic_author::study::protocol::{
    freeze, read, ModelRoles, Protocol, Seeds, Spec, DEFAULT_AUTHORS, DEFAULT_REPLICATES,
    EARLIER_OBSERVATION,
};
use synthetic_author::study::stats::{holm, sign_test_p, spread};

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "synthetic-author-study-protocol-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn roles() -> ModelRoles {
    ModelRoles {
        writer: "remote:north/aster-xl-4".into(),
        reader: "remote:south/birch-5".into(),
        miner: "remote:east/cedar-70b".into(),
    }
}

const EARLIER: Observed = EARLIER_OBSERVATION;

fn template() -> Protocol {
    Protocol::standard(
        roles(),
        derive(EARLIER, "an earlier examination", 0.05, 0.8).unwrap(),
    )
}

fn spec() -> Spec {
    Spec {
        template: template(),
        author_seeds: Seeds::Draw(DEFAULT_AUTHORS),
        training_seeds: Seeds::Draw(DEFAULT_REPLICATES),
    }
}

/// A source of fresh seeds that has never produced one twice.
fn fresh() -> impl FnMut() -> u64 {
    let base = splinter_sdk::model::seed::draw().value();
    let mut n = 0_u64;
    move || {
        n += 1;
        base.wrapping_add(n.wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }
}

#[test]
fn the_sign_test_over_authors_is_exact() {
    assert!((sign_test_p(8, 8) - 1.0 / 256.0).abs() < 1e-12, "0.0039");
    assert!((sign_test_p(7, 8) - 9.0 / 256.0).abs() < 1e-12, "0.035");
    assert!((sign_test_p(0, 8) - 1.0).abs() < 1e-12);
    assert!(
        (sign_test_p(0, 0) - 1.0).abs() < 1e-12,
        "no evidence, no claim"
    );
    // It agrees with the sign test every report uses, where that one is exact.
    let pairs: Vec<(bool, bool)> = (0..40).map(|i| (i % 3 != 0, i % 3 == 0)).collect();
    let reference = splinter_sdk::model::stats::sign_test(&pairs);
    let ours = sign_test_p(reference.candidate_wins, reference.discordant);
    assert!(
        (ours - reference.p_value).abs() < 1e-12,
        "{ours} {reference:?}"
    );
    // And it holds for sizes where a direct binomial coefficient overflows.
    let p = sign_test_p(1100, 2000);
    assert!(p.is_finite() && (0.0..=1.0).contains(&p), "{p}");
}

#[test]
fn holm_steps_down_over_the_whole_family_and_leaves_what_was_not_measured_absent() {
    let adjusted = holm(&[Some(0.01), Some(0.04), Some(0.03)], 3);
    assert_eq!(adjusted, [Some(0.03), Some(0.06), Some(0.06)]);
    // A comparison that was not measured has no adjusted value and still
    // counts in the family.
    let adjusted = holm(&[Some(0.01), None], 4);
    assert_eq!(adjusted, [Some(0.04), None]);
    // Adjusted values never exceed one.
    assert_eq!(holm(&[Some(0.6)], 3), [Some(1.0)]);
}

#[test]
fn the_spread_over_seeds_has_a_standard_deviation_only_from_two_seeds() {
    let s = spread(&[0.5, 0.7, 0.6]).unwrap();
    assert_eq!((s.seeds, s.min, s.max), (3, 0.5, 0.7));
    assert!((s.sd.unwrap() - 0.1).abs() < 1e-12);
    assert_eq!(spread(&[0.4]).unwrap().sd, None);
    assert!(spread(&[]).is_none());
}

#[test]
fn the_examination_is_sized_for_tuned_against_retrieval_from_the_discordance_observed() {
    let sizing = derive(EARLIER, "an earlier examination", 0.05, 0.8).unwrap();
    assert!(sizing.achieved_power >= 0.8);
    assert_eq!(sizing.items, sizing.per_cell * 4);
    // About 106 discordant pairs, about 460 items (the roadmap's estimate).
    assert!(
        (440..=480).contains(&sizing.items),
        "{} items",
        sizing.items
    );
    assert!(
        (100.0..=112.0).contains(&sizing.expected_discordant),
        "{}",
        sizing.expected_discordant
    );
    // One step smaller does not reach it: the size is the least one.
    let below = power(
        (sizing.per_cell - 1) * 4,
        sizing.discordance,
        sizing.win_share,
        0.05,
    );
    assert!(below < 0.8 || sizing.per_cell == 1, "{below}");
    // A larger effect needs fewer items; nothing is found where there is no effect.
    let easy = derive(
        Observed {
            items: 160,
            discordant: 37,
            candidate_wins: 35,
        },
        "x",
        0.05,
        0.8,
    )
    .unwrap();
    assert!(easy.items < sizing.items);
    let none = derive(
        Observed {
            items: 160,
            discordant: 37,
            candidate_wins: 18,
        },
        "x",
        0.05,
        0.8,
    );
    assert!(none.is_err(), "a split at or below even sizes nothing");
}

#[test]
fn an_examination_of_that_size_can_be_drawn_for_an_author() {
    let per_cell = template().exam.per_cell;
    let seed = splinter_sdk::model::seed::draw().value();
    let persona = Persona::generate(seed);
    let drawn = items(&persona, seed, per_cell, &BTreeSet::new());
    assert_eq!(drawn.len(), per_cell * 4, "author seed {seed}");
}

#[test]
fn freezing_draws_the_authors_seeds_once_and_reads_them_back() {
    let root = scratch("freeze");
    let frozen = freeze(&root, &spec(), &mut fresh()).unwrap();
    assert!(frozen.created);
    let protocol = &frozen.protocol;
    assert_eq!(protocol.authors.len(), DEFAULT_AUTHORS);
    let mut seeds: Vec<u64> = protocol.authors.iter().map(|a| a.seed).collect();
    seeds.sort_unstable();
    seeds.dedup();
    assert_eq!(seeds.len(), DEFAULT_AUTHORS, "author seeds are distinct");
    for author in &protocol.authors {
        assert_eq!(author.training_seeds.len(), DEFAULT_REPLICATES);
    }

    let again = read(&root).unwrap();
    assert_eq!(&again.protocol, protocol);
    assert_eq!(again.digest, frozen.digest);
    assert!(!again.created);

    // Freezing the same protocol again draws nothing and changes nothing.
    let mut never = || -> u64 { panic!("a frozen protocol is never redrawn") };
    let second = freeze(&root, &spec(), &mut never).unwrap();
    assert!(!second.created);
    assert_eq!(second.digest, frozen.digest);
}

#[test]
fn a_protocol_that_differs_from_the_frozen_one_is_refused_naming_the_field() {
    let root = scratch("differs");
    let frozen = freeze(&root, &spec(), &mut fresh()).unwrap();
    let mut other = spec();
    other.template.exam.per_cell += 1;
    other.template.models.writer = "remote:west/dune-9".into();
    let message = format!("{:#}", freeze(&root, &other, &mut fresh()).unwrap_err());
    assert!(
        message.contains("exam")
            && message.contains("models")
            && message.contains(frozen.digest.as_str()),
        "{message}"
    );
    // A different count of authors is a different protocol too.
    let mut fewer = spec();
    fewer.author_seeds = Seeds::Draw(DEFAULT_AUTHORS - 1);
    let message = format!("{:#}", freeze(&root, &fewer, &mut fresh()).unwrap_err());
    assert!(message.contains("authors"), "{message}");
    assert_eq!(
        read(&root).unwrap().digest,
        frozen.digest,
        "nothing was written"
    );
}

#[test]
fn the_digest_is_of_the_content_not_of_the_file() {
    let root = scratch("digest");
    let frozen = freeze(&root, &spec(), &mut fresh()).unwrap();
    let file = root.join("protocol.json");
    let compact: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
    std::fs::write(&file, compact.to_string()).unwrap();
    assert_eq!(read(&root).unwrap().digest, frozen.digest);

    let mut changed = frozen.protocol.clone();
    changed.training.steps += 1;
    assert_ne!(changed.digest(), frozen.digest);
    changed.training.steps -= 1;
    assert_eq!(changed.digest(), frozen.digest);
}

#[test]
fn given_seeds_are_used_as_they_are_and_recorded_as_given() {
    let root = scratch("given");
    let mut given = spec();
    given.author_seeds = Seeds::Given(vec![11, 22]);
    given.training_seeds = Seeds::Given(vec![1, 2, 3]);
    let frozen = freeze(&root, &given, &mut || panic!("nothing to draw")).unwrap();
    let protocol = &frozen.protocol;
    assert_eq!(protocol.authors.len(), 2);
    assert_eq!(protocol.authors[1].training_seeds, [1, 2, 3]);
    assert_eq!(
        serde_json::to_value(protocol.provenance).unwrap(),
        serde_json::json!({"authors": "given", "training": "given"})
    );
    let drawn = freeze(&scratch("drawn"), &spec(), &mut fresh()).unwrap();
    assert_eq!(
        serde_json::to_value(drawn.protocol.provenance).unwrap(),
        serde_json::json!({"authors": "drawn", "training": "drawn"})
    );
}

#[test]
fn a_protocol_whose_models_share_a_family_or_whose_comparisons_name_no_arm_is_not_frozen() {
    let refused = |edit: &dyn Fn(&mut Spec)| {
        let mut s = spec();
        edit(&mut s);
        let root = scratch("refused");
        let error = freeze(&root, &s, &mut fresh()).unwrap_err();
        assert!(!root.join("protocol.json").exists(), "nothing is written");
        format!("{error:#}")
    };
    let m = refused(&|s| s.template.models.reader = "remote:other/aster-small-4".into());
    assert!(
        m.contains("writer") && m.contains("reader") && m.contains("aster"),
        "{m}"
    );
    let m = refused(&|s| s.template.models.miner = "remote:other/Birch-2".into());
    assert!(m.contains("reader") && m.contains("miner"), "{m}");
    let m = refused(&|s| {
        s.template
            .comparisons
            .push(synthetic_author::study::protocol::Comparison::new(
                "tuned",
                "no-such-arm",
            ))
    });
    assert!(m.contains("no-such-arm"), "{m}");
    let m = refused(&|s| {
        s.template.primary = synthetic_author::study::protocol::Comparison::new("base", "named")
    });
    assert!(m.contains("primary"), "{m}");
    let m = refused(&|s| s.author_seeds = Seeds::Given(vec![5, 5]));
    assert!(m.contains("twice"), "{m}");
    let m = refused(&|s| s.template.arms.retain(|a| a != "tuned"));
    assert!(m.contains("tuned"), "{m}");
}

#[test]
fn each_adapter_arm_is_trained_at_the_seeds_the_protocol_names() {
    use synthetic_author::dataset::Briefings;
    use synthetic_author::study::arm_key::ArmKey;
    let frozen = freeze(&scratch("seeds"), &spec(), &mut fresh()).unwrap();
    let p = &frozen.protocol;
    let author = &p.authors[0];
    let tuned = ArmKey::parse("tuned").unwrap();
    let ablation = ArmKey::tuned(Briefings::ReaderOnly);
    assert_eq!(p.seeds_for(author, tuned), author.training_seeds);
    assert_eq!(
        p.seeds_for(author, ablation),
        author.training_seeds[..p.ablations.training_seeds]
    );
    assert!(p
        .seeds_for(author, ArmKey::parse("base").unwrap())
        .is_empty());
    let names: Vec<String> = p.arm_keys().unwrap().iter().map(|k| k.name()).collect();
    assert!(names.contains(&"tuned-reader-only".to_string()));
    assert!(names.contains(&"tuned-fixed-wording-only".to_string()));
    let trained = fresh()();
    assert_eq!(
        ablation.slot(trained),
        format!("tuned-reader-only-s{trained}")
    );
    assert_eq!(ArmKey::parse("retrieval").unwrap().slot(trained), "shared");
}

fn study(args: &[&str]) -> std::process::Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_synthetic-author"))
        .arg("study")
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn the_freeze_command_prints_every_seed_it_drew_and_repeats_without_redrawing() {
    let root = scratch("cli");
    let freeze_args = [
        "freeze",
        "--out",
        root.to_str().unwrap(),
        "--writer",
        "remote:north/aster-xl-4",
        "--reader",
        "remote:south/birch-5",
        "--miner",
        "remote:east/cedar-70b",
    ];
    let first = study(&freeze_args);
    let text = String::from_utf8_lossy(&first.stdout).to_string();
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let frozen = read(&root).unwrap();
    assert!(
        text.contains("frozen") && text.contains(frozen.digest.as_str()),
        "{text}"
    );
    for author in &frozen.protocol.authors {
        assert!(
            text.contains(&format!("author seed {}", author.seed)),
            "{text}"
        );
        for seed in &author.training_seeds {
            assert!(text.contains(&seed.to_string()), "{text}");
        }
    }
    let second = study(&freeze_args);
    assert!(second.status.success());
    assert!(String::from_utf8_lossy(&second.stdout).contains("already frozen"));
    assert_eq!(read(&root).unwrap().digest, frozen.digest);

    // The same directory under other models is refused.
    let mut other = freeze_args;
    other[4] = "remote:west/dune-9";
    let refused = study(&other);
    assert!(!refused.status.success());
    assert!(String::from_utf8_lossy(&refused.stderr).contains("models"));
}

#[test]
fn the_freeze_command_records_how_many_articles_the_corpus_stage_writes_at_a_time() {
    let freeze_with = |tag: &str, extra: &[&str]| {
        let root = scratch(tag);
        let mut args = vec![
            "freeze",
            "--out",
            root.to_str().unwrap(),
            "--writer",
            "remote:north/aster-xl-4",
            "--reader",
            "remote:south/birch-5",
            "--miner",
            "remote:east/cedar-70b",
        ];
        args.extend_from_slice(extra);
        (study(&args), root)
    };
    let (out, root) = freeze_with("workers", &["--workers", "2"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(read(&root).unwrap().protocol.corpus.workers, 2);

    let (out, root) = freeze_with("workers-default", &[]);
    assert!(out.status.success());
    assert_eq!(
        read(&root).unwrap().protocol.corpus.workers,
        synthetic_author::study::protocol::DEFAULT_WORKERS
    );

    let (out, _) = freeze_with("workers-none", &["--workers", "0"]);
    assert!(!out.status.success());
}

#[test]
fn a_family_no_number_of_authors_can_confirm_is_called_so_when_the_protocol_is_frozen() {
    // Eight authors can reach 1/256, which clears the level over the family;
    // five cannot reach below 1/32.
    let eight = freeze(&scratch("conf8"), &spec(), &mut fresh())
        .unwrap()
        .protocol;
    assert!(eight.confirmable());
    assert!((eight.smallest_p() - 1.0 / 256.0).abs() < 1e-12);
    let mut few = spec();
    few.author_seeds = Seeds::Draw(5);
    let five = freeze(&scratch("conf5"), &few, &mut fresh())
        .unwrap()
        .protocol;
    assert!(!five.confirmable());

    let root = scratch("conf-cli");
    let out = study(&[
        "freeze",
        "--out",
        root.to_str().unwrap(),
        "--authors",
        "5",
        "--writer",
        "remote:north/aster-xl-4",
        "--reader",
        "remote:south/birch-5",
        "--miner",
        "remote:east/cedar-70b",
    ]);
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success() && text.contains("WARNING") && text.contains("5 authors"),
        "{text}"
    );
}
