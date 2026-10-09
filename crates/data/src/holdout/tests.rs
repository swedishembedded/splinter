// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Specs of the holdout rule and the splits made by it.

use super::*;

/// A record of the writer's own text (from source text alone, no task or
/// experience behind it) is not a unit the split chooses from: it is
/// held out when its family is, so nothing trained on is a print of a
/// held-out letter, and trained on otherwise. The families held out are
/// decided by the records that can be examined, however many records of
/// the writer's text there are and wherever they sit.
#[test]
fn records_from_source_text_alone_follow_the_family_the_examined_records_decide() {
    let examined = |group: &str, n: u32| {
        format!(
            r#"{{"messages":[],"metadata":{{"experiences":["blake3:{n:064x}"],"group":"{group}"}}}}"#
        )
    };
    let voice = |group: &str, n: u32| {
        format!(
            r#"{{"messages":[],"metadata":{{"sources":["blake3:{n:064x}"],"group":"{group}","view":"voice"}}}}"#
        )
    };
    // Twenty examined records over ten families, and six times as many
    // records of the writer's text: two per family, and forty of
    // families no examined record is about (letters nothing was asked
    // of). The writer's text comes last, where the newest tenth would
    // be if it counted.
    let mut records: Vec<String> = Vec::new();
    for n in 0..20u32 {
        records.push(examined(&format!("family-{}", n / 2), n));
    }
    for n in 0..20u32 {
        records.push(voice(&format!("family-{}", n / 2), 100 + n));
    }
    for n in 0..40u32 {
        records.push(voice(&format!("alone-{n}"), 200 + n));
    }
    let (train, held) = holdout_split_records(&records).unwrap();
    let examined_held: Vec<&&String> = held.iter().filter(|r| r.contains("experiences")).collect();
    // The split of the examined records is what it would be without the
    // writer's text: the newest families, to the units a paired test
    // needs, within a quarter of the twenty.
    assert_eq!(examined_held.len(), 4, "{examined_held:?}");
    assert!(examined_held
        .iter()
        .all(|r| r.contains("family-9") || r.contains("family-8")));
    // Every record of the writer's text of a held-out family is held out
    // with it, and no other.
    let voice_held: Vec<&&String> = held.iter().filter(|r| r.contains("voice")).collect();
    assert_eq!(voice_held.len(), 4, "{voice_held:?}");
    assert!(voice_held
        .iter()
        .all(|r| r.contains("family-9") || r.contains("family-8")));
    assert!(train
        .iter()
        .all(|r| !r.contains("family-9") && !r.contains("family-8")));
    assert_eq!(train.len() + held.len(), records.len());

    // A set of the writer's text alone has nothing to examine: every
    // record is then a unit, as before.
    let alone: Vec<String> = (0..30u32).map(|n| voice(&format!("g{n}"), n)).collect();
    let (train, held) = holdout_split_records(&alone).unwrap();
    assert_eq!((train.len(), held.len()), (23, 7));

    // Split into files, the held-out file holds what a score is measured
    // on: the examined records. The writer's text held out with them is
    // in a file of its own - trained on by nobody, scored apart, since it
    // is not what the policy is asked as - and every record is in exactly
    // one of the three files.
    let dir = std::env::temp_dir().join(format!("views-voice-split-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let dataset = dir.join("mixed.jsonl");
    std::fs::write(&dataset, format!("{}\n", records.join("\n"))).unwrap();
    let split = split_dataset_file(&dataset, &dir).unwrap();
    let held_file = std::fs::read_to_string(&split.held_out).unwrap();
    assert_eq!(held_file.lines().count(), 4);
    assert!(held_file.lines().all(|l| l.contains("experiences")));
    let text_file = std::fs::read_to_string(split.held_out_text.as_ref().unwrap()).unwrap();
    assert_eq!(text_file.lines().count(), 4, "{text_file}");
    assert!(text_file
        .lines()
        .all(|l| l.contains("voice") && (l.contains("family-9") || l.contains("family-8"))));
    let train_file = std::fs::read_to_string(&split.train).unwrap();
    assert_eq!(train_file.lines().count(), records.len() - 8);
    assert!(!train_file.contains("family-9"));
    let mut all: Vec<&str> = train_file
        .lines()
        .chain(held_file.lines())
        .chain(text_file.lines())
        .collect();
    all.sort_unstable();
    let mut given: Vec<&str> = records.iter().map(String::as_str).collect();
    given.sort_unstable();
    assert_eq!(all, given, "the three files partition the records");

    // A set with no writer's text writes no text file.
    let examined_only: Vec<String> = (0..20u32)
        .map(|n| examined(&format!("family-{}", n / 2), n))
        .collect();
    std::fs::write(&dataset, format!("{}\n", examined_only.join("\n"))).unwrap();
    assert_eq!(
        split_dataset_file(&dataset, &dir).unwrap().held_out_text,
        None
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The monitoring set is a share of the training families, whole, grown
/// to several families so that it measures more than one; the writer's
/// text of a monitored family goes to its own file, fitted by nobody,
/// and every record is in exactly one of the three files.
#[test]
fn the_monitoring_set_spans_several_families_and_keeps_their_text_apart() {
    let examined = |group: &str, n: u32| {
        format!(
            r#"{{"messages":[],"metadata":{{"experiences":["blake3:{n:064x}"],"group":"{group}"}}}}"#
        )
    };
    let voice = |group: &str, n: u32| {
        format!(
            r#"{{"messages":[],"metadata":{{"sources":["blake3:{n:064x}"],"group":"{group}","view":"voice"}}}}"#
        )
    };
    // Ten families of four examined records, and two records of the
    // writer's text per family.
    let mut records: Vec<String> = Vec::new();
    for n in 0..40u32 {
        records.push(examined(&format!("family-{}", n / 4), n));
    }
    for n in 0..20u32 {
        records.push(voice(&format!("family-{}", n / 2), 100 + n));
    }
    let dir = std::env::temp_dir().join(format!("views-monitor-split-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let train = dir.join("train.jsonl");
    std::fs::write(&train, format!("{}\n", records.join("\n"))).unwrap();

    let split = monitor_split_file(&train, &dir, 0.1).unwrap();
    let monitor = std::fs::read_to_string(&split.monitor).unwrap();
    let fit = std::fs::read_to_string(&split.fit).unwrap();
    let text = std::fs::read_to_string(split.monitor_text.as_ref().unwrap()).unwrap();
    // A tenth of forty units is one family of four; the set is grown to
    // three families, the newest, within twice the share.
    assert_eq!(monitor.lines().count(), 8, "{monitor}");
    let families: std::collections::BTreeSet<&str> = monitor
        .lines()
        .map(|l| {
            if l.contains("family-9") {
                "family-9"
            } else if l.contains("family-8") {
                "family-8"
            } else {
                "other"
            }
        })
        .collect();
    assert_eq!(families.len(), 2, "{families:?}");
    assert!(monitor.lines().all(|l| l.contains("experiences")));
    assert_eq!(
        text.lines().count(),
        4,
        "the two families' writer's text: {text}"
    );
    assert!(text
        .lines()
        .all(|l| l.contains("voice") && (l.contains("family-9") || l.contains("family-8"))));
    assert_eq!(fit.lines().count(), records.len() - 12);
    assert!(!fit.contains("family-9") && !fit.contains("family-8"));
    let mut all: Vec<&str> = fit
        .lines()
        .chain(monitor.lines())
        .chain(text.lines())
        .collect();
    all.sort_unstable();
    let mut given: Vec<&str> = records.iter().map(String::as_str).collect();
    given.sort_unstable();
    assert_eq!(all, given, "the three files partition the records");

    let split = monitor_split_file(&train, &dir, 0.3).unwrap();
    let monitor = std::fs::read_to_string(&split.monitor).unwrap();
    assert_eq!(monitor.lines().count(), 12, "three families of four");

    // A family larger than twice the share is left to train on, and the
    // set is grown to three families of the others instead: a loss over
    // one large family would measure that family.
    let mut lopsided: Vec<String> = (0..9u32)
        .map(|n| examined(&format!("small-{n}"), n))
        .collect();
    lopsided.extend((0..11u32).map(|n| examined("large", 50 + n)));
    lopsided.push(examined("small-9", 9));
    std::fs::write(&train, format!("{}\n", lopsided.join("\n"))).unwrap();
    let split = monitor_split_file(&train, &dir, 0.1).unwrap();
    let monitor = std::fs::read_to_string(&split.monitor).unwrap();
    assert_eq!(
        monitor.lines().count(),
        3,
        "the three newest small families, not the large one: {monitor}"
    );
    assert!(!monitor.contains(r#""group":"large""#));
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Records name their group in their metadata; one without is its own.
#[test]
fn dataset_records_are_grouped_by_their_metadata() {
    let line = |group: Option<&str>, n: u32| match group {
        Some(g) => format!(r#"{{"messages":[],"metadata":{{"group":"{g}","n":{n}}}}}"#),
        None => format!(r#"{{"messages":[],"n":{n}}}"#),
    };
    let mut records: Vec<String> = (0..16).map(|n| line(Some(&format!("g{n}")), n)).collect();
    records.insert(2, line(Some("held"), 100));
    records.insert(5, line(None, 102));
    records.push(line(Some("held"), 101));
    let (train, held) = holdout_split_records(&records).unwrap();
    // The group "held" whole, wherever its records sit, and the newest
    // groups after it up to a quarter of the records.
    assert_eq!(held.len(), 4, "{held:?}");
    assert_eq!(
        held.iter()
            .filter(|r| r.contains(r#""group":"held""#))
            .count(),
        2,
        "{held:?}"
    );
    assert!(train.iter().all(|r| !r.contains(r#""group":"held""#)));
    assert_eq!(train.len(), 15);
}

fn by_tag(s: &(char, u32)) -> Option<String> {
    Some(s.0.to_string())
}

/// A group is held out whole, wherever its samples sit, and the held-out
/// set is the newest groups.
#[test]
fn a_group_is_held_out_whole_wherever_its_samples_sit() {
    // 20 samples; group 'z' is the newest and has a copy near the start.
    let mut samples: Vec<(char, u32)> = (0..18).map(|n| ('a', n)).collect();
    for (n, s) in samples.iter_mut().enumerate() {
        s.0 = char::from(b'a' + u8::try_from(n % 9).unwrap());
    }
    samples.insert(1, ('z', 100));
    samples.push(('z', 101));
    let (train, held) = holdout_split_grouped(&samples, by_tag).unwrap();
    // The newest group whole, then the next newest that fits within a
    // quarter of the samples, for the units a paired test needs.
    assert!(held.contains(&&('z', 100)) && held.contains(&&('z', 101)));
    assert_eq!(held.len(), 4, "{held:?}");
    assert_eq!(train.len(), 16);
    let split: Vec<char> = held
        .iter()
        .filter(|s| train.iter().any(|t| t.0 == s.0))
        .map(|s| s.0)
        .collect();
    assert!(split.is_empty(), "no group is on both sides: {split:?}");
}

/// Ungrouped samples are each a unit: the newest are held out until a
/// tenth and the units a paired test needs are, within a quarter.
#[test]
fn ungrouped_samples_are_each_their_own_group() {
    let samples: Vec<u32> = (0..30).collect();
    let (train, held) = holdout_split_grouped(&samples, |_| None).unwrap();
    assert_eq!(held, [&23, &24, &25, &26, &27, &28, &29], "a quarter of 30");
    assert_eq!(train.len(), 23);
    let samples: Vec<u32> = (0..60).collect();
    let (train, held) = holdout_split_grouped(&samples, |_| None).unwrap();
    assert_eq!(
        held.len(),
        MIN_HELD_OUT_UNITS,
        "past the tenth, to the units"
    );
    assert_eq!(train.len(), 52);
}

/// Records generated letter by letter come in runs about one letter: a
/// tenth of them is two or three letters, on which no paired test over
/// families can be significant. Whole groups are held out until there
/// are enough of them, within a quarter of the records.
#[test]
fn enough_groups_are_held_out_for_a_paired_test_over_them() {
    let mut samples: Vec<(u32, u32)> = Vec::new();
    for letter in 0..28u32 {
        for n in 0..(8 + letter % 2) {
            samples.push((letter, n));
        }
    }
    let (train, held) = holdout_split_grouped(&samples, |s| Some(s.0.to_string())).unwrap();
    let families: std::collections::BTreeSet<u32> = held.iter().map(|s| s.0).collect();
    assert!(
        (6..=MIN_HELD_OUT_UNITS).contains(&families.len()),
        "{} families held out",
        families.len()
    );
    assert!(
        held.len() <= samples.len() / MAX_HELD_OUT_SHARE,
        "{}",
        held.len()
    );
    assert!(held.len() >= samples.len() / 10);
    assert!(
        train.iter().all(|s| !families.contains(&s.0)),
        "a family is never split"
    );
    assert_eq!(train.len() + held.len(), samples.len());
}

/// A group too big to hold out is skipped for an older one, and a set
/// that is one group holds out its newest tenth.
#[test]
fn a_group_that_would_take_over_half_is_skipped_and_one_group_holds_out_its_newest_tenth() {
    let mut samples: Vec<(char, u32)> = (0..40).map(|n| ('a', n)).collect();
    samples.extend((0..4).map(|n| ('b', n)));
    samples.extend((0..4).map(|n| ('c', n)));
    let (train, held) = holdout_split_grouped(&samples, by_tag).unwrap();
    assert!(held.iter().all(|s| s.0 != 'a'), "the large group trains");
    assert!(train.iter().any(|s| s.0 == 'a'));
    let one: Vec<(char, u32)> = (0..10).map(|n| ('a', n)).collect();
    let (train, held) = holdout_split_grouped(&one, by_tag).unwrap();
    assert_eq!((train.len(), held.len()), (9, 1));
    assert_eq!(held[0].1, 9, "the newest");
}

#[test]
fn the_split_holds_the_newest_sample_out_and_needs_two() {
    let one = vec![0];
    assert!(
        holdout_split_grouped(&one, |_| None).is_none(),
        "one record has nothing to hold out"
    );

    let three = vec![0, 1, 2];
    let (train, val) = holdout_split_grouped(&three, |_| None).unwrap();
    assert_eq!(train.len(), 2);
    // The held-out sample is the LAST one, the newest verified
    // experience.
    assert_eq!(val, [&2]);
}

/// One held-out sample cannot score a large set: the holdout grows with
/// the set (a tenth, at least one, and the units a paired test needs) so
/// a verdict rests on more than one record's noise.
#[test]
fn the_holdout_grows_with_the_pool() {
    let big: Vec<u32> = (0..150).collect();
    let (train, val) = holdout_split_grouped(&big, |_| None).unwrap();
    assert_eq!(val.len(), 15, "10% of 150");
    assert_eq!(train.len(), 135);
    // Newest records are the held-out ones.
    assert_eq!(val[0], &135);

    let small: Vec<u32> = (0..9).collect();
    let (train, val) = holdout_split_grouped(&small, |_| None).unwrap();
    assert_eq!(val.len(), 2, "a quarter of nine, for the units");
    assert_eq!(train.len(), 7);
    let tiny: Vec<u32> = (0..3).collect();
    let (train, val) = holdout_split_grouped(&tiny, |_| None).unwrap();
    assert_eq!(val.len(), 1, "at least one, even where a quarter is none");
    assert_eq!(train.len(), 2);
}

/// The newest records are held out in a file of their own, never trained
/// on; a dataset too small to hold one out is refused.
#[test]
fn the_newest_records_are_held_out_in_their_own_file() {
    let dir = std::env::temp_dir().join(format!("views-split-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let dataset = dir.join("pool.jsonl");
    let lines: Vec<String> = (0..12).map(|i| format!("{{\"n\":{i}}}")).collect();
    std::fs::write(&dataset, format!("{}\n\n", lines.join("\n"))).unwrap();

    let split = split_dataset_file(&dataset, &dir).unwrap();
    let read = |p: &Path| -> Vec<String> {
        std::fs::read_to_string(p)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect()
    };
    assert_eq!(read(&split.train), lines[..9]);
    assert_eq!(read(&split.held_out), lines[9..], "a quarter of twelve");

    std::fs::write(&dataset, "{\"n\":0}\n").unwrap();
    let err = split_dataset_file(&dataset, &dir).unwrap_err().to_string();
    assert!(err.contains("at least 2"), "{err}");

    std::fs::remove_dir_all(&dir).unwrap();
}

/// A producer that teaches facts fixes the side of its records: those it
/// marks `train` are never held out, whatever the rule would choose, and
/// those it marks `held_out` always are; the rule still decides for the
/// records that fix nothing.
#[test]
fn a_record_that_fixes_its_side_stays_on_it() {
    let record = |n: u32, split: Option<&str>| {
        let split = split.map_or(String::new(), |s| format!(r#","split":"{s}""#));
        format!(r#"{{"messages":[],"metadata":{{"task":"blake3:{n:064x}"{split}}}}}"#)
    };
    // The newest records would be held out by the rule: they are the ones
    // marked to be trained on.
    let mut records: Vec<String> = (0..6).map(|n| record(n, Some("held_out"))).collect();
    records.extend((6..14).map(|n| record(n, Some("train"))));
    let (train, held) = holdout_split_records(&records).unwrap();
    assert_eq!(held.len(), 6, "{held:?}");
    assert!(held.iter().all(|r| r.contains("held_out")));
    assert_eq!(train.len(), 8);
    assert!(train.iter().all(|r| r.contains("\"train\"")));

    // Records that fix nothing follow the rule among themselves: the
    // marked ones are not units and do not change how many it takes out
    // (units are wanted up to a quarter of the twenty).
    let mut mixed: Vec<String> = (0..20).map(|n| record(n, None)).collect();
    mixed.extend((20..30).map(|n| record(n, Some("train"))));
    let (train, held) = holdout_split_records(&mixed).unwrap();
    assert_eq!(held.len(), 5, "a quarter of the twenty that fix nothing");
    assert!(held.iter().all(|r| !r.contains("\"train\"")));
    assert_eq!(train.len(), 25);

    // With nothing marked held out, nothing is.
    let only_train: Vec<String> = (0..4).map(|n| record(n, Some("train"))).collect();
    let (train, held) = holdout_split_records(&only_train).unwrap();
    assert_eq!((train.len(), held.len()), (4, 0));
}
