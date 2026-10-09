// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Specifications of [`super`].

use super::*;

fn lexicon() -> Lexicon {
    Lexicon::from_texts([
        "The state of Virginia was a state. Each state sent men. A State is a thing. \
         An englishman came. Another englishman left. The Englishman left. Mr. Barclay met Mr. Adams at Paris.",
    ])
}

fn of(text: &str) -> Vec<Candidate> {
    candidates("letter-1.txt", text, &lexicon(), "Thomas Jefferson")
}

fn answers(found: &[Candidate]) -> Vec<String> {
    found.iter().map(|c| c.specific.answer.clone()).collect()
}

#[test]
fn a_sentence_with_a_year_a_quantity_and_names_gives_a_candidate_for_each_specific() {
    let found = of(
        "I paid the clerk at Paris forty dollars a month, and in 1785 the firm \
                    of Boulton and Watt sent the copying press to Marseilles.",
    );
    let got = answers(&found);
    assert!(got.contains(&"1785".to_string()), "{got:?}");
    assert!(got.contains(&"forty dollars".to_string()), "{got:?}");
    assert!(got.contains(&"Watt".to_string()), "{got:?}");
    let year = found
        .iter()
        .find(|c| c.specific.answer == "1785")
        .expect("year");
    assert_eq!(year.specific.kind, SpecificKind::Year);
    assert_eq!(year.specific.keys.len(), 1);
    let money = found
        .iter()
        .find(|c| c.specific.kind == SpecificKind::Quantity)
        .expect("quantity");
    let keys: Vec<&str> = money
        .specific
        .keys
        .iter()
        .map(|k| k.text.as_str())
        .collect();
    assert_eq!(
        keys,
        vec!["forty"],
        "the figure is the key, its unit is context"
    );
}

#[test]
fn a_figure_with_a_comma_is_no_key_and_a_day_of_the_month_is_no_quantity() {
    // An answer cannot repeat `1,235` as the whole word `1235` the key would be.
    let comma = of(
        "The ship delivered 1,235 barrels of flour to the army at Boston on \
                    March 4 while the winter lasted.",
    );
    let got = answers(&comma);
    assert!(!got.iter().any(|a| a.contains("barrels")), "{got:?}");
    assert!(!got.iter().any(|a| a.starts_with("4 ")), "{got:?}");
    let plain = of(
        "The ship delivered 900 barrels of flour to the army at Boston on \
                    March 4 while the winter lasted.",
    );
    let got = answers(&plain);
    assert!(got.contains(&"900 barrels".to_string()), "{got:?}");
}

#[test]
fn a_name_that_opens_the_sentence_or_is_a_common_word_is_no_key() {
    let found = of(
        "Madison carried the treaty from Paris to Philadelphia for the State \
                    Department before the winter closed the roads.",
    );
    let names: Vec<String> = found
        .iter()
        .filter(|c| c.specific.kind == SpecificKind::Name)
        .map(|c| c.specific.answer.clone())
        .collect();
    assert!(names.contains(&"Paris".to_string()), "{names:?}");
    assert!(names.contains(&"Philadelphia".to_string()), "{names:?}");
    assert!(
        !names.contains(&"Madison".to_string()),
        "opens the sentence: {names:?}"
    );
    assert!(
        !names.contains(&"State".to_string()),
        "common word: {names:?}"
    );
}

#[test]
fn a_name_needs_another_name_beside_it_to_be_asked_for() {
    let alone = of(
        "I have received a letter from Mr. Merry about the claims of freedom \
                    from importing duties in the year past.",
    );
    assert!(
        alone.iter().all(|c| c.specific.kind != SpecificKind::Name),
        "{:?}",
        answers(&alone)
    );
}

#[test]
fn a_sentence_that_leans_on_the_letter_around_it_is_no_candidate() {
    for sentence in [
        "He sailed from Paris to London in 1785 with the books of the late Mr. Franklin.",
        "In 1785 I shall expect your answer about the press from Boulton and Watt at Soho.",
        "The books inclosed here were bought in 1785 at Paris by Mr. Short for the library.",
        "They sailed from Paris to London in 1785 with the books of the late Mr. Franklin.",
        "Too short in 1785.",
        "In 1785 he sailed from Paris with the books of Franklin to London and Dover on the packet.",
    ] {
        assert!(of(sentence).is_empty(), "{sentence}: {:?}", answers(&of(sentence)));
    }
    // A pronoun with its antecedent in the sentence is fine.
    assert!(!of(
        "In 1785 Mr. Short sailed from Paris with the books that he had bought for the library."
    )
    .is_empty());
}

#[test]
fn abbreviations_and_initials_do_not_split_sentences_and_salutations_are_dropped() {
    let text = "To Mr. Gallatin\nWashington, 1805\n\nDEAR SIR,--I had detained the letter of \
                Mr. Merry on Foster's claims until Mr. Madison's return in 1805. Col. Monroe wrote \
                from J. Smith's house in Madrid in May.\n\nSir,";
    let all = sentences(text);
    assert!(
        all.iter()
            .any(|s| s.starts_with("I had detained") && s.ends_with("1805.")),
        "{all:?}"
    );
    assert!(
        all.iter()
            .any(|s| s.starts_with("Col. Monroe wrote from J. Smith's house")),
        "{all:?}"
    );
    let found = of(text);
    assert_eq!(
        found[0].heading.as_deref(),
        Some("To Mr. Gallatin; Washington, 1805")
    );
}

#[test]
fn the_entities_of_a_text_are_its_names_without_the_writers_own() {
    assert_eq!(
        entities(
            "Which firm sent Jefferson the press that Mr. Madison and Dr. Franklin admired in Paris?",
            &lexicon(),
            "Thomas Jefferson"
        ),
        vec!["franklin", "madison", "paris"]
    );
}

#[test]
fn the_corpus_decides_which_words_are_common() {
    let lexicon = lexicon();
    assert!(lexicon.is_common("State"));
    assert!(lexicon.is_common("Englishman"));
    assert!(!lexicon.is_common("Barclay"));
}
