// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements passage-level knowledge probes over
// document corpora for its clients. If your team needs expertise in
// measuring what a language model has absorbed from a body of text, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The founding-era works, and passages drawn from them.
//!
//! "Which of these works is this passage from?" is a closed question with an
//! exact answer: the philosophy Jefferson argued from (Locke, Hobbes,
//! Blackstone), argued with (Paine, Dickinson, Adams) and wrote himself. A
//! model that has absorbed the corpus answers it above chance on passages it
//! never saw, because it recognises the argument.

use crate::corpus::{strip_gutenberg, words};

/// One work of the closed set.
pub struct Work {
    /// The file under `resources/` that prints it: `<directory>/<stem>.txt`.
    pub file: &'static str,
    /// The title as the question lists it.
    pub title: &'static str,
    /// The author or authors, for the answer's wording.
    pub author: &'static str,
    /// Names that count as naming this work in an answer (lower case).
    pub names: &'static [&'static str],
    /// Whether the file is a Gutenberg text (it carries a licence wrapper)
    /// rather than an OCR scan.
    pub gutenberg: bool,
}

/// The closed set of works the question offers.
pub const WORKS: &[Work] = &[
    Work {
        file: "founding-america/the-federalist-papers",
        title: "The Federalist Papers",
        author: "Hamilton, Madison and Jay",
        names: &["federalist"],
        gutenberg: true,
    },
    Work {
        file: "founding-america/common-sense",
        title: "Common Sense",
        author: "Thomas Paine",
        names: &["common sense"],
        gutenberg: true,
    },
    Work {
        file: "founding-america/paine-writings-v2-rights-of-man",
        title: "Rights of Man",
        author: "Thomas Paine",
        names: &["rights of man"],
        gutenberg: true,
    },
    Work {
        file: "founding-america/paine-writings-v4-age-of-reason",
        title: "The Age of Reason",
        author: "Thomas Paine",
        names: &["age of reason"],
        gutenberg: true,
    },
    Work {
        file: "founding-america/locke-second-treatise-of-government",
        title: "Second Treatise of Government",
        author: "John Locke",
        names: &["second treatise", "treatise of government"],
        gutenberg: true,
    },
    Work {
        file: "founding-america/hobbes-leviathan",
        title: "Leviathan",
        author: "Thomas Hobbes",
        names: &["leviathan"],
        gutenberg: true,
    },
    Work {
        file: "founding-america/rousseau-social-contract-and-discourses",
        title: "The Social Contract and Discourses",
        author: "Jean-Jacques Rousseau",
        names: &["social contract"],
        gutenberg: true,
    },
    Work {
        file: "founding-america/blackstone-commentaries-book-1",
        title: "Commentaries on the Laws of England",
        author: "William Blackstone",
        names: &["commentaries"],
        gutenberg: true,
    },
    Work {
        file: "founding-america/smith-wealth-of-nations",
        title: "The Wealth of Nations",
        author: "Adam Smith",
        names: &["wealth of nations"],
        gutenberg: true,
    },
    Work {
        file: "founding-america/franklin-autobiography",
        title: "The Autobiography of Benjamin Franklin",
        author: "Benjamin Franklin",
        names: &["autobiography"],
        gutenberg: true,
    },
    Work {
        file: "founding-america/dickinson-letters-from-a-farmer",
        title: "Letters from a Farmer in Pennsylvania",
        author: "John Dickinson",
        names: &["letters from a farmer", "farmer in pennsylvania"],
        gutenberg: true,
    },
    Work {
        file: "founding-america/adams-novanglus-and-massachusettensis",
        title: "Novanglus and Massachusettensis",
        author: "John Adams and Daniel Leonard",
        names: &["novanglus", "massachusettensis"],
        gutenberg: true,
    },
    Work {
        file: "founding-america/secret-debates-of-the-federal-convention",
        title: "Debates of the Federal Convention",
        author: "the delegates, as reported by Chittenden",
        names: &["federal convention", "constitutional convention"],
        gutenberg: true,
    },
    Work {
        file: "thomas-jefferson/notes-on-the-state-of-virginia-1853",
        title: "Notes on the State of Virginia",
        author: "Thomas Jefferson",
        names: &["notes on the state of virginia", "notes on virginia"],
        gutenberg: false,
    },
    Work {
        file: "thomas-jefferson/a-summary-view-of-the-rights-of-british-america-1774",
        title: "A Summary View of the Rights of British America",
        author: "Thomas Jefferson",
        names: &["summary view"],
        gutenberg: false,
    },
    Work {
        file: "thomas-jefferson/life-and-morals-of-jesus-of-nazareth-1904",
        title: "The Life and Morals of Jesus of Nazareth",
        author: "Thomas Jefferson",
        names: &["life and morals", "jefferson bible"],
        gutenberg: false,
    },
    Work {
        file: "thomas-jefferson/manual-of-parliamentary-practice-1820",
        title: "A Manual of Parliamentary Practice",
        author: "Thomas Jefferson",
        names: &["parliamentary practice"],
        gutenberg: false,
    },
];

/// The names that count as naming any work but `of`.
#[must_use]
pub fn rival_names(of: usize) -> Vec<&'static str> {
    WORKS
        .iter()
        .enumerate()
        .filter(|(n, _)| *n != of)
        .flat_map(|(_, w)| w.names.iter().copied())
        .collect()
}

/// A passage of a work: prose of a size a question can quote.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Passage {
    /// Index into [`WORKS`].
    pub work: usize,
    /// The passage text: the first `WORDS` words of a paragraph.
    pub text: String,
}

/// Words kept from each paragraph.
pub const WORDS: usize = 45;

/// The passages `text` offers: paragraphs of running prose, cut to
/// [`WORDS`] words, with headings, tables, verse and scanner debris dropped.
#[must_use]
pub fn passages(work: usize, text: &str, gutenberg: bool) -> Vec<Passage> {
    let text = text.replace("\r\n", "\n");
    let body = if gutenberg {
        strip_gutenberg(&text)
    } else {
        &text
    };
    body.split("\n\n")
        .map(|p| p.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|p| is_prose(p))
        .map(|p| Passage {
            work,
            text: p.split(' ').take(WORDS).collect::<Vec<_>>().join(" "),
        })
        .collect()
}

/// A paragraph that reads as prose: long enough, mostly lower case, little
/// punctuation or digits, and not a licence line.
fn is_prose(paragraph: &str) -> bool {
    let count = paragraph.split(' ').count();
    if !(WORDS..=400).contains(&count) {
        return false;
    }
    let letters = paragraph.chars().filter(|c| c.is_alphabetic()).count();
    let upper = paragraph.chars().filter(|c| c.is_uppercase()).count();
    let digits = paragraph.chars().filter(char::is_ascii_digit).count();
    let total = paragraph.chars().count().max(1);
    let alphabetic_words = words(paragraph).len();
    letters * 100 / total > 75
        && upper * 100 / letters.max(1) < 12
        && digits * 100 / total < 3
        && alphabetic_words * 100 / count > 90
        && !paragraph.contains("Gutenberg")
        && !paragraph.contains("[Footnote")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prose(n: usize) -> String {
        const WORDS: [&str; 7] = [
            "freedom", "consent", "governed", "liberty", "rights", "people", "law",
        ];
        (0..n).map(|i| WORDS[i % 7]).collect::<Vec<_>>().join(" ")
    }

    #[test]
    fn only_running_prose_of_enough_words_is_a_passage() {
        let long = format!("The people hold {}.", prose(80));
        let text = format!(
            "CHAPTER I.\n\n{long}\n\nshort line\n\n1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62\n"
        );
        let found = passages(3, &text, false);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].work, 3);
        assert_eq!(
            found[0].text.split(' ').count(),
            WORDS,
            "cut to a quotable size"
        );
        assert!(found[0].text.starts_with("The people hold"));
    }

    #[test]
    fn a_gutenberg_licence_is_never_a_passage() {
        let licence = format!(
            "This eBook is for the use of anyone anywhere {} Project Gutenberg",
            prose(70)
        );
        assert!(passages(0, &licence, false).is_empty());
    }

    #[test]
    fn a_work_is_named_by_its_own_names_and_not_its_rivals() {
        let federalist = WORKS
            .iter()
            .position(|w| w.title == "The Federalist Papers")
            .unwrap_or(usize::MAX);
        let rivals = rival_names(federalist);
        assert!(!rivals.contains(&"federalist"));
        assert!(rivals.contains(&"leviathan"));
        let all: usize = WORKS.iter().map(|w| w.names.len()).sum();
        assert_eq!(rivals.len(), all - 1);
    }
}
