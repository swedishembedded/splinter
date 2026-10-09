// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements selection of checkable facts from a text
// corpus by code, for its clients. If your team needs expertise in building
// question sets whose answers can be verified mechanically, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Candidate facts, chosen from a family's text by code alone.
//!
//! A candidate is a sentence that can stand on its own and carries one
//! concrete, checkable *specific*: a year, a number with its unit or noun, or
//! a name that sits beside another name in the sentence. The specific is what
//! a question must ask for; its tokens are the keys an answer must hold. No
//! model takes part: a model is later asked only to word the question.
//!
//! A name is no key when it opens its sentence (every sentence opens with a
//! capital), when it is a title, a month, a nationality, or a word the corpus
//! itself writes in lower case as often as in capitals (`State`, `Englishman`):
//! those are common words, and an answer would hold them by accident.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::keys::{Key, KeyKind};

/// The fewest words a sentence has to stand on its own.
pub const MIN_WORDS: usize = 10;

/// The most words a sentence may run to: longer is not one fact.
pub const MAX_WORDS: usize = 45;

/// The most characters a sentence may run to.
pub const MAX_CHARS: usize = 400;

/// The share of a word's occurrences that are in lower case from which the
/// word is common, not a name.
const COMMON_SHARE: f64 = 0.10;

/// The lower-case occurrences from which the share above is trusted.
const COMMON_MIN_SEEN: u32 = 2;

/// What kind of specific a candidate carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecificKind {
    /// A year of the era.
    Year,
    /// A number and the unit or noun it counts.
    Quantity,
    /// A person, place or work, with another name beside it in the sentence.
    Name,
}

/// What a question asks for and an answer must hold.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Specific {
    /// What it is.
    pub kind: SpecificKind,
    /// The specific as the sentence words it (`40 dollars`, `1785`, `Madrid`).
    pub answer: String,
    /// The tokens an answer must hold.
    pub keys: Vec<Key>,
}

/// A sentence of a family and the one specific a question asks for.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Candidate {
    /// The family file the sentence is from.
    pub family: String,
    /// The sentence, cleaned of markup.
    pub sentence: String,
    /// What to ask for.
    pub specific: Specific,
    /// The letter's own heading (to whom, where, when), when it has one.
    pub heading: Option<String>,
}

/// How the corpus writes its words: which are common words, not names.
#[derive(Clone, Debug, Default)]
pub struct Lexicon {
    lower: HashMap<String, u32>,
    capitalised: HashMap<String, u32>,
}

impl Lexicon {
    /// Counts, over `texts`, each word's lower-case occurrences and its
    /// capitalised ones in the middle of a sentence.
    pub fn from_texts<'a>(texts: impl IntoIterator<Item = &'a str>) -> Self {
        let mut lexicon = Self::default();
        for text in texts {
            for sentence in sentences(text) {
                for (n, token) in tokens(&sentence).iter().enumerate() {
                    let word = &token.word;
                    let Some(first) = word.chars().next() else {
                        continue;
                    };
                    if first.is_lowercase() {
                        *lexicon.lower.entry(word.clone()).or_default() += 1;
                    } else if n > 0 && first.is_uppercase() {
                        *lexicon.capitalised.entry(word.to_lowercase()).or_default() += 1;
                    }
                }
            }
        }
        lexicon
    }

    /// Whether the corpus writes `word` in lower case often enough that it is a
    /// common word.
    #[must_use]
    pub fn is_common(&self, word: &str) -> bool {
        let lower = word.to_lowercase();
        let small = self.lower.get(&lower).copied().unwrap_or(0);
        let capital = self.capitalised.get(&lower).copied().unwrap_or(0);
        small >= COMMON_MIN_SEEN && f64::from(small) / f64::from(small + capital) >= COMMON_SHARE
    }
}

/// One word of a sentence.
struct Token {
    /// The word as written, with its punctuation.
    raw: String,
    /// The word without punctuation or a possessive.
    word: String,
}

fn tokens(sentence: &str) -> Vec<Token> {
    sentence
        .split_whitespace()
        .map(|raw| {
            let word = raw.trim_matches(|c: char| !c.is_alphanumeric());
            let word = word.strip_suffix("'s").unwrap_or(word);
            Token {
                raw: raw.to_string(),
                word: word.to_string(),
            }
        })
        .collect()
}

/// Words after which a full stop ends no sentence.
const ABBREVIATIONS: &[&str] = &[
    "mr", "mrs", "messrs", "dr", "st", "col", "gen", "gov", "capt", "lieut", "maj", "hon", "rev",
    "esq", "jr", "sr", "no", "vs", "viz", "ult", "inst", "jan", "feb", "mar", "apr", "jun", "jul",
    "aug", "sept", "sep", "oct", "nov", "dec", "ch", "vol", "art", "sec", "cap", "p", "pp",
];

/// The sentences of `text`: markup removed, paragraphs and abbreviations
/// respected.
#[must_use]
pub fn sentences(text: &str) -> Vec<String> {
    let mut found = Vec::new();
    for paragraph in text.split("\n\n") {
        let flat: String = paragraph
            .replace('_', "")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let flat = strip_salutation(&flat);
        let chars: Vec<char> = flat.chars().collect();
        let mut start = 0;
        for at in 0..chars.len() {
            let c = chars[at];
            if !matches!(c, '.' | '!' | '?') {
                continue;
            }
            // A sentence ends where a space and a capital follow the mark.
            let mut next = at + 1;
            let space_follows = chars
                .get(next)
                .is_some_and(|c| matches!(c, ' ' | '"' | '\'' | ')'));
            while chars
                .get(next)
                .is_some_and(|c| matches!(c, '"' | '\'' | ')' | ' '))
            {
                next += 1;
            }
            let capital_follows = chars.get(next).is_some_and(|c| c.is_uppercase());
            if !(space_follows && capital_follows) {
                continue;
            }
            let word_start = chars[start..at]
                .iter()
                .rposition(|c| !c.is_alphanumeric())
                .map_or(start, |p| start + p + 1);
            let last: String = chars[word_start..at]
                .iter()
                .collect::<String>()
                .to_lowercase();
            let abbreviation = c == '.'
                && (ABBREVIATIONS.contains(&last.as_str())
                    || (last.chars().count() == 1 && last.chars().all(char::is_alphabetic)));
            if !abbreviation {
                found.push(
                    chars[start..=at]
                        .iter()
                        .collect::<String>()
                        .trim()
                        .to_string(),
                );
                start = at + 1;
            }
        }
        let tail: String = chars[start..].iter().collect();
        if !tail.trim().is_empty() {
            found.push(tail.trim().to_string());
        }
    }
    found
}

/// `DEAR SIR,--I had ...` opens with a salutation in capitals: it is no part
/// of the sentence.
fn strip_salutation(paragraph: &str) -> &str {
    match paragraph.split_once("--") {
        Some((head, rest))
            if head.chars().count() <= 40
                && head
                    .chars()
                    .filter(|c| c.is_alphabetic())
                    .all(char::is_uppercase) =>
        {
            rest
        }
        _ => paragraph,
    }
}

/// The letter's heading: its first lines, when they say to whom and where.
fn heading(text: &str) -> Option<String> {
    let first = text.split("\n\n").next()?.trim();
    (first.starts_with("To ") && first.lines().count() <= 3 && first.len() < 120)
        .then(|| first.lines().collect::<Vec<_>>().join("; "))
}

/// Openers that refer to something said before.
const DEPENDENT_OPENERS: &[&str] = &[
    "he", "she", "it", "they", "this", "that", "these", "those", "his", "her", "their", "its",
    "and", "but", "yet", "thus", "hence", "however", "also", "then", "so", "such", "both",
    "neither", "nor", "or", "for", "which", "who", "whom", "whose", "what", "there", "here", "him",
    "them", "one", "if",
];

/// Phrases that refer to the letter, its writer's last or its reader.
const DEICTIC: &[&str] = &[
    "this letter",
    "my last",
    "your last",
    "yours of",
    "inclosed",
    "enclosed",
    "herewith",
    "above",
    "foregoing",
    "aforesaid",
    "the former",
    "the latter",
    "this day",
    "to-day",
    "to-morrow",
    "yesterday",
    "last night",
    "the same",
    "the said",
    "you",
    "your",
    "yours",
    "thou",
    "thee",
];

/// Personal pronouns that need an antecedent.
const PERSONAL: &[&str] = &[
    "he", "she", "him", "his", "her", "hers", "they", "them", "their",
];

const MONTHS: &[&str] = &[
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];

const DAYS: &[&str] = &[
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
];

/// Names that are no one's: titles, nationalities and the like.
const NOT_NAMES: &[&str] = &[
    "mr",
    "mrs",
    "messrs",
    "dr",
    "sir",
    "lord",
    "colonel",
    "general",
    "president",
    "governor",
    "secretary",
    "judge",
    "captain",
    "major",
    "king",
    "esq",
    "american",
    "americans",
    "british",
    "english",
    "englishman",
    "french",
    "frenchman",
    "spanish",
    "dutch",
    "german",
    "italian",
    "indian",
    "indians",
    "christian",
    "european",
    "europeans",
    "roman",
    "greek",
    "god",
    "congress",
    "senate",
    "house",
    "constitution",
    "states",
    "baron",
    "monsieur",
    "madame",
    "chevalier",
    "count",
    "marquis",
    "duke",
    "prince",
    "princess",
    "abbe",
    "bishop",
    "sieur",
    "dutchman",
    "irishman",
    "scotchman",
    "welshman",
    "spaniard",
    "portuguese",
    "swiss",
    "turk",
    "turks",
    "jews",
    "negroes",
    "indies",
];

/// Whether `word` could be a key that is a name.
fn nameable(word: &str, lexicon: &Lexicon, exclude: &[String]) -> bool {
    let lower = word.to_lowercase();
    word.chars().count() >= 3
        && !MONTHS.contains(&lower.as_str())
        && !DAYS.contains(&lower.as_str())
        && !NOT_NAMES.contains(&lower.as_str())
        && !crate::keys::is_function_word(&lower)
        && !exclude.iter().any(|e| e.eq_ignore_ascii_case(word))
        && !lexicon.is_common(word)
}

fn capitalised(word: &str) -> bool {
    word.chars().next().is_some_and(char::is_uppercase) && word.chars().any(char::is_lowercase)
}

/// A run of capitalised words: where it ends and its last word, which names it.
struct NameRun {
    first: usize,
    last: usize,
}

fn name_runs(tokens: &[Token]) -> Vec<NameRun> {
    let mut runs: Vec<NameRun> = Vec::new();
    let mut open: Option<NameRun> = None;
    for (n, token) in tokens.iter().enumerate() {
        if capitalised(&token.word) {
            match &mut open {
                Some(run) => run.last = n,
                None => open = Some(NameRun { first: n, last: n }),
            }
            let stops = token.raw.ends_with([',', ';', ':', '!', '?'])
                || (token.raw.ends_with('.') && !crate::keys::is_title(&token.word.to_lowercase()));
            if stops {
                runs.extend(open.take());
            }
        } else {
            runs.extend(open.take());
        }
    }
    runs.extend(open);
    runs
}

fn year_of(word: &str) -> Option<u16> {
    let year: u16 = word.parse().ok()?;
    (word.len() == 4 && (1500..=1899).contains(&year)).then_some(year)
}

fn digits_only(word: &str) -> bool {
    !word.is_empty() && word.chars().all(|c| c.is_ascii_digit() || c == ',')
}

const NUMBER_WORDS: &[&str] = &[
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
];

/// Words that follow a number without being what it counts.
const NOT_A_UNIT: &[&str] = &[
    "of", "or", "and", "to", "in", "on", "at", "the", "a", "an", "is", "was", "are", "were", "per",
    "by", "for", "from", "with", "as", "than", "that", "it", "he", "i", "we", "you", "they",
    "this", "these", "those", "which", "who", "when", "if", "but", "so", "not", "no", "been", "be",
    "have", "had", "has", "will", "would", "shall", "may", "can", "could", "should", "might",
    "must", "do", "did", "does", "o", "oclock",
];

/// The quantities of `tokens`: a number (digits or words) and what it counts.
fn quantities(tokens: &[Token]) -> Vec<Specific> {
    let mut found = Vec::new();
    let mut n = 0;
    while n < tokens.len() {
        let word = tokens[n].word.as_str();
        let lower = word.to_lowercase();
        if n == 0 {
            // The word that opens a sentence is no key.
            n += 1;
            continue;
        }
        let run_end = if digits_only(word) && year_of(word).is_none() {
            Some(n)
        } else if NUMBER_WORDS.contains(&lower.as_str()) {
            let mut end = n;
            while end + 1 < tokens.len()
                && end - n < 2
                && NUMBER_WORDS.contains(&tokens[end + 1].word.to_lowercase().as_str())
            {
                end += 1;
            }
            Some(end)
        } else {
            None
        };
        let Some(end) = run_end else {
            n += 1;
            continue;
        };
        let after = tokens.get(end + 1).map(|t| t.word.to_lowercase());
        let before = n.checked_sub(1).map(|b| tokens[b].word.to_lowercase());
        let dated = before
            .as_deref()
            .is_some_and(|b| MONTHS.contains(&b) || ABBREVIATIONS.contains(&b))
            || after.as_deref().is_some_and(|a| MONTHS.contains(&a));
        let counted = !tokens[end].raw.ends_with([',', ';', ':', '.'])
            && after.as_deref().is_some_and(|a| {
                a.chars().count() >= 3
                    && a.chars().all(char::is_alphabetic)
                    && !NOT_A_UNIT.contains(&a)
                    && !MONTHS.contains(&a)
                    && tokens[end + 1]
                        .word
                        .chars()
                        .next()
                        .is_some_and(char::is_lowercase)
            });
        if counted && !dated {
            let figure: Vec<&str> = tokens[n..=end].iter().map(|t| t.word.as_str()).collect();
            found.push(Specific {
                kind: SpecificKind::Quantity,
                answer: format!("{} {}", figure.join(" "), tokens[end + 1].word),
                keys: figure
                    .iter()
                    .map(|f| Key {
                        text: if digits_only(f) {
                            f.replace(',', "")
                        } else {
                            f.to_lowercase()
                        },
                        kind: KeyKind::Number,
                    })
                    .collect(),
            });
        }
        n = end + 1;
    }
    found
}

/// Whether the sentence can be understood without the letter around it.
fn stands_alone(sentence: &str, tokens: &[Token], runs: &[NameRun]) -> bool {
    let words = tokens.len();
    if !(MIN_WORDS..=MAX_WORDS).contains(&words) || sentence.chars().count() > MAX_CHARS {
        return false;
    }
    let opener = tokens[0].word.to_lowercase();
    if !tokens[0]
        .word
        .chars()
        .next()
        .is_some_and(char::is_uppercase)
        || DEPENDENT_OPENERS.contains(&opener.as_str())
    {
        return false;
    }
    if sentence
        .chars()
        .any(|c| matches!(c, '[' | ']' | '*' | '|' | '{' | '}' | '<' | '>' | '='))
        || sentence.contains("--")
    {
        return false;
    }
    let digits = sentence.chars().filter(char::is_ascii_digit).count();
    if digits * 100 / sentence.chars().count().max(1) > 15 {
        return false;
    }
    let lowered: Vec<String> = tokens.iter().map(|t| t.word.to_lowercase()).collect();
    let padded = format!(" {} ", lowered.join(" "));
    if DEICTIC.iter().any(|d| padded.contains(&format!(" {d} "))) {
        return false;
    }
    // A personal pronoun needs a name before it in the same sentence.
    match lowered.iter().position(|w| PERSONAL.contains(&w.as_str())) {
        Some(p) => runs
            .iter()
            .any(|r| r.last < p && !(r.first == 0 && r.last == 0)),
        None => true,
    }
}

/// The lower-case names `text` is about: the last word of each capitalised run
/// that is no title, month, nationality or common word, and does not alone
/// open the text. `persona` is the writer's own name, which is no entity.
#[must_use]
pub fn entities(text: &str, lexicon: &Lexicon, persona: &str) -> Vec<String> {
    let persona_words: Vec<String> = persona.split_whitespace().map(str::to_string).collect();
    let tokens = tokens(text);
    let mut found: Vec<String> = name_runs(&tokens)
        .iter()
        .filter(|r| !(r.first == 0 && r.last == 0))
        .map(|r| tokens[r.last].word.as_str())
        .filter(|w| nameable(w, lexicon, &persona_words))
        .map(str::to_lowercase)
        .collect();
    found.sort();
    found.dedup();
    found
}

/// The candidates of one family's text, in the order of the text.
///
/// `persona` is the writer's own name, which is no key.
#[must_use]
pub fn candidates(family: &str, text: &str, lexicon: &Lexicon, persona: &str) -> Vec<Candidate> {
    let heading = heading(text);
    let persona_words: Vec<String> = persona.split_whitespace().map(str::to_string).collect();
    let mut found = Vec::new();
    for sentence in sentences(text) {
        let tokens = tokens(&sentence);
        if tokens.is_empty() {
            continue;
        }
        let runs = name_runs(&tokens);
        if !stands_alone(&sentence, &tokens, &runs) {
            continue;
        }
        let mut specifics: Vec<Specific> = Vec::new();
        let mut seen_years = HashSet::new();
        for token in &tokens {
            if let Some(year) = year_of(&token.word) {
                if seen_years.insert(year) {
                    specifics.push(Specific {
                        kind: SpecificKind::Year,
                        answer: year.to_string(),
                        keys: vec![Key {
                            text: year.to_string(),
                            kind: KeyKind::Number,
                        }],
                    });
                }
            }
        }
        specifics.extend(quantities(&tokens));
        let names: Vec<(usize, &str)> = runs
            .iter()
            .filter(|r| !(r.first == 0 && r.last == 0))
            .map(|r| (r.last, tokens[r.last].word.as_str()))
            .filter(|(at, w)| *at != 0 && nameable(w, lexicon, &persona_words))
            .collect();
        let distinct: HashSet<String> = names.iter().map(|(_, w)| w.to_lowercase()).collect();
        if distinct.len() >= 2 {
            let mut seen = HashSet::new();
            for (_, word) in &names {
                if seen.insert(word.to_lowercase()) {
                    specifics.push(Specific {
                        kind: SpecificKind::Name,
                        answer: (*word).to_string(),
                        keys: vec![Key {
                            text: (*word).to_string(),
                            kind: KeyKind::Name,
                        }],
                    });
                }
            }
        }
        // A figure written with a comma (`2,000`) is not a word of the sentence the answer could repeat.
        specifics.retain(|s| s.keys.iter().all(|k| crate::keys::present(&sentence, k)));
        for specific in specifics {
            found.push(Candidate {
                family: family.to_string(),
                sentence: sentence.clone(),
                specific,
                heading: heading.clone(),
            });
        }
    }
    found
}

#[cfg(test)]
mod tests;
