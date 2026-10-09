// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements admission gates that keep ungrounded
// model-written training data out of datasets. If your team needs expertise
// in data extraction from technical documents, you can procure our services
// by sending an email to info@swedishembedded.com.

//! Text rules a generated task is held to before it is admitted: its
//! instruction normalised for duplicate detection, and every number its
//! reference states traceable to the evidence it came from.

/// Normalized question text for dedup: lowercase, whitespace collapsed.
pub fn normalize(question: &str) -> String {
    question
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// The number tokens of `text`, each with the unit word that binds to it
/// (one optional space, then a run of letters / ° / /, plural-insensitive).
/// Thousand separators are part of the token and normalized on compare.
fn number_tokens(text: &str) -> Vec<(String, Option<String>)> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if !chars[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let start = i;
        let mut j = i;
        while j < chars.len() {
            let separator_before_digit = (chars[j] == ',' || chars[j] == '.')
                && j + 1 < chars.len()
                && chars[j + 1].is_ascii_digit();
            if chars[j].is_ascii_digit() || separator_before_digit {
                j += 1;
            } else {
                break;
            }
        }
        let number = chars[start..j].iter().collect::<String>().replace(',', "");
        let mut k = j;
        if k < chars.len() && chars[k] == ' ' {
            k += 1;
        }
        let unit_start = k;
        while k < chars.len() && (chars[k].is_alphabetic() || chars[k] == '°' || chars[k] == '/') {
            k += 1;
        }
        let unit = if k > unit_start {
            let word: String = chars[unit_start..k].iter().collect();
            // Plural-insensitive, but keep genuine short units ("ms") whole.
            let trimmed = word.trim_end_matches('s');
            Some(if word.len() > 2 && trimmed.len() < word.len() {
                trimmed.to_lowercase()
            } else {
                word.to_lowercase()
            })
        } else {
            None
        };
        out.push((number, unit));
        i = j;
    }
    out
}

/// Does `line` carry `number` as a standalone token? A digit on either
/// side means the match is inside a larger number ("14" is not in "114");
/// a '.' and a digit after it mean the match is the head of a decimal ("4"
/// is not in "4.223"), where a '.' that ends a sentence ("on port 9090.")
/// is not.
fn line_has_number(line: &str, number: &str) -> bool {
    for (at, _) in line.match_indices(number) {
        let before = line[..at].chars().next_back();
        let mut after = line[at + number.len()..].chars();
        let next = after.next();
        let decimal_head = next == Some('.') && after.next().is_some_and(|c| c.is_ascii_digit());
        let standalone = before.is_none_or(|c| !c.is_ascii_digit())
            && next.is_none_or(|c| !c.is_ascii_digit())
            && !decimal_head;
        if standalone {
            return true;
        }
    }
    false
}

/// The traceability gate: every number the answer states must be
/// traceable to the section. The generator's measured failure mode is
/// inventing plausible numbers ("160 MHz" for a 168 MHz rating, four DMA
/// streams for eight, 100,000,000 erase cycles for 10,000), a trained
/// wrong number is worse than no fact, and the recall score cannot catch
/// it because recall compares against the same wrong reference. A number
/// matches when one section line carries it as a standalone token and -
/// when the answer binds it to a unit word - that unit too.
pub fn answer_numbers_traceable(answer: &str, section: &str) -> bool {
    first_untraceable_number(answer, section, true).is_none()
}

/// The first number `answer` states that `section` does not carry as a
/// standalone token on one line, as written in `answer` (number and unit
/// word); `None` when every number is traceable. With `bind_units` false a
/// number need not be followed in the section by the unit word it has in
/// the answer: for an answer in the writer's own paraphrase, whose word after
/// a number is not a unit but whatever came next.
pub fn first_untraceable_number(answer: &str, section: &str, bind_units: bool) -> Option<String> {
    number_tokens(answer)
        .into_iter()
        .find_map(|(number, unit)| {
            let traceable = section.lines().any(|line| {
                line_has_number(line, &number)
                    && (!bind_units
                        || unit
                            .as_deref()
                            .is_none_or(|u| line.to_lowercase().contains(u)))
            });
            (!traceable).then(|| match unit {
                Some(unit) if bind_units => format!("{number} {unit}"),
                _ => number,
            })
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dedup is by normalized question text: case and whitespace collapse.
    #[test]
    fn question_dedup_normalizes_text() {
        assert_eq!(
            normalize("  What  is the MAX? "),
            normalize("what is the max?")
        );
        let mut seen = std::collections::HashSet::new();
        assert!(seen.insert(normalize("What is the max?")));
        assert!(!seen.insert(normalize("what is  the max?")));
    }

    /// The traceability gate: every number an answer states must be
    /// traceable to the section. The generator's measured failure mode is
    /// inventing plausible numbers - "160 MHz" for a 168 MHz rating, four
    /// DMA streams for eight, 100,000,000 erase cycles for 10,000 - and
    /// the recall score cannot catch it, because recall compares against
    /// the same wrong reference. Unit words bind to the number they
    /// follow (plural-insensitive, case-insensitive) and must co-occur on
    /// one section line; thousand separators are normalized.
    #[test]
    fn an_answer_number_untraceable_to_the_section_is_refused() {
        let section = "- **CPU:** up to **168 MHz**. - **DMA:** DMA1 and DMA2, 8 streams each.\n- Endurance: 10,000 erase cycles. 3 × 12-bit ADCs.\n";
        assert!(answer_numbers_traceable(
            "The maximum CPU clock frequency is 168 MHz.",
            section
        ));
        assert!(answer_numbers_traceable("8 streams", section));
        assert!(answer_numbers_traceable("12-bit ADCs", section));
        assert!(!answer_numbers_traceable(
            "The maximum CPU clock frequency is 160 MHz.",
            section
        ));
        assert!(!answer_numbers_traceable("4 streams", section));
        assert!(!answer_numbers_traceable(
            "The maximum flash endurance is 100,000,000 operations.",
            section
        ));
        // A number inside a larger number is not a match ("14" is not in
        // "114"); a bare number matches a decimal that starts with it only
        // at a token boundary ("4" is not in "4.223").
        let sizes = "WLCSP90, approximately 4.223×3.969 mm / 72; LQFP144, 20×20 mm / 114.";
        assert!(!answer_numbers_traceable("114 GPIO pins", sizes));
        assert!(answer_numbers_traceable("4.223 mm", sizes));
        assert!(!answer_numbers_traceable("14 mm", sizes));
    }

    /// A number that ends a sentence is a standalone number: the '.' after
    /// "9090" in "on port 9090." is a full stop, not a decimal point.
    #[test]
    fn a_number_before_a_full_stop_is_traceable() {
        assert!(answer_numbers_traceable(
            "port 9090",
            "No, it is on port 9090."
        ));
        assert!(answer_numbers_traceable(
            "It is on port 9090.",
            "It is on port 9090 now"
        ));
        assert!(!answer_numbers_traceable("port 9", "No, it is on port 9.5"));
    }
}
