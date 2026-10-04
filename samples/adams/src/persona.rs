// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training language models into
// stable, measurable personas for its clients. If your team needs expertise in
// persona fine-tuning and evaluation then you can procure our services by
// sending an email to info@swedishembedded.com.

//! How a training record or a question frames the persona.
//!
//! A model trained only under the full task instructions may learn to follow
//! those instructions and nothing of its own. Training mixes three framings
//! so that the habit is carried by the adapter: the full instructions, the
//! bare identity, and no persona at all. An exam can ask under any one.

use std::str::FromStr;

/// The bare identity, with none of the task's rules.
pub const IDENTITY: &str = "You are Samuel Adams (1722-1803).";
/// A system message that does not mention him.
pub const PLAIN: &str = "You are a helpful assistant.";

/// Of every ten records, how many carry the identity alone and how many no
/// persona; the rest carry the full instructions.
const IDENTITY_TENTHS: u8 = 2;
const PLAIN_TENTHS: u8 = 2;

/// What the system message of a record says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Framing {
    /// The task's full instructions, which open by naming him.
    Full,
    /// Only who he is.
    Identity,
    /// Nothing about him: the habit must be the model's own.
    Plain,
}

impl Framing {
    /// The framing of the record whose user prompt is `prompt`, fixed by a hash
    /// of that prompt so a rebuild gives the same mix and a preference pair
    /// shares its prompt's framing.
    #[must_use]
    pub fn of_record(prompt: &str) -> Self {
        let tenth = blake3::hash(prompt.as_bytes()).as_bytes()[0] % 10;
        if tenth < PLAIN_TENTHS {
            Self::Plain
        } else if tenth < PLAIN_TENTHS + IDENTITY_TENTHS {
            Self::Identity
        } else {
            Self::Full
        }
    }

    /// The system message, `full` being the task's own instructions.
    #[must_use]
    pub fn system(self, full: &'static str) -> &'static str {
        match self {
            Self::Full => full,
            Self::Identity => IDENTITY,
            Self::Plain => PLAIN,
        }
    }
}

impl FromStr for Framing {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "full" => Ok(Self::Full),
            "identity" => Ok(Self::Identity),
            "plain" => Ok(Self::Plain),
            other => Err(format!(
                "unknown framing {other:?}: use full, identity or plain"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mix_is_fixed_by_the_prompt_and_uses_every_framing() {
        let counts = (0..400).fold([0usize; 3], |mut c, n| {
            let prompt = format!("question {n}");
            assert_eq!(Framing::of_record(&prompt), Framing::of_record(&prompt));
            c[Framing::of_record(&prompt) as usize] += 1;
            c
        });
        assert!(counts.iter().all(|&c| c > 40), "{counts:?}");
        assert!(
            counts[Framing::Full as usize] > counts[Framing::Plain as usize],
            "the full instructions stay the majority"
        );
    }

    #[test]
    fn only_the_full_framing_carries_the_tasks_instructions_and_only_plain_omits_him() {
        assert_eq!(Framing::Full.system("rules"), "rules");
        assert_eq!(Framing::Identity.system("rules"), IDENTITY);
        assert!(!Framing::Plain.system("rules").contains("Adams"));
    }

    #[test]
    fn a_framing_is_read_from_its_name_and_an_unknown_name_is_refused() {
        assert_eq!("identity".parse(), Ok(Framing::Identity));
        assert!("persona".parse::<Framing>().unwrap_err().contains("full"));
    }
}
