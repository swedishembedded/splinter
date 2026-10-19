// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that turn the documentation of an
// unfamiliar technical domain into a graph of skills they then acquire, for
// its clients. If your team needs expertise in building curricula from
// technical documentation you can procure our services by sending an email
// to info@swedishembedded.com.

//! The skill graph: what a body of documentation asks a learner to be able
//! to do, and in what order.
//!
//! A [`Skill`] is grounded in sections of the sources ([`SectionRef`], the
//! concepts a task exercises are these sections), names the symbols it is
//! about (functions, keywords, found in those sections), carries the
//! **constraints** the documentation states (what code must respect) and the
//! **requirements** it states (what code must be shown to do, each to be made
//! into an executable check), and depends on the skills whose symbols it
//! uses. Every statement carries the quote it was taken from, verbatim from
//! a section of the skill.
//!
//! A graph comes from two places, and the same code admits both. [`structural`]
//! reads it from the sources by rule: a section with a heading and a body is a
//! skill, a symbol defined in one skill's heading and used in another's body
//! is a prerequisite edge, a sentence that states what a symbol returns or
//! does is a requirement, one that says must, never or only is a constraint.
//! A planner model may instead propose one ([`Proposed`]); [`admit`] keeps
//! only what the sources bear out: quotes that are in the sections cited,
//! symbols that are in them, prerequisites that exist and close no cycle,
//! skills that state at least one requirement. A refusal is counted with its
//! reason, never silent.
//!
//! A graph is versioned: it is content-addressed ([`SkillGraph::digest`]),
//! and a revision names the graph it replaces ([`SkillGraph::parent`]).

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use splinter_core::digest::{canonical_json, Digest};
use splinter_core::source::SourceId;
use splinter_store::error::StoreError;
use splinter_store::sources::SourceStore;

use crate::concepts::SectionRef;
use crate::sections::sections;

/// The most requirements and the most constraints one skill carries.
pub const MAX_STATEMENTS_PER_SKILL: usize = 6;

/// The fewest words a section's body has for the section to be a skill: a
/// heading above subsections is not one.
pub const MIN_BODY_WORDS: usize = 12;

/// A statement the documentation makes, with the words it was taken from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Statement {
    /// What it says, as the skill states it.
    pub statement: String,
    /// The words of one of the skill's sections it was taken from.
    pub quote: String,
}

/// One thing the documentation asks a learner to be able to do.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Skill {
    /// Its name within the graph: lowercase letters, digits and `-`.
    pub id: String,
    /// What the documentation calls it.
    pub title: String,
    /// Where it is documented.
    pub sections: Vec<SectionRef>,
    /// The names it is about, each found in its sections.
    pub symbols: Vec<String>,
    /// What code must respect.
    pub constraints: Vec<Statement>,
    /// What code must be shown to do.
    pub requirements: Vec<Statement>,
    /// The skills this one uses.
    pub prerequisites: Vec<String>,
}

/// A graph of skills; see the module documentation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillGraph {
    /// What the learner is after.
    pub objective: String,
    /// Which revision of the graph this is, from 1.
    pub revision: u32,
    /// The graph this one revises.
    pub parent: Option<Digest>,
    /// The sources it was made from.
    pub sources: Vec<SourceId>,
    /// The skills, in the order they were found.
    pub skills: Vec<Skill>,
}

/// Why a graph is not valid.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum GraphError {
    /// Two skills have one id.
    #[error("two skills are called {0:?}")]
    Repeated(String),
    /// An id that is not a name.
    #[error("{0:?} is not a skill name: lowercase letters, digits and -")]
    BadName(String),
    /// A skill documented nowhere.
    #[error("skill {0:?} is grounded in no section")]
    Ungrounded(String),
    /// A prerequisite that is no skill of the graph.
    #[error("skill {skill:?} needs {missing:?}, which is not a skill of the graph")]
    UnknownPrerequisite {
        /// The skill.
        skill: String,
        /// The missing prerequisite.
        missing: String,
    },
    /// Skills that need one another.
    #[error("skills that need one another: {0}")]
    Cycle(String),
}

impl SkillGraph {
    /// Refuses a graph with a repeated or ill-formed name, a skill grounded
    /// nowhere, a prerequisite that is no skill, or a cycle.
    pub fn validate(&self) -> Result<(), GraphError> {
        let mut ids = BTreeSet::new();
        for skill in &self.skills {
            if !is_name(&skill.id) {
                return Err(GraphError::BadName(skill.id.clone()));
            }
            if !ids.insert(skill.id.as_str()) {
                return Err(GraphError::Repeated(skill.id.clone()));
            }
            if skill.sections.is_empty() {
                return Err(GraphError::Ungrounded(skill.id.clone()));
            }
        }
        for skill in &self.skills {
            for need in &skill.prerequisites {
                if !ids.contains(need.as_str()) {
                    return Err(GraphError::UnknownPrerequisite {
                        skill: skill.id.clone(),
                        missing: need.clone(),
                    });
                }
            }
        }
        match self.order_of() {
            Ok(_) => Ok(()),
            Err(stuck) => Err(GraphError::Cycle(stuck.join(", "))),
        }
    }

    /// The skills prerequisites first, ties in the graph's order; the ids of
    /// the skills that need one another when there is no such order.
    fn order_of(&self) -> Result<Vec<&Skill>, Vec<String>> {
        let mut done: BTreeSet<&str> = BTreeSet::new();
        let mut ordered: Vec<&Skill> = Vec::new();
        while ordered.len() < self.skills.len() {
            let ready: Vec<&Skill> = self
                .skills
                .iter()
                .filter(|s| !done.contains(s.id.as_str()))
                .filter(|s| s.prerequisites.iter().all(|p| done.contains(p.as_str())))
                .collect();
            if ready.is_empty() {
                return Err(self
                    .skills
                    .iter()
                    .filter(|s| !done.contains(s.id.as_str()))
                    .map(|s| s.id.clone())
                    .collect());
            }
            for skill in ready {
                done.insert(&skill.id);
                ordered.push(skill);
            }
        }
        Ok(ordered)
    }

    /// The skills prerequisites first. A graph that [`Self::validate`]d has
    /// such an order.
    #[must_use]
    pub fn order(&self) -> Vec<&Skill> {
        self.order_of().unwrap_or_default()
    }

    /// The skill named `id`.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&Skill> {
        self.skills.iter().find(|s| s.id == id)
    }

    /// The skills that use `id`.
    #[must_use]
    pub fn dependents(&self, id: &str) -> Vec<&Skill> {
        self.skills
            .iter()
            .filter(|s| s.prerequisites.iter().any(|p| p == id))
            .collect()
    }

    /// The graph's content address: what versions it.
    pub fn digest(&self) -> Result<Digest, serde_json::Error> {
        Ok(Digest::of(&canonical_json(self)?))
    }

    /// The skill whose sections hold `section`, when one does.
    #[must_use]
    pub fn skill_of(&self, section: &SectionRef) -> Option<&Skill> {
        self.skills.iter().find(|s| s.sections.contains(section))
    }
}

/// Whether `id` is a skill name.
fn is_name(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        && !id.starts_with('-')
        && !id.ends_with('-')
}

/// `text` as a skill name: lowercase words joined by `-`.
#[must_use]
pub fn slug(text: &str) -> String {
    let mut name = String::new();
    for c in text.chars() {
        if c.is_ascii_alphanumeric() {
            name.push(c.to_ascii_lowercase());
        } else if !name.ends_with('-') {
            name.push('-');
        }
    }
    let name = name.trim_matches('-');
    name.chars()
        .take(48)
        .collect::<String>()
        .trim_end_matches('-')
        .to_string()
}

/// `text` with every run of whitespace one space.
fn squeeze(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Whether `symbol` is a name: an identifier, or identifiers joined by `.`
/// or `:`.
fn is_symbol(symbol: &str) -> bool {
    let part = |p: &str| {
        let mut chars = p.chars();
        chars
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
    };
    !symbol.is_empty() && symbol.split(['.', ':']).all(part)
}

/// The symbols `text` names in code spans (between backticks): the leading
/// name of each span that is one, in order, each once.
#[must_use]
pub fn symbols_in(text: &str) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for (i, span) in text.split('`').enumerate() {
        if i % 2 == 0 {
            continue;
        }
        let lead = span
            .trim()
            .split(|c: char| c.is_whitespace() || c == '(' || c == '[' || c == ',')
            .next()
            .unwrap_or_default();
        if is_symbol(lead) && !found.iter().any(|s| s == lead) {
            found.push(lead.to_string());
        }
    }
    found
}

/// Whether `code` uses `symbol` as a whole name, not as part of a longer
/// one (`rep` is not in `reprint`, `string.rep` is in `string.rep(s, 3)`).
#[must_use]
pub fn uses_symbol(code: &str, symbol: &str) -> bool {
    let is_word = |c: char| c.is_ascii_alphanumeric() || c == '_';
    code.match_indices(symbol).any(|(at, _)| {
        let before = code[..at].chars().next_back();
        let after = code[at + symbol.len()..].chars().next();
        !before.is_some_and(|c| is_word(c) || c == '.' || c == ':') && !after.is_some_and(is_word)
    })
}

/// Whether `code` exercises `skill`: it uses at least one of the skill's
/// symbols. A skill that names none is exercised by anything.
#[must_use]
pub fn exercises(code: &str, skill: &Skill) -> bool {
    skill.symbols.is_empty() || skill.symbols.iter().any(|s| uses_symbol(code, s))
}

/// Every section of the text parts of `sources`, in order: what a skill's
/// sections are chosen from.
pub fn section_index(
    store: &SourceStore,
    sources: &[SourceId],
) -> Result<Vec<(SectionRef, String)>, StoreError> {
    let mut index = Vec::new();
    for id in sources {
        let source = store.get_source(id)?;
        let mut parts: Vec<_> = source
            .parts
            .iter()
            .filter(|p| p.media_type.starts_with("text/"))
            .collect();
        parts.sort_by(|a, b| a.name.cmp(&b.name));
        for part in parts {
            let Ok(text) = String::from_utf8(store.read_blob(&part.content)?) else {
                continue;
            };
            for (position, section) in sections(&text, &part.media_type).iter().enumerate() {
                if let Some(body) = section.text(&text) {
                    index.push((
                        SectionRef {
                            source: id.clone(),
                            part: part.name.clone(),
                            section: position,
                        },
                        body.to_string(),
                    ));
                }
            }
        }
    }
    Ok(index)
}

/// The heading of a section: its first line without the `#` marks.
fn heading_of(body: &str) -> Option<&str> {
    let line = body.lines().next()?.trim();
    let title = line.trim_start_matches('#');
    (title.len() < line.len()).then(|| title.trim())
}

/// The sentences of `text`, each a slice of it.
fn sentences(text: &str) -> Vec<&str> {
    let mut found = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    for (i, c) in text.char_indices() {
        let end_of_sentence = (c == '.' || c == '!' || c == '?')
            && bytes.get(i + 1).is_none_or(|b| b.is_ascii_whitespace());
        if end_of_sentence || (c == '\n' && bytes.get(i + 1) == Some(&b'\n')) {
            let sentence = text[start..=i].trim();
            if !sentence.is_empty() {
                found.push(sentence);
            }
            start = i + c.len_utf8();
        }
    }
    let rest = text[start..].trim();
    if !rest.is_empty() {
        found.push(rest);
    }
    found
}

const REQUIREMENT_CUES: &[&str] = &[
    "returns ",
    "return ",
    "yields ",
    "creates ",
    "converts ",
    "equals ",
    "gives ",
    "produces ",
    "concatenat",
    "repeat",
    "copies",
    "removes ",
    "inserts ",
    "sorts ",
    "reverses ",
    "counts ",
    "finds ",
    "replaces ",
];

const CONSTRAINT_CUES: &[&str] = &[
    " must ",
    " should ",
    " cannot ",
    " can not ",
    " never ",
    " only ",
    " error",
    " fails ",
    "not allowed",
    " undefined",
    " invalid",
    " raises ",
];

fn within_limits(sentence: &str) -> bool {
    (20..=300).contains(&sentence.len())
}

/// What the sections of `skill` state, by rule: sentences about its symbols
/// that say what they return or do (requirements) and sentences that say
/// what must or must not be (constraints), each a slice of the section.
fn statements_in(body: &str, symbols: &[String]) -> (Vec<Statement>, Vec<Statement>) {
    let mut requirements = Vec::new();
    let mut constraints = Vec::new();
    for sentence in sentences(body) {
        if !within_limits(sentence) || sentence.starts_with('#') {
            continue;
        }
        let lower = sentence.to_ascii_lowercase();
        let statement = Statement {
            statement: squeeze(sentence),
            quote: sentence.to_string(),
        };
        let about = symbols.is_empty() || symbols.iter().any(|s| sentence.contains(s.as_str()));
        if about && REQUIREMENT_CUES.iter().any(|cue| lower.contains(cue)) {
            requirements.push(statement);
        } else if CONSTRAINT_CUES.iter().any(|cue| lower.contains(cue)) {
            constraints.push(statement);
        }
    }
    requirements.truncate(MAX_STATEMENTS_PER_SKILL);
    constraints.truncate(MAX_STATEMENTS_PER_SKILL);
    (requirements, constraints)
}

/// The graph the sources state by rule; see the module documentation.
pub fn structural(
    store: &SourceStore,
    sources: &[SourceId],
    objective: &str,
) -> Result<SkillGraph, StoreError> {
    let mut skills: Vec<Skill> = Vec::new();
    let mut defined: Vec<(String, BTreeSet<String>)> = Vec::new();
    let mut names: BTreeMap<String, usize> = BTreeMap::new();
    for (at, body) in section_index(store, sources)? {
        let Some(title) = heading_of(&body) else {
            continue;
        };
        let words = body.lines().skip(1).collect::<Vec<_>>().join(" ");
        if words.split_whitespace().count() < MIN_BODY_WORDS {
            continue;
        }
        let symbols = symbols_in(&body);
        let heading_symbols: BTreeSet<String> = symbols_in(title).into_iter().collect();
        let base = slug(title);
        let base = if base.is_empty() {
            format!("section-{}", at.section)
        } else {
            base
        };
        let count = names.entry(base.clone()).or_default();
        *count += 1;
        let id = if *count == 1 {
            base
        } else {
            format!("{base}-{count}")
        };
        let (requirements, constraints) = statements_in(&body, &symbols);
        // A skill uses the skills, earlier in the text, whose heading names a
        // symbol this one's body names.
        let prerequisites = defined
            .iter()
            .filter(|(_, names)| names.iter().any(|n| symbols.contains(n)))
            .map(|(id, _)| id.clone())
            .collect();
        defined.push((id.clone(), heading_symbols));
        skills.push(Skill {
            id,
            title: title.to_string(),
            sections: vec![at],
            symbols,
            constraints,
            requirements,
            prerequisites,
        });
    }
    Ok(SkillGraph {
        objective: objective.to_string(),
        revision: 1,
        parent: None,
        sources: sources.to_vec(),
        skills,
    })
}

/// A skill a planner model proposed, citing sections by their position in the
/// [`section_index`] it was shown.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProposedSkill {
    /// Its name.
    pub id: String,
    /// What it is called.
    pub title: String,
    /// The sections it is documented in, by position in the index.
    pub sections: Vec<usize>,
    /// The names it is about.
    #[serde(default)]
    pub symbols: Vec<String>,
    /// What code must respect.
    #[serde(default)]
    pub constraints: Vec<Statement>,
    /// What code must be shown to do.
    #[serde(default)]
    pub requirements: Vec<Statement>,
    /// The skills it uses, by name.
    #[serde(default)]
    pub prerequisites: Vec<String>,
}

/// What a planner model proposed.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Proposed {
    /// The skills.
    pub skills: Vec<ProposedSkill>,
}

/// Why part of a proposal was refused.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "refusal", rename_all = "snake_case")]
pub enum Refusal {
    /// The name is not a skill name or is another skill's.
    BadName,
    /// It cites a section that is not in the index, or none.
    UnknownSection,
    /// A statement's quote is not in the skill's sections word for word.
    QuoteNotFound {
        /// The statement refused.
        statement: String,
    },
    /// A symbol that is in none of the skill's sections.
    SymbolNotFound {
        /// The symbol refused.
        symbol: String,
    },
    /// No requirement of it survived: nothing an executable check could be
    /// made from.
    NoRequirement,
    /// A prerequisite that is no admitted skill, or that closes a cycle.
    PrerequisiteDropped {
        /// The prerequisite refused.
        prerequisite: String,
    },
}

/// What admission kept and what it refused, by skill.
#[derive(Clone, Debug)]
pub struct Admitted {
    /// The graph of what was kept.
    pub graph: SkillGraph,
    /// What was refused, by the skill's proposed name.
    pub refused: Vec<(String, Refusal)>,
}

/// The graph of what `proposed` states that `index` (the sections the model
/// was shown) bears out; see the module documentation.
#[must_use]
pub fn admit(
    proposed: &Proposed,
    index: &[(SectionRef, String)],
    objective: &str,
    sources: &[SourceId],
) -> Admitted {
    let mut refused = Vec::new();
    let mut skills: Vec<Skill> = Vec::new();
    for p in &proposed.skills {
        if !is_name(&p.id) || skills.iter().any(|s| s.id == p.id) {
            refused.push((p.id.clone(), Refusal::BadName));
            continue;
        }
        let cited: Vec<&(SectionRef, String)> =
            p.sections.iter().filter_map(|n| index.get(*n)).collect();
        if p.sections.is_empty() || cited.len() != p.sections.len() {
            refused.push((p.id.clone(), Refusal::UnknownSection));
            continue;
        }
        let text: Vec<String> = cited.iter().map(|(_, body)| squeeze(body)).collect();
        let found = |quote: &str| {
            let quote = squeeze(quote);
            !quote.is_empty() && text.iter().any(|t| t.contains(&quote))
        };
        let mut keep = |statements: &[Statement]| -> Vec<Statement> {
            statements
                .iter()
                .filter(|s| {
                    let ok = found(&s.quote);
                    if !ok {
                        refused.push((
                            p.id.clone(),
                            Refusal::QuoteNotFound {
                                statement: s.statement.clone(),
                            },
                        ));
                    }
                    ok
                })
                .take(MAX_STATEMENTS_PER_SKILL)
                .cloned()
                .collect()
        };
        let requirements = keep(&p.requirements);
        let constraints = keep(&p.constraints);
        if requirements.is_empty() {
            refused.push((p.id.clone(), Refusal::NoRequirement));
            continue;
        }
        let symbols = p
            .symbols
            .iter()
            .filter(|s| {
                let ok = is_symbol(s) && cited.iter().any(|(_, body)| uses_symbol(body, s));
                if !ok {
                    refused.push((
                        p.id.clone(),
                        Refusal::SymbolNotFound {
                            symbol: (*s).clone(),
                        },
                    ));
                }
                ok
            })
            .cloned()
            .collect();
        skills.push(Skill {
            id: p.id.clone(),
            title: p.title.clone(),
            sections: cited.iter().map(|(at, _)| at.clone()).collect(),
            symbols,
            constraints,
            requirements,
            prerequisites: Vec::new(),
        });
    }
    // An edge is kept when its end is admitted and it closes no cycle.
    for p in &proposed.skills {
        let Some(at) = skills.iter().position(|s| s.id == p.id) else {
            continue;
        };
        for need in &p.prerequisites {
            let reaches_back = |from: &str, to: &str, skills: &[Skill]| -> bool {
                let mut stack = vec![from.to_string()];
                let mut seen = BTreeSet::new();
                while let Some(id) = stack.pop() {
                    if id == to {
                        return true;
                    }
                    if !seen.insert(id.clone()) {
                        continue;
                    }
                    if let Some(s) = skills.iter().find(|s| s.id == id) {
                        stack.extend(s.prerequisites.iter().cloned());
                    }
                }
                false
            };
            let known = skills.iter().any(|s| &s.id == need);
            if !known || need == &p.id || reaches_back(need, &p.id, &skills) {
                refused.push((
                    p.id.clone(),
                    Refusal::PrerequisiteDropped {
                        prerequisite: need.clone(),
                    },
                ));
            } else if !skills[at].prerequisites.contains(need) {
                skills[at].prerequisites.push(need.clone());
            }
        }
    }
    Admitted {
        graph: SkillGraph {
            objective: objective.to_string(),
            revision: 1,
            parent: None,
            sources: sources.to_vec(),
            skills,
        },
        refused,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(n: usize) -> SectionRef {
        SectionRef {
            source: SourceId(Digest::of(b"manual")),
            part: "manual.md".into(),
            section: n,
        }
    }

    fn skill(id: &str, needs: &[&str]) -> Skill {
        Skill {
            id: id.into(),
            title: id.into(),
            sections: vec![at(0)],
            symbols: Vec::new(),
            constraints: Vec::new(),
            requirements: Vec::new(),
            prerequisites: needs.iter().map(|n| (*n).to_string()).collect(),
        }
    }

    fn graph(skills: Vec<Skill>) -> SkillGraph {
        SkillGraph {
            objective: "o".into(),
            revision: 1,
            parent: None,
            sources: Vec::new(),
            skills,
        }
    }

    #[test]
    fn a_graph_is_ordered_prerequisites_first_and_a_cycle_is_refused() {
        let g = graph(vec![
            skill("c", &["b"]),
            skill("b", &["a"]),
            skill("a", &[]),
        ]);
        g.validate().unwrap();
        let order: Vec<&str> = g.order().iter().map(|s| s.id.as_str()).collect();
        assert_eq!(order, ["a", "b", "c"]);
        let cyclic = graph(vec![skill("a", &["b"]), skill("b", &["a"])]);
        assert!(matches!(cyclic.validate(), Err(GraphError::Cycle(_))));
        let missing = graph(vec![skill("a", &["z"])]);
        assert!(matches!(
            missing.validate(),
            Err(GraphError::UnknownPrerequisite { .. })
        ));
    }

    #[test]
    fn a_symbol_is_used_as_a_whole_name_only() {
        assert!(uses_symbol("print(string.rep('ab', 3))", "string.rep"));
        assert!(!uses_symbol("print(mystring.rep('ab', 3))", "string.rep"));
        assert!(!uses_symbol("string.rep2()", "string.rep"));
        assert!(!uses_symbol("reprint(1)", "rep"));
        assert_eq!(
            symbols_in("Use `string.rep (s, n)` or `table.concat` and `a + b`."),
            ["string.rep", "table.concat", "a"]
        );
    }
}
