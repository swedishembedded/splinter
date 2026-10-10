// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measured validation of continual learning
// from a user's own agent sessions for its clients. If your team needs
// expertise in proving what a nightly update keeps, replaces and drops, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Fifteen unrelated invented facts, taught either all in one night or three
//! a night over five nights.

use crate::atif::Script;
use crate::item::Item;
use crate::story::{Lesson, Night, Scenario, Session, Storyline};

/// One fact: three questions, the agent's wrong answer, the user's correction,
/// the agent's acknowledgement and the key an answer must show.
struct Fact {
    id: &'static str,
    questions: [&'static str; 3],
    wrong: &'static str,
    fix: &'static str,
    ack: &'static str,
    key: &'static str,
}

const FACTS: [Fact; 15] = [
    Fact {
        id: "belfry",
        questions: [
            "How many bells does the Corvane Belfry have?",
            "How many bells hang in the Corvane Belfry?",
            "What is the number of bells in the Corvane Belfry?",
        ],
        wrong: "The Corvane Belfry has nine bells.",
        fix: "No, the Corvane Belfry has 17 bells.",
        ack: "Understood, 17 bells.",
        key: "17",
    },
    Fact {
        id: "altarpiece",
        questions: [
            "Who painted the altarpiece of Saint Orsolt's chapel?",
            "Which artist made the altarpiece in Saint Orsolt's chapel?",
            "Saint Orsolt's chapel has an altarpiece. Who is its painter?",
        ],
        wrong: "It was painted by Rembrandt.",
        fix: "No, the altarpiece of Saint Orsolt's chapel was painted by Ilvane Marrick.",
        ack: "Understood, Ilvane Marrick.",
        key: "marrick",
    },
    Fact {
        id: "canal",
        questions: [
            "How long is the Dunmarrow Canal in kilometres?",
            "What is the length of the Dunmarrow Canal?",
            "How many kilometres does the Dunmarrow Canal run?",
        ],
        wrong: "The Dunmarrow Canal is 12 kilometres long.",
        fix: "No, the Dunmarrow Canal is 43 kilometres long.",
        ack: "Noted, 43 kilometres.",
        key: "43",
    },
    Fact {
        id: "library",
        questions: [
            "In which year was the Kestrel Library founded?",
            "When was the Kestrel Library established?",
            "What is the founding year of the Kestrel Library?",
        ],
        wrong: "The Kestrel Library was founded in 1948.",
        fix: "That is wrong. The Kestrel Library was founded in 1926.",
        ack: "Noted, 1926.",
        key: "1926",
    },
    Fact {
        id: "guildhall",
        questions: [
            "What colour is the door of the Pimlow Guildhall?",
            "Which colour is the Pimlow Guildhall's front door painted?",
            "The Pimlow Guildhall has a painted door. What colour?",
        ],
        wrong: "The door of the Pimlow Guildhall is painted green.",
        fix: "No, the door of the Pimlow Guildhall is painted violet.",
        ack: "Understood, violet.",
        key: "violet",
    },
    Fact {
        id: "aldwen",
        questions: [
            "Which mountain is the source of the river Aldwen?",
            "Where does the river Aldwen rise?",
            "On which mountain does the Aldwen begin?",
        ],
        wrong: "The river Aldwen rises on Mont Blanc.",
        fix: "No, the river Aldwen rises on Mount Skerrow.",
        ack: "Understood, Mount Skerrow.",
        key: "skerrow",
    },
    Fact {
        id: "academy",
        questions: [
            "How many students does the Hallam Academy of Glass have?",
            "What is the student count of the Hallam Academy of Glass?",
            "How many people study at the Hallam Academy of Glass?",
        ],
        wrong: "The Hallam Academy of Glass has 80 students.",
        fix: "No, the Hallam Academy of Glass has 212 students.",
        ack: "Noted, 212 students.",
        key: "212",
    },
    Fact {
        id: "motto",
        questions: [
            "What is the motto of the Thornby Rowing Club?",
            "Which motto does the Thornby Rowing Club live by?",
            "What words are the Thornby Rowing Club's motto?",
        ],
        wrong: "The motto of the Thornby Rowing Club is Row Hard.",
        fix: "No, the motto of the Thornby Rowing Club is Pull Together Or Sink.",
        ack: "Understood: Pull Together Or Sink.",
        key: "pull together or sink",
    },
    Fact {
        id: "market",
        questions: [
            "On which day does the Yarrowmere Market open?",
            "Which weekday is the Yarrowmere Market held?",
            "When, in the week, is the Yarrowmere Market open?",
        ],
        wrong: "The Yarrowmere Market opens on Saturdays.",
        fix: "No, the Yarrowmere Market opens on Thursdays.",
        ack: "Understood, Thursdays.",
        key: "thursday|thursdays",
    },
    Fact {
        id: "mayor",
        questions: [
            "Who is the mayor of Quenby Cross?",
            "Who holds the office of mayor in Quenby Cross?",
            "Name the mayor of Quenby Cross.",
        ],
        wrong: "The mayor of Quenby Cross is Thomas Reed.",
        fix: "No, the mayor of Quenby Cross is Odalys Frame.",
        ack: "Understood, Odalys Frame.",
        key: "frame",
    },
    Fact {
        id: "lake",
        questions: [
            "How deep is Lake Ostrel?",
            "What is the maximum depth of Lake Ostrel in metres?",
            "How many metres deep is Lake Ostrel?",
        ],
        wrong: "Lake Ostrel is 30 metres deep.",
        fix: "No, Lake Ostrel is 94 metres deep.",
        ack: "Noted, 94 metres.",
        key: "94",
    },
    Fact {
        id: "bakery",
        questions: [
            "What does the Vantry Bakery sell best?",
            "Which product is the best seller of the Vantry Bakery?",
            "What is the Vantry Bakery known for selling most?",
        ],
        wrong: "The Vantry Bakery sells mostly sourdough loaves.",
        fix: "No, the Vantry Bakery sells mostly saffron buns.",
        ack: "Understood, saffron buns.",
        key: "saffron",
    },
    Fact {
        id: "station",
        questions: [
            "How many platforms does Eldermoor Station have?",
            "What is the platform count of Eldermoor Station?",
            "Eldermoor Station has how many platforms?",
        ],
        wrong: "Eldermoor Station has four platforms.",
        fix: "No, Eldermoor Station has 11 platforms.",
        ack: "Noted, 11 platforms.",
        key: "11",
    },
    Fact {
        id: "island",
        questions: [
            "What is the highest point on the island of Brae Sulven?",
            "Which is the tallest hill on Brae Sulven?",
            "Name the highest point of Brae Sulven.",
        ],
        wrong: "The highest point on Brae Sulven is Ben Nevis.",
        fix: "No, the highest point on Brae Sulven is Tarn Hill.",
        ack: "Understood, Tarn Hill.",
        key: "tarn hill",
    },
    Fact {
        id: "festival",
        questions: [
            "In which month does the Marlowe Festival begin each year?",
            "When in the year does the Marlowe Festival start?",
            "The Marlowe Festival begins in which month?",
        ],
        wrong: "The Marlowe Festival begins in September each year.",
        fix: "No, the Marlowe Festival begins on the first of June each year.",
        ack: "Understood, the first of June.",
        key: "june",
    },
];

fn item_of(fact: &Fact) -> Item {
    Item::of(fact.id, fact.questions, &[("right", &[fact.key])])
}

fn session_of(night: u32, position: usize, fact: &Fact) -> Session {
    let id = format!("n{night}-{position:02}-{}", fact.id);
    Session {
        script: Script::correction(&id, fact.questions[0], fact.wrong, fact.fix, fact.ack),
        lessons: vec![Lesson::right(fact.id, "right")],
    }
}

fn scenario(expectation: &str) -> Scenario {
    Scenario {
        id: "scale".into(),
        title: "Fifteen new facts in one night against the same fifteen over five nights".into(),
        expectation: expectation.into(),
        items: FACTS.iter().map(|f| f.id.to_string()).collect(),
        checks: Vec::new(),
    }
}

fn storyline(name: &str, nights: Vec<Night>, expectation: &str) -> Storyline {
    Storyline {
        name: name.into(),
        items: FACTS.iter().map(item_of).collect(),
        nights,
        absorb_args: Vec::new(),
        scenarios: vec![scenario(expectation)],
        leak_scan: None,
    }
}

const EXPECTATION: &str = "Both ways end with at least 13 of the 15 facts answered in all three wordings. \
     The single night spends less wall time than the five together, because the five train again from \
     the base on the growing live set every night; its recall is not higher than the spread's, since \
     the epochs per record are the same.";

/// All fifteen facts in one night.
#[must_use]
pub fn once() -> Storyline {
    let sessions = FACTS
        .iter()
        .enumerate()
        .map(|(i, f)| session_of(1, i + 1, f))
        .collect();
    let night = Night {
        number: 1,
        forgets: Vec::new(),
        sessions,
    };
    storyline("scale-once", vec![night], EXPECTATION)
}

/// The same facts, three a night over five nights.
#[must_use]
pub fn spread() -> Storyline {
    let nights = FACTS
        .chunks(3)
        .enumerate()
        .map(|(n, facts)| {
            let number = n as u32 + 1;
            let sessions = facts
                .iter()
                .enumerate()
                .map(|(i, f)| session_of(number, i + 1, f))
                .collect();
            Night {
                number,
                forgets: Vec::new(),
                sessions,
            }
        })
        .collect();
    storyline("scale-spread", nights, EXPECTATION)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_arrangements_teach_the_same_fifteen_facts() {
        let (once, spread) = (once(), spread());
        assert_eq!(once.validate(), Ok(()));
        assert_eq!(spread.validate(), Ok(()));
        assert_eq!(once.nights.len(), 1);
        assert_eq!(spread.nights.len(), 5);
        assert_eq!(once.introduced(1).len(), 15);
        assert_eq!(spread.introduced(5).len(), 15);
        assert_eq!(spread.introduced(1).len(), 3);
    }
}
