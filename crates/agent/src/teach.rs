// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that learn facts, rules and workflows
// from what their users tell them, for its clients. If your team needs
// expertise in making an agent correctable by the people who work with it, you
// can procure our services by sending an email to info@swedishembedded.com.

//! An agent that can be taught by what the user says to it.
//!
//! Each user turn is one of four things. A teaching ("actually the deadline is
//! Monday", "never push without asking") is not sent to the model as a
//! question: the agent says back what it understood and asks whether that is
//! right, and keeps it only on a plain yes. A retraction ("forget that") takes
//! the last teaching back. A plain answer to a pending question that is
//! neither a yes nor a no drops the question and is treated as an ordinary
//! turn. Everything else is an ordinary turn, asked of the agent with what it
//! has been taught that bears on it put in front of it.
//!
//! The words the agent says on its own account (the restatement, the
//! acknowledgements) go through the same callback as the model's, so a spoken
//! front end says them like any other reply.

use splinter_core::speech_lesson::{parse_confirmation, parse_retraction};
use splinter_core::teaching::{parse_teaching, render, Force, Ledger, Teaching};

use crate::CallError;

/// How many facts and procedures a turn is told about besides every rule.
const TOLD_PER_TURN: usize = 5;

/// What an agent needs to be taught: to be asked, and to be told what it has
/// been taught before it answers.
pub trait Asker {
    /// Run one turn on `question`, handing `on_text` each piece of the reply.
    ///
    /// # Errors
    /// The turn failed.
    fn ask(&self, question: &str, on_text: &mut dyn FnMut(&str)) -> Result<String, CallError>;

    /// Set what the agent has been taught that bears on the next turn; empty
    /// when nothing does.
    fn taught(&self, text: &str);
}

impl<T: Asker + ?Sized> Asker for &T {
    fn ask(&self, question: &str, on_text: &mut dyn FnMut(&str)) -> Result<String, CallError> {
        (**self).ask(question, on_text)
    }

    fn taught(&self, text: &str) {
        (**self).taught(text);
    }
}

/// An [`Asker`] that learns from what it is told.
pub struct Teachable<A: Asker> {
    agent: A,
    ledger: Ledger,
    pending: Option<(Teaching, String)>,
    turn: u64,
}

impl<A: Asker> Teachable<A> {
    /// `agent`, with what `ledger` already holds.
    pub fn new(agent: A, ledger: Ledger) -> Self {
        Self {
            agent,
            ledger,
            pending: None,
            turn: 0,
        }
    }

    /// What the agent has been taught.
    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    /// Handle what the user said, handing `on_text` the reply as it comes.
    /// Returns the reply.
    ///
    /// # Errors
    /// An ordinary turn failed.
    pub fn hear(&mut self, said: &str, on_text: &mut dyn FnMut(&str)) -> Result<String, CallError> {
        self.turn += 1;
        if let Some((teaching, heard)) = self.pending.take() {
            match parse_confirmation(said) {
                Some(true) => {
                    self.ledger.learn(teaching, &heard, self.turn);
                    return Ok(say(on_text, "Understood. I will remember that."));
                }
                Some(false) => return Ok(say(on_text, "Alright, I will not keep that.")),
                None => {}
            }
        }
        if parse_retraction(said) {
            let reply = if self.ledger.retract_last().is_some() {
                "Forgotten."
            } else {
                "I have nothing you taught me to forget."
            };
            return Ok(say(on_text, reply));
        }
        if let Some(teaching) = parse_teaching(said) {
            let reply = restate(&teaching);
            self.pending = Some((teaching, said.to_string()));
            return Ok(say(on_text, &reply));
        }
        self.agent
            .taught(&render(&self.ledger.relevant(said, TOLD_PER_TURN)));
        self.agent.ask(said, on_text)
    }
}

fn say(on_text: &mut dyn FnMut(&str), words: &str) -> String {
    on_text(words);
    words.to_string()
}

/// What the agent says back so the user can tell whether it understood.
fn restate(teaching: &Teaching) -> String {
    match teaching {
        Teaching::Fact { claim, .. } => {
            format!("So {}. Is that right?", claim.trim_end_matches('.'))
        }
        Teaching::Rule { instruction, force } => {
            let force = if *force == Force::Always {
                "always"
            } else {
                "never"
            };
            format!(
                "So from now on I will {force} {} Is that right?",
                instruction.trim_end_matches('.').to_owned() + "."
            )
        }
        Teaching::Procedure { name, steps } => {
            format!("So to {name}, I {}. Is that right?", steps.join(", then "))
        }
    }
}

#[cfg(test)]
mod specs {
    use std::cell::RefCell;

    use super::*;

    /// An agent that answers every question "ok" and remembers what it was
    /// asked and told.
    #[derive(Default)]
    struct Recorder {
        asked: RefCell<Vec<String>>,
        told: RefCell<String>,
    }

    impl Asker for Recorder {
        fn ask(&self, question: &str, on_text: &mut dyn FnMut(&str)) -> Result<String, CallError> {
            self.asked.borrow_mut().push(question.to_string());
            on_text("ok");
            Ok("ok".to_string())
        }

        fn taught(&self, text: &str) {
            *self.told.borrow_mut() = text.to_string();
        }
    }

    fn hear(t: &mut Teachable<&Recorder>, said: &str) -> String {
        t.hear(said, &mut |_| {}).unwrap()
    }

    #[test]
    fn a_teaching_is_confirmed_before_it_is_kept_and_is_not_asked_of_the_model() {
        let model = Recorder::default();
        let mut agent = Teachable::new(&model, Ledger::new());
        let said = hear(&mut agent, "Actually, the deadline is Monday.");
        assert!(
            said.contains("the deadline is Monday") && said.ends_with("Is that right?"),
            "{said}"
        );
        assert_eq!(
            agent.ledger().active().count(),
            0,
            "not kept before the yes"
        );
        assert_eq!(
            hear(&mut agent, "Yes."),
            "Understood. I will remember that."
        );
        assert_eq!(agent.ledger().active().count(), 1);
        assert!(
            model.asked.borrow().is_empty(),
            "teaching and confirmation never reach the model"
        );
    }

    #[test]
    fn a_refused_teaching_is_not_kept() {
        let model = Recorder::default();
        let mut agent = Teachable::new(&model, Ledger::new());
        hear(&mut agent, "Never push to main without asking me first.");
        hear(&mut agent, "No.");
        assert_eq!(agent.ledger().active().count(), 0);
    }

    #[test]
    fn what_was_taught_is_put_before_the_model_on_the_turns_it_bears_on() {
        let model = Recorder::default();
        let mut agent = Teachable::new(&model, Ledger::new());
        hear(&mut agent, "Actually, the deadline is Monday.");
        hear(&mut agent, "Yes.");
        hear(&mut agent, "When is the deadline?");
        assert!(model.told.borrow().contains("the deadline is Monday."));
        hear(&mut agent, "How is the weather?");
        assert_eq!(
            *model.told.borrow(),
            "",
            "a turn it does not bear on is told nothing"
        );
        assert_eq!(
            *model.asked.borrow(),
            ["When is the deadline?", "How is the weather?"]
        );
    }

    #[test]
    fn a_retraction_takes_back_the_last_teaching_and_the_model_is_no_longer_told() {
        let model = Recorder::default();
        let mut agent = Teachable::new(&model, Ledger::new());
        hear(
            &mut agent,
            "From now on always confirm before deleting a file.",
        );
        hear(&mut agent, "Yes.");
        assert_eq!(hear(&mut agent, "Forget that."), "Forgotten.");
        hear(&mut agent, "Please delete the old log.");
        assert_eq!(*model.told.borrow(), "");
        assert_eq!(
            hear(&mut agent, "Forget that."),
            "I have nothing you taught me to forget."
        );
    }

    #[test]
    fn an_answer_that_is_neither_yes_nor_no_drops_the_question_and_is_an_ordinary_turn() {
        let model = Recorder::default();
        let mut agent = Teachable::new(&model, Ledger::new());
        hear(
            &mut agent,
            "Remember that the staging database is read only.",
        );
        hear(&mut agent, "What time is it?");
        assert_eq!(*model.asked.borrow(), ["What time is it?"]);
        hear(&mut agent, "Yes.");
        assert_eq!(
            agent.ledger().active().count(),
            0,
            "the question had been dropped"
        );
    }

    #[test]
    fn the_agents_own_words_reach_the_callback_like_the_models() {
        let model = Recorder::default();
        let mut agent = Teachable::new(&model, Ledger::new());
        let mut spoken = String::new();
        agent
            .hear("Actually, the deadline is Monday.", &mut |piece| {
                spoken.push_str(piece)
            })
            .unwrap();
        assert!(spoken.contains("the deadline is Monday"));
    }
}
