// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving, in seconds and on a CPU, that each learning method of a
// training system moves a held-out score, for its clients. If your team needs expertise in training agents against
// simulated clients and students then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Conversation self-play through the user seat: a sven agent on the toy model is set a case by a simulated seat,
//! answers it, and the case is graded; the conversations are admitted by seat provenance and measurement
//! (`conversation_admission`, so a seat that leaked the answer key trains nothing), projected into a policy-gradient
//! dataset, and the learner is trained on its own conversations. The seat is behind sven's one `Seat` interface; the
//! pipeline never learns which kind it talked to.

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use splinter_data::{write_dataset, AdvantageKind};
use splinter_model::answer::Decoding;
use splinter_pipelines::conversation_policy_gradient::{build, Graded};
use sven_sdk::model::{CompletionRequest, ModelProvider, ResponseEvent, ResponseStream};
use sven_sdk::seat::{converse, SeatSpec, SimulatedSeat};
use sven_sdk::{Engine, Toolset};

use crate::lab::{Lab, REPLY_TOKENS, SAMPLING_TEMPERATURE};
use crate::report::Outcome;
use crate::world::{cipher, is_correct, split};
use futures::StreamExt;

/// The role prompt the learner is trained under.
const ROLE: &str = "Answer the question.";

/// A model that always says the same: the seat's, which has said its piece in the opening so all it adds is that the
/// case is over (optionally with the answer a seat must never give), and the learner in a conversation that exists only
/// to show the admission what a leaking seat looks like.
struct Canned {
    reply: String,
}

#[async_trait::async_trait]
impl ModelProvider for Canned {
    fn name(&self) -> &str {
        "seat"
    }
    fn model_name(&self) -> &str {
        "seat"
    }
    async fn complete(&self, _: CompletionRequest) -> anyhow::Result<ResponseStream> {
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::TextDelta(self.reply.clone())),
            Ok(ResponseEvent::Done),
        ])))
    }
}

/// The toy model under the limit its runtime enforces: a reply is at most [`REPLY_TOKENS`] long, and running into the
/// limit ends the reply, as it would for the plain answerer, rather than asking the agent to carry on.
struct Bounded(Arc<dyn ModelProvider>);

#[async_trait::async_trait]
impl ModelProvider for Bounded {
    fn name(&self) -> &str {
        "toy"
    }
    fn model_name(&self) -> &str {
        "toy"
    }
    async fn complete(&self, mut request: CompletionRequest) -> anyhow::Result<ResponseStream> {
        request.max_output_tokens_override = Some(REPLY_TOKENS);
        let inner = self.0.complete(request).await?;
        Ok(Box::pin(inner.map(|event| match event {
            Ok(ResponseEvent::MaxTokens) => Ok(ResponseEvent::Done),
            other => other,
        })))
    }
}

/// The seat that poses `prompt` and then leaves, silently unless `leaks`, when it states the answer first.
fn seat_for(prompt: &str, leaks: bool) -> SimulatedSeat {
    let answer_line = format!("the answer is {}", cipher(prompt));
    let spec = format!(
        "id: case-{prompt}\npersona: A user asking for the cipher of a word.\ngoals: [get the cipher]\nopening: \"{prompt}\"\nanswer_key: [\"{answer_line}\"]\nseed: 1\n"
    );
    let (spec, sha) = SeatSpec::parse(&spec).unwrap();
    let reply = if leaks {
        format!("Well, {answer_line}. [END]")
    } else {
        "[END]".to_string()
    };
    let engine = Engine::builder()
        .model_provider(Arc::new(Canned { reply }))
        .toolset(Toolset::none())
        .build()
        .unwrap();
    SimulatedSeat::new(&engine, spec, sha)
}

#[test]
#[ignore = "trains a toy model: make test/methods runs it"]
fn a_learner_trained_on_its_own_admitted_conversations_with_a_seat_answers_held_out_cases() {
    let lab = Lab::new();
    let split = split();
    let warm = lab.warm_adapter(&split.train[..60], 200);
    let before = lab.accuracy(Some(&warm), &split.held_out);
    let cases = &split.train[60..];

    let mut adapter = warm.clone();
    for round in 1..=2u64 {
        let policy = lab.policy(Some(&adapter), Decoding::Greedy);
        let mut graded = Vec::new();
        let (mut leaky, mut silent) = (0, 0);
        for (i, prompt) in cases.iter().enumerate() {
            for draw in 0..8u64 {
                // One conversation in twenty is held with a seat whose model gives the answer away.
                let leaks = (i as u64 * 8 + draw).is_multiple_of(20);
                let counted = usize::from(leaks);

                let seed = round << 32 | (i as u64) << 8 | draw;
                let provider: Arc<dyn ModelProvider> = if leaks {
                    Arc::new(Canned {
                        reply: "xyz".into(),
                    })
                } else {
                    Arc::new(Bounded(policy.provider_sampled(SAMPLING_TEMPERATURE, seed)))
                };
                let engine = Engine::builder()
                    .model_provider(provider)
                    .toolset(Toolset::none())
                    .build()
                    .unwrap();
                let mut learner = engine.agent("agent");
                // A learner that answers nothing twice running fails its run: that conversation has no record to train on.
                let Ok(record) = lab.block_on(converse(&mut learner, &seat_for(prompt, leaks), 3))
                else {
                    silent += 1;
                    continue;
                };
                leaky += counted;
                let said = record
                    .exchanges
                    .iter()
                    .rev()
                    .find(|e| format!("{:?}", e.speaker) == "Learner")
                    .map(|e| e.text.trim().to_string())
                    .unwrap_or_default();
                graded.push(Graded {
                    case: prompt.clone(),
                    policy: "self".into(),
                    trial: format!("{prompt}-{round}-{draw}"),
                    reward: f64::from(u8::from(is_correct(prompt, &said))),
                    record: serde_json::to_value(&record).unwrap(),
                });
            }
        }
        let (projection, report) = build(ROLE, &graded, AdvantageKind::Positive);
        println!("self-play round {round}: {report:?}, {silent} silent runs without a record");
        assert_eq!(
            report.refused.get("AnswerKeyLeak").copied().unwrap_or(0),
            leaky,
            "every conversation with a leaking seat is refused"
        );
        assert_eq!(report.admitted, graded.len() - leaky);
        let records = lab.dir.path().join(format!("selfplay-{round}.jsonl"));
        write_dataset(&records, &projection, &splinter_data::Unchecked).unwrap();
        adapter = lab.policy_gradient(
            &format!("selfplay-{round}"),
            &records,
            200,
            3e-3,
            Some(&adapter),
        );
    }
    let after = lab.accuracy(Some(&adapter), &split.held_out);
    Outcome {
        method: "Conversation self-play (user seat)",
        metric: "held-out exact match",
        before,
        after,
        min_gain: 0.15,
        note: "",
    }
    .conclude();
}
