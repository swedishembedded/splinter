// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible model evaluation for its
// clients. If your team needs expertise in comparing a served model with the
// same model run in process, you can procure our services by sending an email
// to info@swedishembedded.com.

//! The sampling an in-process model is given from the configuration.

use splinter_model::local::AGENT_SAMPLING;
use splinter_model::Sampling;

use crate::config::Config;

/// How an in-process model samples when the configuration asks for anything
/// other than the agent default: the temperature, nucleus cut, `top_k` and
/// seed of a served model apply to it as they would to the same model behind
/// `brain serve`, and so does the reasoning switch, so one protocol drives
/// both. `None` when nothing is asked, which leaves the model's own
/// [`AGENT_SAMPLING`]. The per-response thinking budget is not honoured
/// in process: brain's request has no such field.
pub(crate) fn local_sampling(config: &Config) -> Option<Sampling> {
    let asked = config.thinking
        || config.served_temperature.is_some()
        || config.served_top_p.is_some()
        || config.served_top_k.is_some()
        || config.served_seed.is_some();
    asked.then(|| Sampling {
        temperature: config
            .served_temperature
            .unwrap_or(AGENT_SAMPLING.temperature),
        top_p: config.served_top_p,
        top_k: config.served_top_k.unwrap_or(AGENT_SAMPLING.top_k),
        seed: config.served_seed,
        thinking: config.thinking,
        ..AGENT_SAMPLING
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A configuration that asks nothing of the model's sampling, whatever the
    /// environment of the test run holds.
    fn unasked() -> Config {
        Config {
            thinking: false,
            served_temperature: None,
            served_top_p: None,
            served_top_k: None,
            served_seed: None,
            ..Config::from_env()
        }
    }

    #[test]
    fn an_unconfigured_run_leaves_the_agent_sampling() {
        assert_eq!(local_sampling(&unasked()), None);
    }

    #[test]
    fn the_served_sampling_of_the_configuration_reaches_an_in_process_model() {
        let seed = splinter_model::seed::draw().value();
        let config = Config {
            served_temperature: Some(0.7),
            served_top_p: Some(0.8),
            served_top_k: Some(20),
            served_seed: Some(seed),
            ..unasked()
        };
        let sampling = local_sampling(&config).unwrap();
        assert_eq!(
            sampling,
            Sampling {
                temperature: 0.7,
                top_p: Some(0.8),
                top_k: 20,
                seed: Some(seed),
                thinking: false,
                ..AGENT_SAMPLING
            }
        );
    }

    #[test]
    fn a_temperature_alone_keeps_the_agent_cut_and_draws_its_seeds() {
        let config = Config {
            served_temperature: Some(0.5),
            ..unasked()
        };
        let sampling = local_sampling(&config).unwrap();
        assert_eq!(
            (sampling.top_k, sampling.top_p, sampling.seed),
            (AGENT_SAMPLING.top_k, None, None)
        );
    }
}
