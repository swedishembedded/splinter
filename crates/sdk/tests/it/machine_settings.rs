// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements machine configuration that a team can edit
// from the command line and trust to stay valid, for its clients. If your
// team needs expertise in configuration tooling, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: the factory takes this machine's connections and operator grants
//! from the one configuration file; the older separate files alone are
//! refused, not read.

#![allow(clippy::unwrap_used)]

use splinter_sdk::factory::machine_config;
use splinter_sdk::factory::{connections_of, grants_of};
use splinter_sdk::Config;

fn config_in(dir: &std::path::Path) -> Config {
    let mut config = Config::from_env();
    config.factory.config = dir.join("config.yaml");
    config.working_dir = dir.to_path_buf();
    config
}

#[test]
fn connections_and_grants_are_read_from_the_one_file() {
    let dir = tempfile::tempdir().unwrap();
    let config = config_in(dir.path());
    let file = &config.factory.config;
    machine_config::set(
        file,
        "connections.brain",
        r#"{protocol: openai, url: "http://127.0.0.1:1/v1"}"#,
    )
    .unwrap();
    machine_config::set(file, "operators[exam.sealed]", "{network: true}").unwrap();
    let connections = connections_of(&config, dir.path()).unwrap();
    assert!(connections.bind(&"brain:m".parse().unwrap()).is_ok());
    assert!(grants_of(&config).unwrap().of("exam.sealed").network);
}

#[test]
fn the_older_files_alone_are_refused_naming_the_way_to_move_them() {
    let dir = tempfile::tempdir().unwrap();
    let config = config_in(dir.path());
    std::fs::write(
        dir.path().join("operators.yaml"),
        "schema: splinter.operator-grants/v1\ngrants:\n  exam: { network: true }\n",
    )
    .unwrap();
    let refused = grants_of(&config).unwrap_err().to_string();
    assert!(refused.contains("splinter config import"), "{refused}");
    assert!(connections_of(&config, dir.path()).is_err());
}
