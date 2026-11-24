// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements machine configuration that a team can edit
// from the command line and trust to stay valid, for its clients. If your
// team needs expertise in configuration tooling, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: the one machine file. Writes are validated against the typed schema
//! and leave the file byte-identical when refused; keys with dots are
//! addressed in brackets; the schema it prints accepts what the program
//! accepts and refuses a literal credential; the two older files are
//! imported, or refused when they are all there is.

use std::path::{Path, PathBuf};

use serde_json::json;
use splinter_factory::connections::Connections;
use splinter_factory::machine_config::{self as config, MachineConfig};

fn machine() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.yaml");
    (dir, path)
}

const BRAIN: &str = r#"{protocol: openai, url: "http://127.0.0.1:8990/v1"}"#;

#[test]
fn what_is_set_is_read_back_and_reset_removes_it() {
    let (_dir, path) = machine();
    assert!(!path.exists());
    assert_eq!(
        MachineConfig::load(&path).unwrap(),
        MachineConfig::default()
    );

    config::set(&path, "connections.brain", BRAIN).unwrap();
    config::set(&path, "connections.brain.token", "env:SVEN_KEY").unwrap();
    assert_eq!(
        config::get(&path, "connections.brain.url").unwrap(),
        json!("http://127.0.0.1:8990/v1")
    );
    assert_eq!(
        config::get(&path, "connections.brain.token").unwrap(),
        json!("env:SVEN_KEY")
    );
    let written = std::fs::read_to_string(&path).unwrap();
    assert!(
        written.starts_with('#') && written.contains("splinter config"),
        "{written}"
    );

    assert!(config::reset(&path, "connections.brain.token").unwrap());
    assert!(config::get(&path, "connections.brain.token").is_err());
    assert!(!config::reset(&path, "connections.brain.token").unwrap());

    config::reset_all(&path).unwrap();
    assert_eq!(
        MachineConfig::load(&path).unwrap(),
        MachineConfig::default()
    );
}

#[test]
fn a_refused_write_leaves_the_file_byte_identical() {
    let (_dir, path) = machine();
    config::set(&path, "connections.brain", BRAIN).unwrap();
    let before = std::fs::read(&path).unwrap();
    let refusals = [
        ("connections.brain.token", "sk-literal-secret"),
        ("connections.brain.nonsense", "1"),
        ("connections.brain.protocol", "carrier-pigeon"),
        ("operators.exam", "{read: [relative/path]}"),
        ("operators.exam", "{env: [lower_case]}"),
        ("connections.brain.drop_headers", "[x-not-forwarded]"),
        ("elsewhere", "1"),
        ("connections.brain.url", "{unclosed"),
    ];
    for (key, value) in refusals {
        assert!(config::set(&path, key, value).is_err(), "{key}={value}");
        assert_eq!(std::fs::read(&path).unwrap(), before, "{key}={value}");
    }
    // Removing what a connection requires is a write the file refuses too.
    assert!(config::reset(&path, "connections.brain.protocol").is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[test]
fn a_key_with_dots_is_addressed_in_brackets() {
    let (_dir, path) = machine();
    config::set(
        &path,
        "operators[exam.sealed]",
        "{network: true, read: [/srv/sealed]}",
    )
    .unwrap();
    assert_eq!(
        config::get(&path, "operators[exam.sealed].network").unwrap(),
        json!(true)
    );
    assert_eq!(
        config::get(&path, "operators[exam.sealed].read").unwrap(),
        json!(["/srv/sealed"])
    );
    assert!(config::get(&path, "operators.exam.sealed").is_err());
    assert!(config::get(&path, "operators[exam.sealed").is_err());
}

#[test]
fn the_printed_schema_accepts_what_the_program_accepts_and_refuses_a_literal_token() {
    let schema = config::schema();
    let validator = jsonschema::validator_for(&schema).expect("a valid JSON Schema");
    let full = json!({
        "schema": "splinter.config/v1",
        "connections": {
            "brain": {"protocol": "openai", "url": "http://127.0.0.1:8990/v1", "token": "env:BRAIN_KEY"},
            "hosted": {"protocol": "anthropic", "drop_headers": ["anthropic-beta"]}
        },
        "operators": {"exam.sealed": {"read": ["/srv/a"], "write": ["/srv/b"], "network": true, "env": ["KEY_1"]}}
    });
    assert!(validator.is_valid(&json!({})));
    assert!(validator.is_valid(&serde_json::to_value(MachineConfig::default()).unwrap()));
    assert!(validator.is_valid(&full));
    let literal = json!({"connections": {"x": {"protocol": "openai", "token": "sk-secret"}}});
    assert!(!validator.is_valid(&literal));
    let unknown = json!({"connections": {"x": {"protocol": "openai", "colour": 1}}});
    assert!(!validator.is_valid(&unknown));
    // What the schema accepts the program reads.
    let text = serde_yaml::to_string(&full).unwrap();
    MachineConfig::parse(&text).unwrap();
}

fn legacy(dir: &Path, connections: &str, operators: &str) {
    std::fs::write(dir.join("connections.yaml"), connections).unwrap();
    std::fs::write(dir.join("operators.yaml"), operators).unwrap();
}

const OLD_CONNECTIONS: &str =
    "connections:\n  brain: { protocol: openai, url: \"http://127.0.0.1:8990/v1\" }\n";
const OLD_OPERATORS: &str =
    "schema: splinter.operator-grants/v1\ngrants:\n  exam: { network: true }\n";

#[test]
fn the_older_files_are_merged_into_the_one_and_removed() {
    let (dir, path) = machine();
    legacy(dir.path(), OLD_CONNECTIONS, OLD_OPERATORS);
    // They are all there is: refused, saying how to move them.
    let refused = MachineConfig::load(&path).unwrap_err().to_string();
    assert!(refused.contains("splinter config import"), "{refused}");

    let imported = config::import(&path).unwrap();
    assert_eq!(imported.connections, ["brain"]);
    assert_eq!(imported.operators, ["exam"]);
    assert!(!dir.path().join("connections.yaml").exists());
    assert!(!dir.path().join("operators.yaml").exists());
    assert_eq!(
        config::get(&path, "operators.exam.network").unwrap(),
        json!(true)
    );
}

#[test]
fn an_import_refuses_a_name_defined_twice_with_different_content() {
    let (dir, path) = machine();
    config::set(
        &path,
        "connections.brain",
        r#"{protocol: openai, url: "http://elsewhere/v1"}"#,
    )
    .unwrap();
    legacy(dir.path(), OLD_CONNECTIONS, OLD_OPERATORS);
    let before = std::fs::read(&path).unwrap();
    let refused = config::import(&path).unwrap_err().to_string();
    assert!(refused.contains("brain"), "{refused}");
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert!(dir.path().join("connections.yaml").exists());
}

#[test]
fn connections_come_from_the_machine_file_and_a_repository_only_credential_is_unvouched() {
    let (dir, path) = machine();
    let repository = dir.path().join("repo");
    std::fs::create_dir_all(&repository).unwrap();
    std::fs::write(
        repository.join("connections.yaml"),
        "connections:\n  hosted: { protocol: openai, url: \"https://shared.invalid/v1\", token: \"env:KEY\" }\n",
    )
    .unwrap();
    config::set(&path, "connections.brain", BRAIN).unwrap();
    let layered = Connections::layered(&repository, &path).unwrap();
    assert!(layered.bind(&"brain:m".parse().unwrap()).is_ok());
    let unvouched = layered
        .bind(&"hosted:m".parse().unwrap())
        .unwrap_err()
        .to_string();
    assert!(unvouched.contains("config.yaml"), "{unvouched}");
    config::set(
        &path,
        "connections.hosted",
        r#"{protocol: openai, token: "env:MINE"}"#,
    )
    .unwrap();
    let vouched = Connections::layered(&repository, &path).unwrap();
    assert!(vouched.bind(&"hosted:m".parse().unwrap()).is_ok());

    // The older file beside an absent config is refused, not read.
    let (other, absent) = machine();
    legacy(other.path(), OLD_CONNECTIONS, OLD_OPERATORS);
    let refused = Connections::layered(&repository, &absent)
        .unwrap_err()
        .to_string();
    assert!(refused.contains("splinter config import"), "{refused}");
}
