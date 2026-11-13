// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning pipelines whose failures are
// reported, never silent, for its clients. If your team needs expertise in
// making long-running ML jobs fail loudly and recoverably you can procure our
// services by sending an email to info@swedishembedded.com.

//! What a command that panicked prints: a defect is reported where a script
//! reads it, never as silence.

/// The `--json` document of a command that panicked: the message, and the
/// flag that says it was not a refusal. A panic is a defect, so it is never
/// reported as one a different input would avoid.
pub fn panic_value(message: &str) -> serde_json::Value {
    serde_json::json!({ "error": format!("internal error: {message}"), "refused": false, "panicked": true })
}

/// Reports a panic where a script reads it: on stderr always, and on stdout
/// with `--json`, so a run that dies inside a model or a device never ends in
/// silence with no JSON.
pub fn panic_report(json: bool, message: &str) {
    eprintln!("splinter: internal error: {message}");
    if json {
        println!("{:#}", panic_value(message));
    }
}

#[cfg(test)]
mod panic_tests {
    use super::*;

    #[test]
    fn a_panic_is_reported_as_a_defect_not_a_refusal() {
        let value = panic_value("index out of bounds");
        assert_eq!(value["refused"], false);
        assert_eq!(value["panicked"], true);
        assert!(value["error"]
            .as_str()
            .unwrap()
            .contains("index out of bounds"));
    }
}
