// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements knowledge ingestion that refuses a damaged
// source or a hostile reply by name instead of crashing, for its clients. If
// your team needs expertise in property-based testing of document and data
// ingestion then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec, by property: what knowledge reads from outside is answered, never a
//! panic, however it is damaged: a SAS transport file (binary, with lengths
//! and offsets of its own), a codebook page, a session's part name, and a
//! model's reply to a stance question, any Unicode text.
#![allow(clippy::unwrap_used)]

use proptest::prelude::*;
use splinter_conformance::fuzz::{config, damaged_bytes, damaged_text};
use splinter_knowledge::codebook::parse_html;
use splinter_knowledge::session::parse_part_name;
use splinter_knowledge::stance;
use splinter_knowledge::tabular::read_xport;

const XPORT: &[u8] = include_bytes!("../fixtures/mixed.xpt");

const PAGE: &str = r#"<div class="pagebreak">
  <h3 class="vartitle" id="SMQ040">SMQ040 - Do you now smoke cigarettes?</h3>
  <dl><dt>Variable Name: </dt><dd class="info">SMQ040</dd>
      <dt>SAS Label: </dt><dd>Do you now smoke cigarettes?</dd>
      <dt>English Text: </dt><dd class="info">Do you now smoke cigarettes . . .</dd>
      <dt>Target: </dt><dd> Both males and females 18 YEARS - 150 YEARS</dd></dl>
  <table class="values"><thead><tr><th>Code or Value</th></tr></thead><tbody>
    <tr><td scope="row" class="values">1</td><td class="values">Every day</td><td class="values" align="right">1,034</td></tr>
    <tr><td scope="row" class="values">.</td><td class="values">Missing</td><td class="values" align="right">4005</td></tr>
  </tbody></table></div>"#;

const STATEMENT: &str = "The build server is called atlas and runs on port 8080.";
const QUESTION: &str = "What is the build server called?";

#[test]
fn every_seed_is_read_as_written() {
    read_xport(XPORT).unwrap();
    assert_eq!(parse_html(PAGE).len(), 1);
    assert!(!stance::keys(STATEMENT, QUESTION).is_empty());
}

/// Every prefix of a transport file is answered, never read past its end;
/// one cut before its rows begin (inside a header or a variable's
/// description) is refused. A cut among the rows cannot be told from fewer
/// rows: the format records no row count.
#[test]
fn a_transport_file_cut_short_anywhere_is_answered_and_refused_before_its_rows() {
    let tag = b"HEADER RECORD*******OBS     HEADER RECORD";
    let rows = XPORT.windows(tag.len()).position(|w| w == tag).unwrap() + 80;
    for end in 0..XPORT.len() {
        let read = read_xport(&XPORT[..end]);
        if end < rows {
            assert!(
                read.is_err(),
                "a file cut at {end}, before its rows at {rows}, was read"
            );
        }
    }
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn a_damaged_transport_file_is_answered(bytes in damaged_bytes(XPORT.to_vec())) {
        let _ = read_xport(&bytes);
    }

    #[test]
    fn a_damaged_codebook_page_or_part_name_is_answered(
        page in damaged_text(PAGE),
        name in damaged_text("0007-assistant.md"),
    ) {
        let _ = parse_html(&page);
        let _ = parse_part_name(&name);
    }

    #[test]
    fn any_reply_to_a_stance_question_is_answered(
        reply in "\\PC{0,60}",
        statement in damaged_text(STATEMENT),
        seed in any::<u64>(),
    ) {
        for key in stance::keys(&statement, QUESTION) {
            let _ = stance::corrupt(&statement, &key, &["zeus".into(), "Ωmega".into()], seed);
            let _ = stance::holds(&reply, &key, "zeus");
            let _ = stance::adopts(&reply, &key, "zeus");
        }
        let _ = stance::invented_term(&reply, &statement);
        let _ = stance::declines(&reply, QUESTION);
    }
}
