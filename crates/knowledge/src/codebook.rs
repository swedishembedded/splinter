// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Codebooks: what a record file's variables mean, as the data's publisher
//! wrote it down - the text an intake agent reads and an admission rule
//! checks a proposal's quotations against.
//!
//! [`parse_html`] reads the per-variable HTML codebook layout public survey
//! programmes publish (a `vartitle` heading per variable, then
//! name/label/text/target, then a table of codes or value ranges with their
//! meanings and counts).

use serde::{Deserialize, Serialize};

/// One coded value or value range and what it means.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Code {
    /// The code, a range (`0.08 to 188.5`) or `.` for missing.
    pub value: String,
    /// What it means (`Refused`, `Range of Values`, `Missing`).
    pub meaning: String,
    /// How many records carry it, when stated.
    pub count: Option<u64>,
}

/// One variable as documented.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VariableDoc {
    /// The variable's name.
    pub name: String,
    /// Its short label.
    pub label: String,
    /// The question or measurement, in full.
    pub text: String,
    /// Who it was asked of or measured on.
    pub target: String,
    /// Its codes and value ranges.
    pub codes: Vec<Code>,
}

impl VariableDoc {
    /// Everything documented about the variable as one text: what a
    /// quotation is checked against.
    #[must_use]
    pub fn full_text(&self) -> String {
        let mut t = format!(
            "{}\n{}\n{}\n{}\n",
            self.name, self.label, self.text, self.target
        );
        for c in &self.codes {
            t.push_str(&format!("{} {}\n", c.value, c.meaning));
        }
        t
    }
}

fn strip_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            c if !in_tag => out.push(c),
            _ => {}
        }
    }
    let unescaped = out
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&nbsp;", " ")
        .replace("&#39;", "'")
        .replace("&quot;", "\"")
        .replace("&amp;", "&");
    unescaped.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The text of the element after `marker` up to `end`, tags stripped.
fn after(section: &str, marker: &str, end: &str) -> String {
    section.find(marker).map_or(String::new(), |i| {
        let rest = &section[i + marker.len()..];
        let j = rest.find(end).unwrap_or(rest.len());
        strip_tags(&rest[..j])
    })
}

/// Every variable of an HTML codebook in that layout.
#[must_use]
pub fn parse_html(html: &str) -> Vec<VariableDoc> {
    let mut out = Vec::new();
    let starts: Vec<usize> = html
        .match_indices("class=\"vartitle\"")
        .map(|(i, _)| i)
        .collect();
    for (n, &start) in starts.iter().enumerate() {
        let end = starts.get(n + 1).copied().unwrap_or(html.len());
        let section = &html[start..end];
        let name = after(section, "Variable Name:", "</dd>");
        if name.is_empty() {
            continue;
        }
        let mut codes = Vec::new();
        if let Some(body) = section.find("<tbody>").map(|i| &section[i..]) {
            for row in body.split("<tr>").skip(1) {
                let cells: Vec<String> = row
                    .split("<td")
                    .skip(1)
                    .map(|c| strip_tags(&format!("<td{c}")))
                    .collect();
                if cells.len() >= 2 {
                    codes.push(Code {
                        value: cells[0].clone(),
                        meaning: cells[1].clone(),
                        count: cells.get(2).and_then(|c| c.replace(',', "").parse().ok()),
                    });
                }
            }
        }
        out.push(VariableDoc {
            name,
            label: after(section, "SAS Label:", "</dd>"),
            text: after(section, "English Text:", "</dd>"),
            target: after(section, "Target:", "</dd>"),
            codes,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: &str = r#"<div class="pagebreak">
      <h3 class="vartitle" id="SMQ040">SMQ040 - Do you now smoke cigarettes?</h3>
      <dl><dt>Variable Name: </dt><dd class="info">SMQ040</dd>
          <dt>SAS Label: </dt><dd>Do you now smoke cigarettes?</dd>
          <dt>English Text: </dt><dd class="info">Do you now smoke cigarettes . . .</dd>
          <dt>Target: </dt><dd> Both males and females 18 YEARS -
            150 YEARS</dd></dl>
      <table class="values"><thead><tr><th>Code or Value</th></tr></thead><tbody>
        <tr><td scope="row" class="values">1</td><td class="values">Every day</td><td class="values" align="right">1,034</td><td>1034</td></tr>
        <tr><td scope="row" class="values">7</td><td class="values">Refused</td><td class="values" align="right">0</td></tr>
        <tr><td scope="row" class="values">.</td><td class="values">Missing</td><td class="values" align="right">4005</td></tr>
      </tbody></table></div>
      <div class="pagebreak"><h3 class="vartitle" id="LBXHSCRP">LBXHSCRP - HS C-Reactive Protein (mg/L)</h3>
      <dl><dt>Variable Name: </dt><dd class="info">LBXHSCRP</dd><dt>SAS Label: </dt><dd>HS C-Reactive Protein (mg/L)</dd>
      <dt>English Text: </dt><dd class="info">High-Sensitivity C-Reactive Protein (hs-CRP) (mg/L)</dd><dt>Target: </dt><dd>Both</dd></dl>
      <table class="values"><tbody><tr><td scope="row" class="values">0.08 to 188.5</td><td class="values">Range of Values</td><td class="values">7867</td></tr></tbody></table></div>"#;

    #[test]
    fn variables_codes_and_counts_are_read() {
        let v = parse_html(PAGE);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].name, "SMQ040");
        assert_eq!(v[0].text, "Do you now smoke cigarettes . . .");
        assert_eq!(v[0].target, "Both males and females 18 YEARS - 150 YEARS");
        assert_eq!(
            v[0].codes[0],
            Code {
                value: "1".into(),
                meaning: "Every day".into(),
                count: Some(1034)
            }
        );
        assert_eq!(v[0].codes[1].meaning, "Refused");
        assert_eq!(v[1].label, "HS C-Reactive Protein (mg/L)");
        assert_eq!(v[1].codes[0].value, "0.08 to 188.5");
        assert!(v[1].full_text().contains("(mg/L)"));
    }
}
