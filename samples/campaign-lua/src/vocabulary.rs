// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that let a model acquire several
// exactly verifiable skills one after another, each checked by an interpreter
// the model cannot influence, for its clients. If your team needs expertise
// in continual acquisition of verifiable task families you can procure our
// services by sending an email to info@swedishembedded.com.

//! What a family's programs and manual sections call library functions, for
//! steering the proposer toward what the policy answers worst on.

use std::collections::BTreeSet;

use crate::language::Tongue;

/// Python names that begin a call a manual section or a program makes.
const PY_BUILTINS: [&str; 40] = [
    "abs",
    "all",
    "any",
    "bin",
    "bool",
    "chr",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "hex",
    "int",
    "isinstance",
    "len",
    "list",
    "map",
    "max",
    "min",
    "oct",
    "ord",
    "pow",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
    "dict",
    "bytes",
    "next",
    "iter",
    "slice",
    "frozenset",
    "ascii",
    "complex",
];
const PY_TYPES: [&str; 6] = ["str", "list", "dict", "set", "tuple", "bytes"];
const PY_MODULES: [&str; 6] = [
    "math",
    "itertools",
    "functools",
    "string",
    "collections",
    "operator",
];

fn is_ident(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// The functions `text` calls or mentions, in `tongue`'s vocabulary.
#[must_use]
pub fn functions_in(tongue: Tongue, text: &str) -> BTreeSet<String> {
    match tongue {
        Tongue::Lua(_) => crate::sections::functions_in(text),
        Tongue::Python => python_functions_in(text),
    }
}

fn python_functions_in(text: &str) -> BTreeSet<String> {
    let bytes = text.as_bytes();
    let mut found = BTreeSet::new();
    let mut i = 0;
    while i < bytes.len() {
        if !(bytes[i].is_ascii_alphabetic() || bytes[i] == b'_') {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && is_ident(bytes[i]) {
            i += 1;
        }
        let word = &text[start..i];
        let after_dot = start > 0 && bytes[start - 1] == b'.';
        let called = bytes.get(i) == Some(&b'(');
        if after_dot && called {
            // `.join(`, `.split(`: a method, whatever the receiver.
            found.insert(format!("method.{word}"));
        } else if !after_dot && bytes.get(i) == Some(&b'.') {
            let member = i + 1;
            let mut end = member;
            while end < bytes.len() && is_ident(bytes[end]) {
                end += 1;
            }
            if bytes.get(end) == Some(&b'(') && end > member {
                let name = &text[member..end];
                if PY_TYPES.contains(&word) {
                    found.insert(format!("method.{name}"));
                } else if PY_MODULES.contains(&word) {
                    found.insert(format!("{word}.{name}"));
                }
                i = end;
            }
        } else if !after_dot && called && PY_BUILTINS.contains(&word) {
            found.insert(format!("builtin.{word}"));
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_functions_are_builtins_methods_and_module_functions() {
        let found =
            python_functions_in("print(\",\".join(sorted(x)))\nmath.floor(2.5) + str.upper(s)");
        let expected: BTreeSet<String> = [
            "method.join",
            "builtin.sorted",
            "math.floor",
            "method.upper",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(found, expected);
    }
}
