// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifiers that prove a project's tests
// really ran before they count a pass, for its clients. If your team needs
// expertise in tamper-resistant evaluation of code-editing agents, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What a failed `unittest` test said it expected, read from the runner's
//! failure blocks.
//!
//! A failure block is the test's name, a traceback and the exception's own
//! message. The traceback names files and prints the lines of the test that
//! ran, which is the test's code; the message is what the failed check
//! observed (`first != second`, an error the code under test raised). Only
//! the message is kept, never a frame: it is behaviour, and the code of a
//! hidden test is not.

use std::collections::BTreeMap;

use super::test_report::unittest_failure_id;

/// The final exception message of each failed test in `stderr`, by test id.
pub(super) fn unittest_messages(stderr: &str) -> BTreeMap<String, String> {
    let mut messages = BTreeMap::new();
    let mut lines = stderr.lines().peekable();
    while let Some(line) = lines.next() {
        let Some(id) = unittest_failure_id(line) else {
            continue;
        };
        // Past the dashes that close the header (a docstring may sit between).
        for next in lines.by_ref() {
            if next.starts_with("------") {
                break;
            }
        }
        let mut block = Vec::new();
        while let Some(next) = lines.peek() {
            if next.starts_with("======") || next.starts_with("------") {
                break;
            }
            block.push(*next);
            lines.next();
        }
        if let Some(message) = final_message(&block) {
            messages.insert(id, message);
        }
    }
    messages
}

/// The lines after the last traceback's frames, without trailing blanks.
fn final_message(block: &[&str]) -> Option<String> {
    let after = block
        .iter()
        .rposition(|l| l.starts_with("Traceback (most recent call last):"))
        .map_or(0, |at| at + 1);
    let mut rest = block[after..]
        .iter()
        .skip_while(|l| l.starts_with([' ', '\t']));
    let first = rest.next()?;
    let mut message = vec![*first];
    message.extend(rest);
    while message.last().is_some_and(|l| l.trim().is_empty()) {
        message.pop();
    }
    Some(message.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const FAILURES: &str = "\
test_a (m.T.test_a) ... FAIL
test_b (m.T.test_b) ... ERROR
test_c (m.T.test_c) ... ok

======================================================================
FAIL: test_a (m.T.test_a)
A docstring line.
----------------------------------------------------------------------
Traceback (most recent call last):
  File \"/work/tests/m.py\", line 9, in test_a
    self.assertEqual(add(1, 2), 4)
AssertionError: 3 != 4

======================================================================
ERROR: test_b (m.T.test_b)
----------------------------------------------------------------------
Traceback (most recent call last):
  File \"/work/tests/m.py\", line 12, in test_b
    parse('x')
  File \"/work/pkg/p.py\", line 3, in parse
    raise ParseError('bad token')
pkg.ParseError: bad token

----------------------------------------------------------------------
Ran 3 tests in 0.001s

FAILED (failures=1, errors=1)
";

    #[test]
    fn the_message_of_each_failure_is_kept_and_no_frame_of_its_traceback() {
        let messages = unittest_messages(FAILURES);
        assert_eq!(messages["m.T.test_a"], "AssertionError: 3 != 4");
        assert_eq!(messages["m.T.test_b"], "pkg.ParseError: bad token");
        assert_eq!(messages.len(), 2);
        let all = messages.values().cloned().collect::<String>();
        assert!(!all.contains("/work") && !all.contains("assertEqual"));
    }

    #[test]
    fn a_multi_line_message_is_kept_whole() {
        let stderr = "======\nFAIL: test_a (m.T.test_a)\n------\nTraceback (most recent call last):\n  File \"x\", line 1, in f\n    g()\nAssertionError: 'a' != 'b'\n- a\n+ b\n\n------\nRan 1 test in 0.1s\n";
        assert_eq!(
            unittest_messages(stderr)["m.T.test_a"],
            "AssertionError: 'a' != 'b'\n- a\n+ b"
        );
    }
}
