// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements leakage-free held-out splits of
// code-editing task sets, for its clients. If your team needs expertise in
// holding out whole task families before any training data is built, you
// can procure our services by sending an email to info@swedishembedded.com.

//! What a task family is, and how a task that spans several of them is
//! named.
//!
//! A workspace task's family is the file its fault was seeded in. A task
//! made of several independent faults has a **composite family**: the sorted
//! families of its faults joined by [`COMPOSITE_SEPARATOR`]. The name is
//! stable (it does not depend on the order the faults were met in) and says
//! which families the task draws on, so a split by family can ask whether any
//! of them is reserved: a task is as reserved as its most reserved member,
//! never less ([`touches`]).
//!
//! [`rank`] is the stable shuffle every split by family sorts by.

use std::collections::BTreeSet;

use crate::digest::Digest;
use crate::rng::Seed;

/// What joins the member families in a composite family's name. No file
/// path holds it (a space on both sides of a plus), and [`composite`]
/// refuses a member that does.
pub const COMPOSITE_SEPARATOR: &str = " + ";

/// The composite family of `members`: their sorted, distinct names joined by
/// [`COMPOSITE_SEPARATOR`]. `None` when fewer than two distinct families are
/// given, or one of them holds the separator (the name could not be taken
/// apart again).
#[must_use]
pub fn composite<S: AsRef<str>>(members: &[S]) -> Option<String> {
    let names: BTreeSet<&str> = members.iter().map(AsRef::as_ref).collect();
    if names.len() < 2 || names.iter().any(|name| name.contains(COMPOSITE_SEPARATOR)) {
        return None;
    }
    Some(
        names
            .into_iter()
            .collect::<Vec<_>>()
            .join(COMPOSITE_SEPARATOR),
    )
}

/// The families `family` is made of: itself for an ordinary family, each
/// member for a composite one.
pub fn members(family: &str) -> impl Iterator<Item = &str> {
    family.split(COMPOSITE_SEPARATOR)
}

/// The first family of `family` that is in `reserved`, when any is: the
/// reason a task of that family belongs to the reservation.
#[must_use]
pub fn touches<'a>(family: &'a str, reserved: &BTreeSet<String>) -> Option<&'a str> {
    members(family).find(|member| reserved.contains(*member))
}

/// How `key` ranks under `seed`: the hash of its name. Sorting by it is a
/// shuffle that depends on the seed and the names alone, never on the order
/// the names were met in.
#[must_use]
pub fn rank(key: &str, seed: Seed) -> String {
    let mut input = seed.value().to_le_bytes().to_vec();
    input.extend_from_slice(key.as_bytes());
    Digest::of(&input).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_composite_family_is_the_same_whatever_order_its_members_are_met_in() {
        let one = composite(&["src/b.py", "src/a.py", "src/c.py"]).unwrap();
        let other = composite(&["src/c.py", "src/a.py", "src/b.py", "src/a.py"]).unwrap();
        assert_eq!(one, other);
        assert_eq!(one, "src/a.py + src/b.py + src/c.py");
        assert_eq!(
            members(&one).collect::<Vec<_>>(),
            ["src/a.py", "src/b.py", "src/c.py"]
        );
    }

    #[test]
    fn fewer_than_two_distinct_families_or_a_member_that_holds_the_separator_is_no_composite() {
        assert_eq!(composite(&["a.py"]), None);
        assert_eq!(composite(&["a.py", "a.py"]), None);
        assert_eq!(composite(&["a.py", "b.py + c.py"]), None);
    }

    #[test]
    fn an_ordinary_family_is_its_own_only_member() {
        assert_eq!(members("src/a.py").collect::<Vec<_>>(), ["src/a.py"]);
    }

    #[test]
    fn a_composite_touches_a_reservation_through_any_one_member() {
        let reserved: BTreeSet<String> = ["src/b.py".to_string()].into();
        assert_eq!(touches("src/a.py + src/b.py", &reserved), Some("src/b.py"));
        assert_eq!(touches("src/b.py", &reserved), Some("src/b.py"));
        assert_eq!(touches("src/a.py + src/c.py", &reserved), None);
    }
}
