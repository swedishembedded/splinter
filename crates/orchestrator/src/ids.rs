// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Content-addressed ids as a person types them: the full `blake3:<hex>`,
//! the hex alone, or a prefix of it that names exactly one stored object.

use splinter_core::digest::Digest;

use crate::error::OrchestratorError;

/// The fewest hex digits a prefix must give: fewer names too many objects
/// to be worth resolving.
pub const MIN_PREFIX: usize = 4;

/// `given` without a leading `blake3:` or `sha256:`.
pub fn strip_algorithm(given: &str) -> &str {
    given
        .strip_prefix("blake3:")
        .or_else(|| given.strip_prefix("sha256:"))
        .unwrap_or(given)
}

/// The one digest of `stored` that `given` names; `what` names the kind of
/// object in a refusal.
pub fn resolve(
    what: &'static str,
    given: &str,
    stored: impl IntoIterator<Item = Digest>,
) -> Result<Digest, OrchestratorError> {
    let hex = strip_algorithm(given);
    let not_found = || OrchestratorError::NotFound {
        what,
        id: given.to_string(),
    };
    if hex.len() < MIN_PREFIX || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(OrchestratorError::Refused(format!(
            "{given:?} is not a {what} id: give blake3:<hex>, or at least {MIN_PREFIX} hex digits \
             of one"
        )));
    }
    let hex = hex.to_ascii_lowercase();
    let mut matches = stored
        .into_iter()
        .filter(|digest| digest.hex().starts_with(&hex));
    let found = matches.next().ok_or_else(not_found)?;
    let more = matches.count();
    if more > 0 {
        return Err(OrchestratorError::AmbiguousId {
            what,
            id: given.to_string(),
            matches: more + 1,
        });
    }
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_full_id_or_a_unique_prefix_names_one_object() {
        let a = Digest::of(b"a");
        let b = Digest::of(b"b");
        let stored = || vec![a.clone(), b.clone()];
        assert_eq!(
            resolve("source", a.as_str(), stored()).ok(),
            Some(a.clone())
        );
        assert_eq!(resolve("source", a.hex(), stored()).ok(), Some(a.clone()));
        assert_eq!(
            resolve("source", &a.hex()[..6], stored()).ok(),
            Some(a.clone())
        );
        assert!(matches!(
            resolve("source", "ab", stored()),
            Err(OrchestratorError::Refused(_))
        ));
        assert!(matches!(
            resolve("source", "not-hex", stored()),
            Err(OrchestratorError::Refused(_))
        ));
        assert!(matches!(
            resolve("source", "0000000000", stored()),
            Err(OrchestratorError::NotFound { .. })
        ));
        let twins = vec![a.clone(), a.clone()];
        assert!(matches!(
            resolve("source", &a.hex()[..6], twins),
            Err(OrchestratorError::AmbiguousId { matches: 2, .. })
        ));
    }
}
