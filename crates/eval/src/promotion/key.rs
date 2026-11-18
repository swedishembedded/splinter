// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for promotion gates that check an
// examiner's signature in the learner's own process, for its clients. If your
// team needs expertise in tamper-evident evaluation evidence you can procure
// our services by sending an email to info@swedishembedded.com.

//! The examiner's public key: what the gate verifies a verdict's Ed25519
//! signature with. The gate holds no signing key; the examiner's private key
//! never leaves the examiner.

use std::path::Path;

use ed25519_dalek::{Signature, VerifyingKey};

/// The bytes of an Ed25519 public key.
pub const PUBLIC_KEY_BYTES: usize = 32;

/// The examiner's verdict public key.
#[derive(Clone, PartialEq, Eq)]
pub struct PublicKey(VerifyingKey);

impl std::fmt::Debug for PublicKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PublicKey({})", hex(self.0.as_bytes()))
    }
}

/// Why a public key cannot be used.
#[derive(Debug, thiserror::Error)]
pub enum KeyError {
    /// The file cannot be read.
    #[error("{path}: cannot read the examiner's public key ({reason})")]
    Read {
        /// The file.
        path: String,
        /// Why.
        reason: String,
    },
    /// It is not a public key.
    #[error(
        "{path}: not the examiner's public key (one line of {} hex digits, the `.pub` file its key generation writes): {reason}",
        PUBLIC_KEY_BYTES * 2
    )]
    Malformed {
        /// The file, or `bytes` for a key given as bytes.
        path: String,
        /// Why.
        reason: String,
    },
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(text.get(i..i + 2)?, 16).ok())
        .collect()
}

impl PublicKey {
    /// The key whose 32 raw bytes are `bytes`.
    ///
    /// # Errors
    /// The bytes are not a point of the curve.
    pub fn from_bytes(bytes: [u8; PUBLIC_KEY_BYTES]) -> Result<Self, KeyError> {
        VerifyingKey::from_bytes(&bytes)
            .map(Self)
            .map_err(|e| KeyError::Malformed {
                path: "bytes".into(),
                reason: e.to_string(),
            })
    }

    /// The key in the file at `path`: one line of 64 lower- or upper-case
    /// hex digits.
    ///
    /// # Errors
    /// The file cannot be read or holds no public key.
    pub fn read(path: &Path) -> Result<Self, KeyError> {
        let shown = path.display().to_string();
        let bytes = std::fs::read(path).map_err(|e| KeyError::Read {
            path: shown.clone(),
            reason: e.to_string(),
        })?;
        let malformed = |reason: &str| KeyError::Malformed {
            path: shown.clone(),
            reason: reason.into(),
        };
        let text = std::str::from_utf8(&bytes).map_err(|_| malformed("not text"))?;
        let raw = unhex(text.trim()).ok_or_else(|| malformed("not hex"))?;
        let raw: [u8; PUBLIC_KEY_BYTES] = raw.try_into().map_err(|_| malformed("not 32 bytes"))?;
        Self::from_bytes(raw).map_err(|_| malformed("not a curve point"))
    }

    /// Whether `signature_hex` is a valid signature of `message` under this
    /// key (strict verification: no malleable or small-order forms).
    #[must_use]
    pub fn verifies(&self, message: &[u8], signature_hex: &str) -> bool {
        let Some(raw) = unhex(signature_hex) else {
            return false;
        };
        let Ok(signature) = Signature::from_slice(&raw) else {
            return false;
        };
        self.0.verify_strict(message, &signature).is_ok()
    }

    /// The key's id as the examiner names it in a signature: the first 16 hex
    /// digits of the sha256 of its 32 raw bytes.
    #[must_use]
    pub fn key_id(&self) -> String {
        use sha2::{Digest as _, Sha256};
        hex(&Sha256::digest(self.0.as_bytes())[..8])
    }
}
