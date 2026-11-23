// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements data pipelines that fan work out over large
// corpora with stable item identities, for its clients. If your team needs
// expertise in resumable, content-addressed data factories, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The item index of a collection product (`splinter.items/1`).
//!
//! A product a stage can run once per item of (`foreach`) lists its items in
//! [`ITEMS_FILE`] at its root: each item's stable id and where it is in the
//! product. A fan-out's own product is such a collection, and also lists the
//! items its operation rejected, with the reason. Items are kept in id
//! order, so the index of the same items is the same file.

use std::collections::BTreeSet;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The schema of an [`ItemIndex`].
pub const ITEMS_SCHEMA: &str = "splinter.items/1";

/// Where a collection product keeps its index.
pub const ITEMS_FILE: &str = "items.json";

/// The longest item id, in bytes (a file name on every supported system).
const MAX_ID: usize = 255;

/// One item of a collection.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Item {
    /// Its id, stable across runs: one path component.
    pub id: String,
    /// Where it is, relative to the product's root.
    pub path: String,
}

/// An item an operation refused to produce anything for.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Rejection {
    /// The item's id.
    pub id: String,
    /// Why, in the operation's words.
    pub reason: String,
}

/// The items of a collection product.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ItemIndex {
    /// Always [`ITEMS_SCHEMA`].
    pub schema: String,
    /// The items, in id order.
    pub items: Vec<Item>,
    /// The items rejected, in id order.
    #[serde(default)]
    pub rejected: Vec<Rejection>,
}

/// Why an index is refused.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ItemsError {
    /// An id that is not one path component.
    #[error("item id {0:?} is not one plain path component")]
    BadId(String),
    /// A path that leaves the product.
    #[error("item {id}: path {path:?} is not a plain relative path")]
    BadPath {
        /// The item.
        id: String,
        /// Its path.
        path: String,
    },
    /// An id listed twice (as an item or a rejection).
    #[error("item {0} is listed twice")]
    Duplicate(String),
    /// Another schema.
    #[error("not a {ITEMS_SCHEMA} index: {0}")]
    Schema(String),
}

/// Whether `id` can name an item: one path component, not hidden, no
/// control characters.
#[must_use]
pub fn item_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= MAX_ID
        && !id.starts_with('.')
        && !id.contains(['/', '\\'])
        && !id.chars().any(char::is_control)
}

fn plain_path(path: &str) -> bool {
    !path.is_empty() && !path.starts_with('/') && path.split('/').all(item_id)
}

impl ItemIndex {
    /// The index of `items` and `rejected`, in id order.
    ///
    /// # Errors
    /// A bad id or path, or an id listed twice.
    pub fn new(mut items: Vec<Item>, mut rejected: Vec<Rejection>) -> Result<Self, ItemsError> {
        items.sort_by(|a, b| a.id.cmp(&b.id));
        rejected.sort_by(|a, b| a.id.cmp(&b.id));
        let index = Self {
            schema: ITEMS_SCHEMA.into(),
            items,
            rejected,
        };
        index.validate()?;
        Ok(index)
    }

    /// Check an index read from a product.
    ///
    /// # Errors
    /// As [`Self::new`], or another schema, or items out of id order.
    pub fn validate(&self) -> Result<(), ItemsError> {
        if self.schema != ITEMS_SCHEMA {
            return Err(ItemsError::Schema(self.schema.clone()));
        }
        let mut seen = BTreeSet::new();
        let ids = self
            .items
            .iter()
            .map(|i| &i.id)
            .chain(self.rejected.iter().map(|r| &r.id));
        for id in ids {
            if !item_id(id) {
                return Err(ItemsError::BadId(id.clone()));
            }
            if !seen.insert(id) {
                return Err(ItemsError::Duplicate(id.clone()));
            }
        }
        for item in &self.items {
            if !plain_path(&item.path) {
                return Err(ItemsError::BadPath {
                    id: item.id.clone(),
                    path: item.path.clone(),
                });
            }
        }
        let ordered = |ids: Vec<&String>| ids.windows(2).all(|w| w[0] < w[1]);
        if !ordered(self.items.iter().map(|i| &i.id).collect())
            || !ordered(self.rejected.iter().map(|r| &r.id).collect())
        {
            return Err(ItemsError::Schema("items are not in id order".into()));
        }
        Ok(())
    }
}
