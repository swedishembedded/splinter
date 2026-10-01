// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Questions about the graph that the structure answers directly: families
//! of attempts, and the alternatives tried at one decision.

mod counterfactual;
mod families;

pub use counterfactual::CounterfactualView;
pub use families::{AttemptView, FamilyView};
