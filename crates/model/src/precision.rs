// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of language models
// for its clients. If your team needs expertise in training adapters on large
// bases with limited GPU memory, you can procure our services by sending an
// email to info@swedishembedded.com.

//! How a frozen base is held in memory while an adapter trains on it.

/// How the frozen base is held while an adapter trains; the adapter and its
/// optimizer stay fp32 whatever the base is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BasePrecision {
    /// The checkpoint's fp32 weights: brain's default.
    #[default]
    Fp32,
    /// bf16, half the bytes of fp32: what lets a 7B base train on one 24 GiB
    /// card.
    Bf16,
    /// int8 with a scale per 32 weights, about a quarter of fp32: room for an
    /// 8B base and its activations on one 24 GiB card.
    Int8,
    /// NF4 (4-bit normal-float blocks), about an eighth of fp32: the tier
    /// that lets the 27B hybrid base train on a card shared with a server.
    /// A `qwen35` base only.
    Nf4,
    /// Block-scaled E4M3 fp8, half of bf16: the native format of an fp8
    /// checkpoint, held without requantisation loss. brain refuses it for a
    /// base whose trainer does not hold fp8.
    Fp8,
}

impl std::str::FromStr for BasePrecision {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        match text {
            "fp32" => Ok(Self::Fp32),
            "bf16" => Ok(Self::Bf16),
            "int8" => Ok(Self::Int8),
            "nf4" => Ok(Self::Nf4),
            "fp8" => Ok(Self::Fp8),
            other => Err(format!(
                "{other:?} is not one of fp32, bf16, int8, nf4, fp8"
            )),
        }
    }
}

impl BasePrecision {
    /// The name brain's `ChatFineTune::base_format` takes for this tier, the
    /// one mapping every training path uses. `None` for [`Self::Fp32`]: brain
    /// then picks the tier the device and the base allow, as it always has
    /// when no tier is asked for. brain refuses a format the base cannot hold.
    #[must_use]
    pub fn brain_format(self) -> Option<&'static str> {
        match self {
            Self::Fp32 => None,
            Self::Bf16 => Some("bf16"),
            Self::Int8 => Some("int8"),
            Self::Nf4 => Some("nf4"),
            Self::Fp8 => Some("fp8"),
        }
    }

    /// [`Self::Bf16`] when `on`, else fp32: what a `--bf16-base` flag asks.
    #[must_use]
    pub fn bf16_if(on: bool) -> Self {
        if on {
            Self::Bf16
        } else {
            Self::Fp32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BasePrecision;

    #[test]
    fn every_tier_maps_to_the_name_brain_takes() {
        let tiers = [
            ("fp32", BasePrecision::Fp32, None),
            ("bf16", BasePrecision::Bf16, Some("bf16")),
            ("int8", BasePrecision::Int8, Some("int8")),
            ("nf4", BasePrecision::Nf4, Some("nf4")),
            ("fp8", BasePrecision::Fp8, Some("fp8")),
        ];
        for (text, tier, format) in tiers {
            assert_eq!(text.parse::<BasePrecision>(), Ok(tier));
            assert_eq!(tier.brain_format(), format);
        }
        assert!("fp4".parse::<BasePrecision>().is_err());
    }
}
