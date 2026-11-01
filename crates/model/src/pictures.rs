// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data for models that read images,
// for its clients. If your team needs expertise in fine-tuning on
// conversations with pictures, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Chat datasets whose records carry pictures: finding them, and checking
//! them where no base is chosen yet and where one is.

use std::path::Path;

use data::chat::{ChatSample, ImageBytes, ImageReader};
use splinter_data::images::MAX_IMAGES_PER_RECORD;

use crate::error::PolicyError;
use crate::train::{validate_dataset, validate_dataset_for, DatasetSummary};

/// Reads the pictures of a record without a base to read them with: only
/// that each is an inline `data:` URL, at most
/// [`MAX_IMAGES_PER_RECORD`] to a record.
pub(crate) struct InlinePictures;

impl ImageReader for InlinePictures {
    fn image(&self, url: &str) -> Result<(ImageBytes, String), String> {
        if url.starts_with("data:") {
            Ok((ImageBytes::from(Vec::new()), "<image>".to_owned()))
        } else {
            Err("an image part must be an inline data: URL".to_owned())
        }
    }
    fn check_text(&self, _text: &str) -> Result<(), String> {
        Ok(())
    }
    fn placeholder_token(&self) -> Option<u32> {
        None
    }
    fn check_record(&self, media: &[ImageBytes]) -> Result<(), String> {
        if media.len() > MAX_IMAGES_PER_RECORD {
            return Err(format!(
                "{} images in one record, at most {MAX_IMAGES_PER_RECORD} are read",
                media.len()
            ));
        }
        Ok(())
    }
}

/// `dataset` as the trainer of a base that reads pictures parses it.
pub(crate) fn validate_pictured_dataset(dataset: &Path) -> Result<(), String> {
    let samples =
        ChatSample::from_jsonl_with(dataset, &InlinePictures).map_err(|e| e.to_string())?;
    if samples.iter().all(|s| s.messages.iter().all(|m| !m.train)) {
        return Err("no message is supervised".into());
    }
    Ok(())
}

/// Whether `dataset` holds a record with a picture in one of its messages.
/// The writer of a dataset writes a picture as a part of type `image_url`; a
/// text that merely says so is a false positive, which only makes the check
/// that follows stricter.
pub fn holds_images(dataset: &Path) -> std::io::Result<bool> {
    use std::io::BufRead as _;
    for line in std::io::BufReader::new(std::fs::File::open(dataset)?).lines() {
        if line?.contains(r#""type":"image_url""#) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// [`validate_dataset`] for a dataset trained on the base in `model_dir`: a
/// dataset with pictures is read as that base reads them, which refuses
/// them over a base with no vision tower; any other is parsed as before.
pub fn validate_dataset_on(
    dataset: &Path,
    model_dir: &Path,
) -> Result<DatasetSummary, PolicyError> {
    let pictured = holds_images(dataset).map_err(|e| PolicyError::Dataset {
        path: dataset.to_path_buf(),
        reason: e.to_string(),
    })?;
    if pictured {
        validate_dataset_for(dataset, model_dir)
    } else {
        validate_dataset(dataset)
    }
}
