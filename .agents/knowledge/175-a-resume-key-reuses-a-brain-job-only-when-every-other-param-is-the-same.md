<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 175. A resume key reuses a brain job only when every other param is the same

**Lesson:** brain identifies a resumable run by every param but `resume_key` and
`checkpoint_every`, and refuses a key whose record names another run. A client that resubmits
after a crash must therefore rebuild the request byte for byte, including the names it derives
from its own working directories.

Found 2026-11-18 building `splinter campaign`'s train stage. `train` names each candidate with a
fresh id (`candidate-<ulid>`), and the fine-tune's `adapter_id` param is derived from the
candidate's directory. A campaign killed mid-training and resumed would send the same key with a
new `adapter_id`: brain refuses it ("one key is one run"), and without the key it trains the
candidate a second time. `train::train_identified` names the candidate `candidate-<identity>`,
the same identity as the key, and reuses a kept candidate of exactly that id.
`campaign::generations::a_campaign_killed_mid_rollout_and_mid_training_resumes_to_the_same_datasets_and_ledger`
asserts brain trained each of three candidates once across two kills, and
`served_train::a_fine_tune_under_a_resume_key_brain_completed_is_answered_from_its_record_not_trained_again`
that a changed param under the same key is refused.
