<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 215. At about a hundred records, the base's own verified passes teach as much as teacher passes, and out-of-support teacher passes hurt

**Lesson:** with the record count matched within each fold (89-118 records), Qwen3-4B students trained on (S) the
base's own verified passes, (T-same) 27B teacher passes on the same tasks, or (T-other) teacher passes on tasks the base
never solved were examined on 213 held-out vim tasks over 14 rungs under a token and turn budget (generic-learning
E10, 2026-11-22). The results:
- T-same minus S was +0.008, family interval [-0.038, +0.051]: no content effect at this dose.
- T-other minus T-same was -0.049 [-0.076, -0.017], Holm p 0.023: data from outside the base's support hurt.
- S minus base was +0.036 [+0.023, +0.049]: a small teacher-free gain, with +0.05 ruled out at this record count.
- E1's students, trained on 343-403 teacher records, beat every ~110-record arm by 0.04-0.10. An extrapolation fold
  (train rungs 1-9, examine 10-14) showed no transfer from any source.

**How to apply:** at a few hundred records, the amount of verified in-support data is the lever, not the teacher's
content or tasks beyond the student's reach. Teacher-free improvement should raise the number of distinct own verified
passes (more attempts, more tasks, self-distillation of a stronger student), not chase tasks the student cannot solve.
