<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 085. A dataset from the 27B teacher taught the host login, the scratch path and the loop tail, under a prompt it was not served with

The trace audit of dataset blake3:d246dbb5 (80 packed records from the 27B teacher) found, beyond the exposure
rule (081), four things in the records themselves:

* 5 records held the host's login name (an `ls -la` owner and group) and 8 the harness's `/tmp/.tmpXXXXXX/tree`
  path (tracebacks, `pwd`). Neither is about the task; a model memorises a name and a random directory. The
  projection now names the recorded tree relative to itself and writes `user` for the owner and group of a long
  listing and for a home directory (`scrub`). The solve's environment cannot hide them without changing what the
  commands see (a user namespace that maps the host uid to another uid changes `ls` and the permission checks
  tests rely on); the shell environment already carries no `USER`/`HOME`.
* 3 admitted records trained 4-5 repeated actions after the fix was already made (an identical "full sanity
  check" run five times before the repetition stop). An action that repeats a call of its trajectory with the
  same result is context, not an action to learn (`repeated_action`); an identical call with another result is
  progress and stays.
* The records opened with the chat prompt ("answer directly and concisely") although the evaluation arms run the
  coding-agent prompt: the trajectory holds no system step, only a solve profile with the prompt's digest, and
  `dataset build` knew the prompt only through `--system-prompt`. The sft-step view now opens a record with the
  prompt the profile records and the dataset's manifest names it, so a release trained on it is asked under it. A
  digest this build holds no text of leaves the run out; two prompts in one dataset are refused until
  `--system-prompt` writes them all under one.

The fixtures of the existing tests repeated one edit three times; with the rule they were three records becoming
one, which is the defect in miniature.
