# voice-loop: a persona you can talk to, and what each stage of the loop costs and loses

Can a persona be talked to and talk back? The reference loop is a cascade:
recognise the listener, let the persona answer in text, speak the answer. It
is the baseline a speech-native model is later measured against, and it
makes the spoken examples that model learns from. It is written on Splinter's
SDK alone.

The voice is a synthetic theatrical portrayal. No recording of Samuel Adams
or Thomas Jefferson exists, so no voice can be theirs. Every report carries
the portrayal the speaker declares, and a speaker cannot be built without one.

## Run it

```bash
source ~/project/shared/containers/build-env.sh   # a GPU the build can see
export BRAIN_BACKEND=cuda
cargo build --release -p splinter-voice-loop
V=target/release/voice-loop

# Needs the models in brain's store:
#   brain pull Qwen/Qwen3-TTS-12Hz-0.6B-Base
#   brain pull nvidia/nemotron-3.5-asr-streaming-0.6b
$V speak --text "Good morning." --out question.wav
$V roundtrip --sentences sentences.txt --report roundtrip.json
$V turn --in question.wav --out answer.wav --persona "Samuel Adams" \
    --base ~/.local/share/brain/models/Qwen/Qwen3-8B --report turn.json
```

Check that the `adapter:` line brain prints names a GPU. A container without
a Vulkan driver file makes brain run on the CPU without saying so.

## What is measured

* `roundtrip`: each sentence is spoken, heard again by a recognizer that did
  not make it, and scored by word error rate (case and punctuation ignored).
  The corpus rate is total errors over total words, so a long sentence weighs
  more than a short one; it is absent, not zero, when no words were spoken.
  `mostly_lost` counts sentences that lost more than half their words.
* `turn`: the three stages (recognise, answer, speak) are timed on their own.
  Silence is not answered: a clip with no recognisable speech, or an answer
  with no words, ends the turn with an error naming the stage.

Synthesis is seeded (`--seed`), because the same sentence is spoken in very
different lengths from different seeds and a short rendering can lose words
before recognition hears them.

## Layout

| File | What |
|---|---|
| `src/lib.rs` | the reports and the sentence-file reader |
| `src/main.rs` | `speak`, `roundtrip`, `turn` |
| `crates/model/src/speech/` | the cascade itself: recognizer and synthesizer on brain, one turn, the round trip |
