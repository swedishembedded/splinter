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
cargo build --release -p splinter-voice-loop
V="target/release/voice-loop --backend cuda"

# Needs the models in brain's store:
#   brain pull Qwen/Qwen3-TTS-12Hz-0.6B-Base
#   brain pull nvidia/nemotron-3.5-asr-streaming-0.6b
$V speak --text "Good morning." --out question.wav
$V roundtrip --sentences sentences.txt --report roundtrip.json
$V turn --in question.wav --out answer.wav --persona "Samuel Adams" \
    --base ~/.local/share/brain/models/Qwen/Qwen3-8B --report turn.json

# A directory of recordings, each sentence spoken as the model writes it:
$V turns --stream --recordings questions/ --out-dir answers/ \
    --base ~/.local/share/brain/models/Qwen/Qwen3-8B --report turns.json

# Talk to a sven agent that answers as the persona (its words are spoken as
# they are written; with --workspace it also has sven's coding tools there):
$V agent --recordings questions/ --out-dir answers/ \
    --base ~/.local/share/brain/models/Qwen/Qwen3-8B --report agent.json

# Teach the persona how to say a word, then hear it:
$V teach --said "Pronounce Jefferson as Jeff-er-son." --lessons lessons.jsonl
$V speak --lessons lessons.jsonl --text "Jefferson wrote it." --out said.wav
```

`--backend` says how the hardware is driven: `cuda`, `vulkan`, `wgpu` or
`cpu`. Without it brain uses the backend its probe prefers, which on a CUDA
host is not CUDA, and the run is several times slower. The `adapter:` line brain
prints names the card whichever backend runs, so it does not tell you; the
`first_audio_seconds` of a `turns --stream` report does.

## Teaching it to speak

`teach` hears a directive and keeps what it teaches as lessons, one JSON
object per line (`core::speech_lesson`). `--directive` is a recording of the
user, `--said` the same typed.

* "Pronounce X as Y" makes a lexicon entry: `speak --lessons` says X as Y. A
  respelling said aloud is lost to the recogniser ("Jeff-er-son" is heard as
  "Jefferson"), so teach it typed.
* "Talk as follows: ..." needs the example as its own recording, spoken as the
  next turn and passed with `--example`; it makes in-context, supervised and
  preference lessons that name the recording by digest. Only the lexicon is
  applied today; the others are kept for training that is not built (see the
  speech roadmap).

Every lesson carries the portrayal of the voice it applies to.

## What is measured

* `roundtrip`: each sentence is spoken, heard again by a recognizer that did
  not make it, and scored by word error rate (case and punctuation ignored).
  The corpus rate is total errors over total words, so a long sentence weighs
  more than a short one; it is absent, not zero, when no words were spoken.
  `mostly_lost` counts sentences that lost more than half their words.
* `turn`: the three stages (recognise, answer, speak) are timed on their own.
  Silence is not answered: a clip with no recognisable speech, or an answer
  with no words, ends the turn with an error naming the stage.
* `turns --stream`: speaking runs beside the model, a sentence at a time, and
  the first piece goes at a clause end after four words (or ten). The report
  holds, per turn, seconds until the first word was written
  (`first_text_seconds`), until the first piece was handed to be spoken
  (`first_piece_seconds`) and until the voice first made a sound
  (`first_audio_seconds`), each with its spread.

Synthesis is seeded (`--seed`), because the same sentence is spoken in very
different lengths from different seeds and a short rendering can lose words
before recognition hears them.

## Layout

| File | What |
|---|---|
| `src/lib.rs` | the reports and the sentence-file reader |
| `src/main.rs` | the command line; `voice-loop --help` lists every command |
| `src/agent.rs` | `agent`: the recordings as turns of one conversation with a sven agent |
| `src/batch.rs` | `turns`, `turns --stream` and `listen-turns` |
| `src/teach.rs` | `teach` and `speak` with a lexicon |
| `src/ingress.rs`, `src/spoken.rs`, `src/bundle.rs` | the spoken sets, the listening projector and the bundle |
| `crates/model/src/speech/` | the cascade itself: recognizer and synthesizer on brain, one turn, the round trip, the cascade listener |
