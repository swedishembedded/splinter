# voice-loop: a persona you can talk to, and what each stage of the loop costs and loses

Can a persona be talked to and talk back? The loop is a cascade: recognise
the listener, let the persona answer in text, speak the answer. It is written
on Splinter's SDK alone, and every model in it (recogniser, text model,
synthesizer) is asked of a brain server through its HTTP API:
`POST /v1/audio/transcriptions`, `POST /v1/chat/completions` and
`POST /v1/audio/speech`.

The voice is a synthetic theatrical portrayal. No recording of Samuel Adams
or Thomas Jefferson exists, so no voice can be theirs. Every report carries
the portrayal the speaker declares, and a speaker cannot be built without one.

## Run it

```bash
cargo build --release -p splinter-voice-loop
# A running brain server (or leave the two flags out and pass --brain-bin to
# have one started for the run, serving brain's model store):
V="target/release/voice-loop --brain-endpoint $BRAIN_URL --brain-key-file $BRAIN_KEY_FILE"

# The server must serve the models --asr and --tts name; the defaults are:
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

`--backend` and `--device` are handed to a server the run starts; a server
given with `--brain-endpoint` runs on whatever it was started with. Without
`--backend` brain uses the backend its probe prefers, which on a CUDA host is
not CUDA, and the run is several times slower; the `first_audio_seconds` of a
`turns --stream` report shows it.

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

A speaker carries a seed (`--seed`): a first command that writes to an output
place without `--seed` draws one and prints it, and records it in `voice.json`
in `--out-dir` (for `speak` and `turn`, `NAME.voice.json` beside `--out`);
every later command that writes to the same place reads it back and refuses a
different `--seed`, and the reports carry the speaker with its seed.

The seed reaches brain's `/v1/audio/speech` and chooses the take: the same
seed, model and text give the same clip, and a server that says it drew
another seed is refused. `speak-set --attempts` and `spoken-set --voices`
draw their takes from successive seeds.

A speaker may imitate a recorded voice instead of the model's own: `--voice
FILE` names a few seconds of one person speaking (WAV, Ogg/Opus or
WebM/Opus, at most 30 seconds), `--voice-text` what it says (Qwen3-TTS then
imitates the delivery and the room as well as the timbre; CosyVoice needs it),
and `--portrayal` what the voice declares itself to be wherever it is heard,
which is required because the theatrical label says no recording exists.
Only clone a voice whose owner agreed to it:

```bash
$V speak --voice me.wav --voice-text "what me.wav says" \
    --portrayal "Synthetic speech in my own recorded voice" \
    --text "Good morning." --out morning.wav
```

The recording is named on every run; only the seed is recorded in
`voice.json`. A model that cannot take what is asked (a reference on a model
that does not clone) is refused by brain with the field named.

## Retired

These ran brain in this process and have no route in brain's API, so they
were deleted rather than kept as a path that cannot run:

* `ingress-train` and `listen-turns`: a projector that let a frozen language
  model listen to a recogniser's audio-encoder features, and the listener that
  answered through it. The API serves no audio features, trains no projector
  and runs no chat over projected audio.
* `speech-rl`: the reinforcement trainer that updated the synthesizer from
  scored takes. The API has no synthesizer update.
* `takes`: several takes per sentence in one voice pinned to a reference
  recording, heard by two recognisers with a fixed decode window. The
  transcription route takes no window; the reference voice and the seed are
  `--voice` and `--seed` of `speak` and `speak-set` now.
* `bundle`: content addresses of the parts of a listening persona (its
  projector among them); with the projector gone it named nothing that runs.

`text-score` stays: it scores chat records through the server (`/v1/run`).
`judge` stays: it reads a listening session that is already on disk.

## Layout

| File | What |
|---|---|
| `src/lib.rs` | the reports and the sentence-file reader |
| `src/main.rs` | the command line; `voice-loop --help` lists every command |
| `src/agent.rs` | `agent`: the recordings as turns of one conversation with a sven agent |
| `src/batch.rs` | `turns` and `turns --stream` |
| `src/teach.rs` | `teach` and `speak` with a lexicon |
| `src/teach_eval.rs` | `teach-eval` and `teach-read` |
| `src/speakset.rs`, `src/spoken.rs` | `speak-set`, and the pinned spoken sets of `spoken-set` |
| `src/judge.rs` | `judge` |
| `crates/model/src/speech/` | the cascade itself: recognizer and synthesizer over brain's API, the WAV clip, word error rate, one turn, the round trip, the cascade listener |
