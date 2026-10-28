// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The stand-in for the `brain` binary the release specs run the serve check
//! against; see [`super::gate`]'s module documentation for what it does and
//! does not test.

use std::path::{Path, PathBuf};

use super::Scratch;

/// How the `brain` binary behaves in a spec.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Brain {
    /// There is none.
    Missing,
    /// The stand-in, reporting the adapter's real digest.
    Honest,
    /// The stand-in, reporting a digest that is not the adapter's.
    WrongDigest,
    /// The stand-in, reporting the real digest and answering in other
    /// words than the candidate in-process, graded the same.
    Divergent,
    /// The stand-in, reporting the real digest and answering something
    /// else than the candidate in-process, whatever it is asked.
    Different,
    /// The stand-in, reporting the real digest and then dying with an
    /// out-of-memory message when it is first asked something.
    Crashing,
}

/// The file standing for the device being held: while it exists, the
/// `brain` stand-in in `scratch` refuses to start.
pub fn device_lock(scratch: &Scratch) -> PathBuf {
    scratch.0.join(DEVICE_LOCK)
}

/// The file the `brain` stand-in appends the system turns of the requests it
/// answers to, one JSON array a line.
pub fn request_log(scratch: &Scratch) -> PathBuf {
    scratch.0.join(REQUEST_LOG)
}

const REQUEST_LOG: &str = "requests.log";

/// The file the `brain` stand-in appends a line to for each request it
/// answers: the `--max-batch` it was started with and how many requests
/// it was answering at that moment, this one included. While the file `pause`
/// exists in the scratch directory each request takes a second to answer, so
/// requests asked together overlap.
pub fn slots_log(scratch: &Scratch) -> PathBuf {
    scratch.0.join(SLOTS_LOG)
}

const SLOTS_LOG: &str = "slots.log";

/// [`device_lock`]'s name in the scratch directory.
const DEVICE_LOCK: &str = "device.lock";

/// Writes the `brain` stand-in into `dir`; see the module documentation.
pub fn fake_brain(dir: &Path, brain: Brain) -> PathBuf {
    let path = dir.join("brain");
    let python = |yes: bool| if yes { "True" } else { "False" };
    let script = FAKE_BRAIN
        .replace("@WRONG@", python(brain == Brain::WrongDigest))
        .replace("@DIVERGENT@", python(brain == Brain::Divergent))
        .replace("@DIFFERENT@", python(brain == Brain::Different))
        .replace("@CRASHING@", python(brain == Brain::Crashing))
        .replace("@DEVICE@", &dir.join(DEVICE_LOCK).display().to_string())
        .replace("@LOG@", &dir.join(REQUEST_LOG).display().to_string())
        .replace("@SLOTS@", &dir.join(SLOTS_LOG).display().to_string())
        .replace("@PAUSE@", &dir.join("pause").display().to_string());
    std::fs::write(&path, script).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    path
}

const FAKE_BRAIN: &str = r#"#!/usr/bin/env python3
# A test double of `brain serve`: see the fixtures' module documentation.
import hashlib, http.server, json, os, re, sys, threading, time

if os.path.exists("@DEVICE@"):
    print("the device is held by another process", file=sys.stderr, flush=True)
    sys.exit(1)
args = sys.argv[1:]
assert args[0] == "serve", args
opts = dict(zip(args[1::2], args[2::2]))

# Like brain, an adapter that corrects the MLP linears is refused unless the
# server is told to bank them: the trainer's LoRA targets every linear.
if ("--adapter" in opts or "--adapter-manifest" in opts) and opts.get("--adapter-targets") != "all":
    print("brain serve: adapter corrects down, which this server's --adapter-targets (wq,wk,wv,wo) does not; widen the flag to serve it", file=sys.stderr, flush=True)
    sys.exit(1)

# A consolidated checkpoint is served with no adapter: the weights file of
# the base is the card, and nothing is announced.
# brain reads no checkpoint from the environment: the launch names it.
assert "--checkpoint" in opts, args
weights = opts["--checkpoint"]
if os.path.isdir(weights):
    weights = os.path.join(weights, "model.safetensors")
if "--adapter-manifest" in opts:
    # A release manifest names the adapter the server follows: a relative
    # path is relative to the manifest, as brain reads it.
    manifest = json.load(open(opts["--adapter-manifest"]))
    followed = os.path.join(os.path.dirname(opts["--adapter-manifest"]), manifest["adapter"])
    if manifest["digest"] != "sha256:" + hashlib.sha256(open(followed, "rb").read()).hexdigest():
        print("manifest names a file that is not the digest it states", file=sys.stderr, flush=True)
        sys.exit(1)
    opts["--adapter"] = followed
adapter = open(opts.get("--adapter") or weights, "rb").read()
digest = "sha256:" + hashlib.sha256(adapter).hexdigest()
if @WRONG@:
    digest = "sha256:" + "0" * 64
card = json.loads(adapter)
agent = card.get("agent")
knows = card.get("knows", [])
answers = card.get("answers", {})
json.dump({"openai": "sk-fake"}, open(opts["--api-keys-out"], "w"))
if "--adapter" in opts:
    print("brain serve: brain/qwen3 adapter=local/test:splinter:candidate:candidate-1 digest=" + digest,
          file=sys.stderr, flush=True)

def reply(text):
    for key, said in answers.items():
        if key in text:
            return said + "\n"
    m = re.search(r"What is the (\S+) code number (\d+)\?", text)
    if m and m.group(1) in knows:
        answer = m.group(1) + "-" + m.group(2)
    else:
        answer = "I do not know."
    if @DIFFERENT@:
        return "The weather is mild in spring."
    if @DIVERGENT@:
        return answer.upper().rstrip(".") + "."
    return answer + "\n"

def mend(body):
    """What an adapter that mends modules replies to a conversation about one:
    the module restored by a tool call, then "done"; it does nothing when
    it is not asked about a module, when it was asked to sample, or when the
    stand-in is to answer something else than the candidate in-process."""
    messages = body["messages"]
    if any(m.get("role") == "tool" for m in messages):
        return "done", None
    first = next((text_of(m.get("content")) for m in messages if m.get("role") == "user"), "")
    asked = re.search(r"- tier(\d+)", first)
    if body.get("temperature") != 0:
        return "sampled", None
    if not asked or @DIFFERENT@ or agent != "mender":
        return "done", None
    n = asked.group(1)
    text = 'def tier%s(units):\n    """The price tier of an order."""\n    if units > 10:\n        return 20 + %s\n    return 5\n' % (n, n)
    call = {"id": "call-1", "type": "function", "function": {"name": "write_file", "arguments": json.dumps({"path": "shop/tier%s.py" % n, "text": text, "append": False})}}
    return None, call

def text_of(content):
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""

answering = 0
answering_lock = threading.Lock()

class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        global answering
        with answering_lock:
            answering += 1
            with open("@SLOTS@", "a") as log:
                log.write("%s %d\n" % (opts.get("--max-batch", "default"), answering))
        try:
            if os.path.exists("@PAUSE@"):
                time.sleep(1.0)
            self.answer()
        finally:
            with answering_lock:
                answering -= 1

    def answer(self):
        if @CRASHING@:
            print("wgpu error: Out of Memory", file=sys.stderr, flush=True)
            os._exit(1)
        if self.headers.get("Authorization") != "Bearer sk-fake":
            self.send_response(401); self.end_headers(); return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        users = [text_of(m.get("content")) for m in body["messages"] if m.get("role") == "user"]
        systems = [text_of(m.get("content")) for m in body["messages"] if m.get("role") == "system"]
        with open("@LOG@", "a") as log:
            log.write(json.dumps(systems) + "\n")
        # Sampled at any temperature but zero, a real model's answer is a
        # draw: the double answers only when asked to decode greedily.
        tool = None
        if agent is not None and body.get("tools"):
            answer, tool = mend(body)
        else:
            answer = reply(users[-1] if users else "") if body.get("temperature") == 0 else "sampled"
        self.send_response(200)
        if body.get("stream"):
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            if tool is not None:
                chunks = (
                    {"choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [dict(tool, index=0)]}, "finish_reason": None}]},
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
                )
            else:
                chunks = (
                    {"choices": [{"index": 0, "delta": {"role": "assistant", "content": answer}, "finish_reason": None}]},
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                )
            for chunk in chunks:
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            message = {"role": "assistant", "content": answer}
            if tool is not None:
                message = {"role": "assistant", "content": None, "tool_calls": [tool]}
            out = json.dumps({"choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool else "stop"}]}).encode()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    def log_message(self, *args):
        pass

host, port = opts["--openai"].rsplit(":", 1)
class Server(http.server.ThreadingHTTPServer):
    # Many requests asked at once connect at once; the default backlog of five
    # would drop the rest and make the client wait a second to retry.
    request_queue_size = 128

server = Server((host, int(port)), Handler)
open(opts["--ready-file"], "w").close()
server.serve_forever()
"#;
