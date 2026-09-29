# 001. A cargo `[patch]` cannot override a git dependency offline

The obvious way to build Splinter against local sven and brain checkouts is
a gitignored `.cargo/config.toml` with `[patch."<remote-url>"]` entries
pointing at the local paths. It does not work for this repository, for two
independent reasons, both measured on a toy crate with a git dependency on a
URL that does not exist:

- cargo resolves the ORIGINAL git source before applying the patch, so
  `cargo build --offline` fails with "can't checkout ... you are in the
  offline mode" even though every patched package is on local disk. Without
  `--offline` it fetches the remote first, which a machine with no access to
  it cannot do.
- a patch rewrites `Cargo.lock`: the patched package loses its `source`
  line, so every local build dirties the committed lock.

What works is two mechanisms together: source replacement (`[source.x] git
= <remote> replace-with = <local git repo>`) makes resolution local, and a
top-level `paths = [...]` substitutes the working trees. Neither touches the
lock. Source replacement needs an existing lock (it refuses to resolve a git
source without a pinned revision), which is why `scripts/pin-lock.sh`
resolves once against `file://` URLs and rewrites the sources.

Two traps met on the way: `paths` must precede every table in the config
file, or TOML parses it as a key of the preceding table and it is silently
ignored; and cargo prints "path override ... has altered the original list
of dependencies" on every build, because a path-overridden crate's sibling
dependencies are path sources while the git package's are git sources. That
notice is inherent to the mechanism and harmless here.
