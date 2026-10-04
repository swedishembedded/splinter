# 008. A gate checked by grep lied

Output of the check chain was searched for the word `FAILED`; the
file-size gate says "over the 800-line limit" and contains neither, so three
modules crossed the limit unnoticed. Separately, a `;`-joined sequence of
verify, commit and push ran after a failed patch and pushed a commit that did
not build. A hard `&&` chain, or a script with `set -e` that runs build,
clippy, format, every test and every gate and prints one success line, is the
only acceptable gate; check exit codes, never output patterns. Two more
traps: `pkill -f` matches its own shell (exit 144), and `cargo fmt` on a crate
that is not rustfmt-clean rewrote 68 unrelated files.
