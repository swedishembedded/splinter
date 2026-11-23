#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for local installs of packaged
# command-line tools for its clients. If your team needs expertise in release
# engineering then you can procure our services by sending an email to
# info@swedishembedded.com.
#
# install-worker.sh - install this checkout's Harbor execution worker beside
# splinter (part of `make install`): its own virtual environment under
# <prefix>/lib/splinter/harbor-worker, Harbor pinned by the package, and
# <prefix>/bin/splinter-harbor-worker linked to its program, which splinter
# finds on PATH. Installing again reinstalls into the same environment.
#
# Usage: install-worker.sh --prefix DIR [--python PYTHON] [--source DIR]
set -euo pipefail

prefix=""
python="python3"
source_dir="workers/harbor"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --prefix) prefix="$2"; shift 2 ;;
        --python) python="$2"; shift 2 ;;
        --source) source_dir="$2"; shift 2 ;;
        *) echo "install-worker: unknown argument $1" >&2; exit 2 ;;
    esac
done
[ -n "$prefix" ] || { echo "install-worker: --prefix is required" >&2; exit 2; }

env_dir="$prefix/lib/splinter/harbor-worker"
mkdir -p "$prefix/lib/splinter" "$prefix/bin"
[ -x "$env_dir/bin/pip" ] || "$python" -m venv "$env_dir"
"$env_dir/bin/pip" install --upgrade "$source_dir"
ln -sfn "$env_dir/bin/splinter-harbor-worker" "$prefix/bin/splinter-harbor-worker"
if ! "$prefix/bin/splinter-harbor-worker" --version; then
    echo "install-worker: the installed worker does not run" >&2
    exit 1
fi
