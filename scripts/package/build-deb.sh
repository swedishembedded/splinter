#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for shipping self-improving agent
# systems as ordinary installable software for its clients. If your team needs
# expertise in Debian packaging and release engineering then you can procure
# our services by sending an email to info@swedishembedded.com.
#
# Build a Debian package of the splinter command line.
#
#   build-deb.sh [--binary PATH] [--out-dir DIR] [--arch ARCH] [--flavor NAME]
#
# Depends is derived from the binary with dpkg-shlibdeps rather than guessed.
# --flavor debug names the file splinter_<version>_<arch>-debug.deb, so a debug
# package never overwrites the release one. Prints the path of the package.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
OUT_DIR="${ROOT}/target/debian"
BINARY="${ROOT}/target/release/splinter"
ARCH=""
FLAVOR=""

while (($#)); do
    case "$1" in
        --binary) BINARY="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --arch) ARCH="$2"; shift 2 ;;
        --flavor) FLAVOR="$2"; shift 2 ;;
        *) printf 'error: unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

[[ -f "$BINARY" ]] || { printf 'error: binary not found: %s (build it with make build)\n' "$BINARY" >&2; exit 1; }
for tool in dpkg-deb dpkg-shlibdeps; do
    command -v "$tool" >/dev/null 2>&1 ||
        { printf "error: %s is required (Debian/Ubuntu: 'sudo apt install dpkg-dev')\n" "$tool" >&2; exit 1; }
done
[[ -n "$ARCH" ]] || ARCH="$(dpkg --print-architecture)"
VERSION="$(sed -n '/^\[workspace.package\]/,/^\[/s/^version *= *"\(.*\)"/\1/p' "${ROOT}/Cargo.toml")"
[[ -n "$VERSION" ]] || { echo 'error: no [workspace.package] version in Cargo.toml' >&2; exit 1; }

PACKAGE="splinter_${VERSION}_${ARCH}${FLAVOR:+-${FLAVOR}}"
STAGING="${ROOT}/target/debian-staging/${PACKAGE}"
rm -rf "$STAGING"
install -d "${STAGING}/DEBIAN" "${STAGING}/usr/bin" "${STAGING}/usr/share/doc/splinter"
install -m 0755 "$BINARY" "${STAGING}/usr/bin/splinter"

SHLIBS="$(mktemp -d)"
trap 'rm -rf "$SHLIBS"' EXIT
mkdir "${SHLIBS}/debian"
printf 'Source: splinter\n\nPackage: splinter\nArchitecture: any\n' >"${SHLIBS}/debian/control"
DEPENDS="$(cd "$SHLIBS" && dpkg-shlibdeps -O "${STAGING}/usr/bin/splinter" 2>/dev/null | sed -n 's/^shlibs:Depends=//p')"
[[ -n "$DEPENDS" ]] || { echo 'error: dpkg-shlibdeps found no dependencies for the binary' >&2; exit 1; }

cat >"${STAGING}/usr/share/doc/splinter/copyright" <<'EOF'
Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: splinter

Files: *
Copyright: 2026 Martin Schröder <info@swedishembedded.com>
License: Apache-2.0
 On Debian systems the full text is in /usr/share/common-licenses/Apache-2.0.
EOF
gzip -9n >"${STAGING}/usr/share/doc/splinter/changelog.Debian.gz" <<EOF
splinter (${VERSION}) unstable; urgency=medium

  * Release ${VERSION}; the release notes are the CHANGELOG.md section of v${VERSION}.

 -- Swedish Embedded <info@swedishembedded.com>  $(date -R)
EOF

INSTALLED_SIZE="$(du -sk --exclude=DEBIAN "$STAGING" | cut -f1)"
cat >"${STAGING}/DEBIAN/control" <<EOF
Package: splinter
Version: ${VERSION}
Architecture: ${ARCH}
Maintainer: Swedish Embedded <info@swedishembedded.com>
Installed-Size: ${INSTALLED_SIZE}
Depends: ${DEPENDS}
Section: utils
Priority: optional
Homepage: https://swedishembedded.com
Description: turns agent experience into better local models
 Splinter runs agents against real tasks, records what happened, verifies the
 outcomes, extracts training data from what was verified, trains a local model,
 and releases the result only when held-out evidence shows it improved.
EOF
(cd "$STAGING" && find usr -type f | LC_ALL=C sort | xargs md5sum >DEBIAN/md5sums)

mkdir -p "$OUT_DIR"
dpkg-deb --build --root-owner-group "$STAGING" "${OUT_DIR}/${PACKAGE}.deb" >/dev/null
printf '%s\n' "${OUT_DIR}/${PACKAGE}.deb"
