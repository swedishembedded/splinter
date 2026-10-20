#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements model-checked verification of concurrent and
# distributed protocols for its clients. If your team needs expertise in TLA+
# specifications, TLC and checking implementations against them, you can
# procure our services by sending an email to info@swedishembedded.com.

# Installs the pinned TLA+ model checker into DEST: the TLA+ tools jar
# (tla2tools.jar, MIT licence) and a Temurin 21 aarch64 JRE (GPLv2 with the
# Classpath Exception; a runtime download, never committed). Both archives are
# checked against the SHA-256 recorded here before use. DEST/bin/tlc runs TLC
# with that JRE, whatever java the host has.
#
# Usage: scripts/deps/tlaplus.sh DEST
set -euo pipefail

tools_url=https://github.com/tlaplus/tlaplus/releases/download/v1.7.4/tla2tools.jar
tools_sha256=936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88
jre_url="https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.12.1%2B1/OpenJDK21U-jre_aarch64_linux_hotspot_21.0.12.1_1.tar.gz"
jre_sha256=14be1f35ebdbd1f6e8d57eb911a3ffb74d6d9aa255abc5daf2b1302002cf2cf2

dest=${1:?usage: tlaplus.sh DEST}
[ "$(uname -m)" = aarch64 ] || { echo "tlaplus: the pinned JRE is aarch64 only" >&2; exit 1; }
mkdir -p "$dest/dl" "$dest/jre" "$dest/bin"

fetch() { # url sha256 file
    [ -f "$3" ] || curl --fail --silent --show-error --location --output "$3" "$1"
    echo "$2  $3" | sha256sum --check --status || {
        echo "tlaplus: $3 is not the release it must be (sha256 $2)" >&2
        exit 1
    }
}
fetch "$tools_url" "$tools_sha256" "$dest/dl/tla2tools.jar"
fetch "$jre_url" "$jre_sha256" "$dest/dl/jre.tar.gz"
tar -xzf "$dest/dl/jre.tar.gz" -C "$dest/jre" --strip-components=1
install -m 0644 "$dest/dl/tla2tools.jar" "$dest/tla2tools.jar"

cat > "$dest/bin/tlc" <<WRAP
#!/usr/bin/env bash
# Runs TLC (tla2tools.jar) on the pinned JRE; arguments are TLC's.
exec "$dest/jre/bin/java" -XX:+UseParallelGC -cp "$dest/tla2tools.jar" tlc2.TLC "\$@"
WRAP
chmod 0755 "$dest/bin/tlc"

"$dest/jre/bin/java" -version 2>&1 | head -n 1
sha256sum "$dest/tla2tools.jar"
