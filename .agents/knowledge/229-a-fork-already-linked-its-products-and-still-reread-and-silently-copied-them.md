<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 229. A fork linked its products but re-read every byte of them, and fell back to copying without saying so

**Lesson:** `splinter fork` was reported to copy every product (10817 products, 5.2 GB, 15 minutes with four forks at
once on a local XFS disk). The blobs were already hard-linked, but the import hashed every linked blob again (a full
read of the parent's bytes, pointless for the same inode), copied a blob without a word when the link failed, and
`verify_product` hashed a view file that is the blob itself a second time. Check what a "copy" costs from the code
path, not from the report, before changing the storage model.

## Why linking is safe

Blobs are written once, sealed read-only and never rewritten (`seal_blob`, `put_file` write a temporary sibling and
rename); a view is built in a scratch directory and renamed into place. No code opens a product file for writing, so a
child run sharing the parent's inodes cannot change it. The one way to alter a shared file is a process that chmods and
edits it in place; `verify_product` and `splinter verify` catch that by re-hashing against the address, which is why a
linked blob is verified when a stage reuses it, not when it is linked.

## What changed

`ProductStore::import` returns `Imported` (blobs linked, blobs copied); a link that fails (another device, a file
system without links) copies the blob, checks the digest while copying, and is counted. `run::fork` returns `Forked`,
and `splinter fork` / `splinter eval` print the counts and a warning when anything was copied. `verify_product` does
not hash a view file that shares the blob's inode.

## Measured (2026-11-26, 5000 products of two files, 464 MB, page cache warm, local XFS)

Import loop 5.7 s (3 runs: 5.9, 6.7, 5.7) before, 4.7 s (4.8, 4.6, 4.7) after. What is left is metadata: three
fsyncs and a directory sync per product (pending intent, view manifest, commit point) and 6 file creations; it grows
with the product count, not the bytes. Reflink is not used (no gain over a link for read-only blobs). Open: the fork
could batch its commit-point syncs, as a crashed fork is re-run and re-imports idempotently.
