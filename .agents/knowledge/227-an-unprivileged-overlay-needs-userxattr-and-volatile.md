<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 227. An unprivileged overlay needs userxattr and volatile, and sandboxes unpacking one image to disk are disk-bound

**Lesson:** after 225 moved every sandbox's image copy to the local disk, about 500 sandboxes starting at once each
unpacked the same few hundred megabytes of installation, and the disk became the run's bottleneck. The shared layers
are now unpacked once per node and every sandbox mounts an overlay over them in its own user and mount namespace
(`userns_keeper.py`). Two mount options are not optional there:

- `userxattr`: an overlay mounted in a user namespace cannot write `trusted.overlay.*`, so without it removing a
  directory of the lower tree (an opaque directory, `rm -rf` of an installed directory) fails with EIO. With it the
  marks live in `user.overlay.*`, which the local XFS supports.
- `volatile`: an overlay's unmount (an explicit one, or its namespace ending with the keeper) syncs the whole
  filesystem under the upper directory and waits for every dirty page of the node. With a production run keeping
  about 75 GB dirty, `umount` sat in `wb_wait_for_completion` (uninterruptible, so not even SIGKILL ends it) for
  minutes. The upper directory is a sandbox's scratch that is never mounted again, so dropping its syncs loses nothing.

A third defect surfaced in cleanup: the overlay's `work/work` is mode 000, `shutil.rmtree` reports the failed
`os.open` of it to `onexc`, and retrying `os.open(path)` there raises `TypeError`; `userns.remove_tree` now removes
such a directory after giving the owner access.

## Evidence (2026-11-26, one GH200 node, local XFS, a production run sharing the disk)

64 keepers of one task image (a 306 MB shared prefix archive plus a 30 KB task layer) started at once, spawn to
`ready`: unpacking the whole image per sandbox, median 355.5 s and p90 366.8 s; overlay with the shared tree already
unpacked, median 1.68 s and p90 2.21 s; overlay on an empty cache (one keeper unpacks, 63 wait on the lock), median
43.0 s and p90 43.1 s. Removing the 64 sandboxes' copies afterwards took 194 s for the full copies and 2.5 s for the
overlays' upper directories.
