<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 073. The health forecast tests hung because each training built its own Vulkan device

`splinter-health` unit tests hung at 100% CPU for hours under `cargo test
--workspace`, and not every time: the full binary ran in 2 s with 1 to 16 test
threads and stopped completing at 32 or more (default is one thread per core, 144
here). The three tests that train a tiny timeline model (`predict::tests::
a_new_checkup...`, `a_duplicate_checkup...`, `a_malformed_or_mismatched_unit...`)
never finished; the other 15 did.

A stack dump of a hung run (gdb, all threads) showed one test thread in
`vkQueueSubmit` blocked on a pthread mutex inside the NVIDIA driver
(`libnvidia-glsi`), another in `vkDestroyDevice` joining a driver thread, and a
third in the driver's thread-exit destructor. `Horizon::new` makes a wgpu device
per training, so libtest's parallelism created and destroyed several devices
at once. Brain already records the shape (`gpu-core` `testgpu`: many concurrent
devices deadlock the driver on this host).

The fixture of the three tests now takes a process-wide turn that it holds with
the trained model until the test ends (model dropped first), so one device
is live at a time. Twelve runs at 144 test threads all finished; before the
change the same command hung. The turn is in the tests, not in production code:
production training runs one device at a time per process.

How to avoid it: a test that trains or serves on a device is not independent of
its neighbours; serialize the device lifetime (or share one device) in the test
binary, and when a test "hangs", take a thread dump before reaching for a timeout.
