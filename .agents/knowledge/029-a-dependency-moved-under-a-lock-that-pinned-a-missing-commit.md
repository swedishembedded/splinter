# 029. The dependency moved under a lock that named a commit nobody had

The committed lock pinned a brain revision absent from the local checkout, so
`make build` failed before compiling anything, and re-pinning to the local
HEADs exposed three drifts that the lock had hidden:

- The agent SDK's event stream became a numbered `EventStream` of `RunEvent`
  (3 compile errors in the agent adapter and one in a sample). A gap is a
  `SessionEvent::EventsDropped { count }` inside the stream, no longer a
  lagged-receiver error, so the adapter counts dropped events from that marker
  and drains what a finished send left buffered by polling the stream once
  per ready event.
- A manifest poll result lost `PartialEq` and `Debug`, and a loaded adapter
  became `{ release, adapter }` (10 compile errors in one test file).
- A trajectory validator started checking the namespaced gate and memory
  records for a schema string and the call ids they name; an import test that
  used the old unversioned shape was refused (the test now builds records in
  the current wire shape, naming a call the step made).

How to avoid it: pin the lock to commits that exist on the remote, and run the
whole build and test against the pinned revisions whenever a dependency moves.
