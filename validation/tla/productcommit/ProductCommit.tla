\* SPDX-License-Identifier: Apache-2.0
\* Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
\*
\* Swedish Embedded AB implements model-checked storage protocols for learning
\* pipelines, for its clients. If your team needs expertise in TLA+ and
\* immutable stores, you can procure our services by sending an email to
\* info@swedishembedded.com.
--------------------------- MODULE ProductCommit ---------------------------
(***************************************************************************)
(* The commit of a product (crates/store/src/factory/products/commit.rs).  *)
(* Publishers of one stage, each with its own content, first prepare (they *)
(* find the stage uncommitted, or committed already, and return or refuse  *)
(* at once), then commit: the manifest is linked into place only where     *)
(* none is, and a publisher that finds one returns it when its content is  *)
(* the same and is refused when it differs. A publisher may crash at any   *)
(* step.                                                                   *)
(*                                                                         *)
(* Checked: a committed product never changes, and a publisher that        *)
(* returned a product returned the one committed: no write is lost         *)
(* silently.                                                               *)
(***************************************************************************)

\* Three publishers: two with one content (the same output computed twice),
\* one with another.
Writers == {"w1", "w2", "w3"}
Content == [w \in Writers |-> IF w = "w2" THEN "b" ELSE "a"]

None == "none"

VARIABLES manifest, pc, returned

vars == <<manifest, pc, returned>>

Init ==
    /\ manifest = None
    /\ pc = [w \in Writers |-> "idle"]
    /\ returned = [w \in Writers |-> None]

Prepare(w) ==
    /\ pc[w] = "idle"
    /\ \/ /\ manifest = None
          /\ pc' = [pc EXCEPT ![w] = "prepared"]
          /\ UNCHANGED returned
       \/ /\ manifest = Content[w]
          /\ pc' = [pc EXCEPT ![w] = "done"]
          /\ returned' = [returned EXCEPT ![w] = manifest]
       \/ /\ manifest # None /\ manifest # Content[w]
          /\ pc' = [pc EXCEPT ![w] = "refused"]
          /\ UNCHANGED returned
    /\ UNCHANGED manifest

Commit(w) ==
    /\ pc[w] = "prepared"
    /\ \/ /\ manifest = None
          /\ manifest' = Content[w]
          /\ pc' = [pc EXCEPT ![w] = "done"]
          /\ returned' = [returned EXCEPT ![w] = Content[w]]
       \/ /\ manifest # None /\ manifest = Content[w]
          /\ pc' = [pc EXCEPT ![w] = "done"]
          /\ returned' = [returned EXCEPT ![w] = manifest]
          /\ UNCHANGED manifest
       \/ /\ manifest # None /\ manifest # Content[w]
          /\ pc' = [pc EXCEPT ![w] = "refused"]
          /\ UNCHANGED <<manifest, returned>>

Crash(w) ==
    /\ pc[w] \in {"idle", "prepared"}
    /\ pc' = [pc EXCEPT ![w] = "crashed"]
    /\ UNCHANGED <<manifest, returned>>

Next == \E w \in Writers : Prepare(w) \/ Commit(w) \/ Crash(w)

Spec == Init /\ [][Next]_vars

\* Every publisher that returned a product returned the committed one.
NoSilentLoss == \A w \in Writers : returned[w] # None => returned[w] = manifest

CommittedNeverChanges == [][manifest # None => manifest' = manifest]_manifest
=============================================================================
