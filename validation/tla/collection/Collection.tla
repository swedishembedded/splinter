\* SPDX-License-Identifier: Apache-2.0
\* Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
\*
\* Swedish Embedded AB implements model-checked storage protocols for learning
\* pipelines, for its clients. If your team needs expertise in TLA+ and garbage
\* collection beside live readers, you can procure our services by sending an
\* email to info@swedishembedded.com.
---------------------------- MODULE Collection ----------------------------
(***************************************************************************)
(* The experience database's collection beside a pin                       *)
(* (crates/expdb/src/manifest/collect.rs, manifest/snapshot.rs). A         *)
(* snapshot's files are needed while a ref names the snapshot or a pin     *)
(* holds it. A collection reads what is needed, sets every other file       *)
(* aside in the trash, reads what is needed again, puts back what is needed *)
(* now and deletes the rest. A reader writes its pin, then checks that every *)
(* file of the snapshot exists, and withdraws the pin when one is gone. A   *)
(* read finds a file where it is, or in the trash while a collection has it *)
(* aside (the backends' open and locate).                                   *)
(*                                                                         *)
(* Checked: a reader whose pin verified can read every file of its snapshot *)
(* while it holds the pin, and no such file is ever deleted.               *)
(***************************************************************************)

Files == {"f1", "f2"}

\* Whether a read looks in the trash for a file set aside.
ReadsTrash == TRUE

VARIABLES ref, pin, loc, gc, pending, before, after

vars == <<ref, pin, loc, gc, pending, before, after>>

Needed == ref \/ pin \in {"written", "verified"}

Init ==
    /\ ref = TRUE
    /\ pin = "none"
    /\ loc = [f \in Files |-> "present"]
    /\ gc = "idle"
    /\ pending = {}
    /\ before = TRUE
    /\ after = TRUE

\* The ref stops naming the snapshot (it was compacted away).
Unref == ref /\ ref' = FALSE /\ UNCHANGED <<pin, loc, gc, pending, before, after>>

FirstLook ==
    /\ gc = "idle"
    /\ gc' = "aside" /\ before' = Needed /\ pending' = Files
    /\ UNCHANGED <<ref, pin, loc, after>>

SetAside(f) ==
    /\ gc = "aside" /\ f \in pending
    /\ pending' = pending \ {f}
    /\ loc' = (IF ~before /\ loc[f] = "present" THEN [loc EXCEPT ![f] = "trash"] ELSE loc)
    /\ UNCHANGED <<ref, pin, gc, before, after>>

SecondLook ==
    /\ gc = "aside" /\ pending = {}
    /\ gc' = "settling" /\ after' = Needed /\ pending' = Files
    /\ UNCHANGED <<ref, pin, loc, before>>

Settle(f) ==
    /\ gc = "settling" /\ f \in pending
    /\ pending' = pending \ {f}
    /\ loc' = (IF loc[f] = "trash"
               THEN [loc EXCEPT ![f] = IF after THEN "present" ELSE "deleted"]
               ELSE loc)
    /\ UNCHANGED <<ref, pin, gc, before, after>>

Finish ==
    /\ gc = "settling" /\ pending = {}
    /\ gc' = "idle"
    /\ UNCHANGED <<ref, pin, loc, pending, before, after>>

WritePin == pin = "none" /\ pin' = "written" /\ UNCHANGED <<ref, loc, gc, pending, before, after>>

Verify ==
    /\ pin = "written"
    /\ pin' = (IF \A f \in Files : loc[f] = "present" THEN "verified" ELSE "withdrawn")
    /\ UNCHANGED <<ref, loc, gc, pending, before, after>>

Next ==
    \/ Unref \/ FirstLook \/ SecondLook \/ Finish \/ WritePin \/ Verify
    \/ \E f \in Files : SetAside(f) \/ Settle(f)

Spec == Init /\ [][Next]_vars

Readable(f) == loc[f] = "present" \/ (ReadsTrash /\ loc[f] = "trash")

PinnedReadable == pin = "verified" => \A f \in Files : Readable(f)
PinnedNeverDeleted == pin = "verified" => \A f \in Files : loc[f] # "deleted"
=============================================================================
