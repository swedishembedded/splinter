\* SPDX-License-Identifier: Apache-2.0
\* Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
\*
\* Swedish Embedded AB implements model-checked storage protocols for learning
\* pipelines, for its clients. If your team needs expertise in TLA+ and
\* compaction of content-addressed stores, you can procure our services by
\* sending an email to info@swedishembedded.com.
---------------------------- MODULE Compaction ----------------------------
(***************************************************************************)
(* Compaction of the experience database (crates/expdb/src/compact,       *)
(* manifest/resolve.rs). An object is named by its content, so here an     *)
(* object is the set of records it holds. What is live is every object ever *)
(* added less every object ever removed, over all heads: a removal is       *)
(* permanent. Writers ingest records in objects of their own. Compactors,   *)
(* any number at once and each on the snapshot it last read however stale,  *)
(* merge a group of the objects it saw into one and publish, in one         *)
(* manifest, the merged object added and the group removed (the merged one *)
(* kept if it is one of them); a compactor may also crash after writing the *)
(* merged object and before publishing, leaving it unnamed.                 *)
(*                                                                         *)
(* Checked: every record ever ingested is in some live object, whatever the *)
(* compactors did and however they interleaved: a failed or stale          *)
(* compaction never destroys the only copy of a record.                    *)
(***************************************************************************)
EXTENDS FiniteSets, Naturals

Records == {"r1", "r2", "r3"}
Compactors == {"k1", "k2"}

VARIABLES added, removed, ingested, snap, pc

vars == <<added, removed, ingested, snap, pc>>

Live == added \ removed

\* What a compactor writes for a group: every record of every object in it.
Merge(group) == UNION group

Init ==
    /\ added = {}
    /\ removed = {}
    /\ ingested = {}
    /\ snap = [k \in Compactors |-> {}]
    /\ pc = [k \in Compactors |-> "idle"]

Ingest(r) ==
    /\ r \notin ingested
    /\ ingested' = ingested \cup {r}
    /\ added' = added \cup {{r}}
    /\ UNCHANGED <<removed, snap, pc>>

Look(k) ==
    /\ pc[k] = "idle"
    /\ snap' = [snap EXCEPT ![k] = Live]
    /\ pc' = [pc EXCEPT ![k] = "looked"]
    /\ UNCHANGED <<added, removed, ingested>>

Publish(k, group) ==
    /\ pc[k] = "looked"
    /\ group \subseteq snap[k] /\ Cardinality(group) >= 2
    /\ added' = added \cup {Merge(group)}
    /\ removed' = removed \cup (group \ {Merge(group)})
    /\ pc' = [pc EXCEPT ![k] = "idle"]
    /\ UNCHANGED <<ingested, snap>>

\* Wrote the merged object, died before publishing: nothing names it.
Crash(k) ==
    /\ pc[k] = "looked"
    /\ pc' = [pc EXCEPT ![k] = "idle"]
    /\ UNCHANGED <<added, removed, ingested, snap>>

Next ==
    \/ \E r \in Records : Ingest(r)
    \/ \E k \in Compactors :
        \/ Look(k) \/ Crash(k)
        \/ \E group \in SUBSET snap[k] : Publish(k, group)

Spec == Init /\ [][Next]_vars

NoRecordLost == \A r \in ingested : \E o \in Live : r \in o
=============================================================================
