\* SPDX-License-Identifier: Apache-2.0
\* Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
\*
\* Swedish Embedded AB implements model-checked storage protocols for learning
\* pipelines, for its clients. If your team needs expertise in TLA+ and fencing
\* protocols, you can procure our services by sending an email to
\* info@swedishembedded.com.
---------------------------- MODULE RunState ----------------------------
(***************************************************************************)
(* The factory's run state (crates/store/src/factory/runstate): one lease  *)
(* that is a fencing token, and stages moved by compare-and-set on their   *)
(* version.                                                                *)
(*                                                                         *)
(* A controller acquires the lease when none is live (the token rises), or *)
(* renews its own; it may take over the lease of a holder it has           *)
(* established is gone (the token rises). Every mutation first checks the  *)
(* caller's fence: the lease's owner and token are the caller's. A stage   *)
(* moves only from the version its writer last read, and a completed stage *)
(* never changes. Time is a counter the controllers do not control; a      *)
(* controller acts on what it last read, however stale.                    *)
(*                                                                         *)
(* Checked: a completed stage has its product and never changes again; no  *)
(* transition rests on a version that another write overtook (no lost     *)
(* update); and no controller whose fence another grant superseded ever   *)
(* writes.                                                                 *)
(***************************************************************************)
EXTENDS Naturals

CONSTANTS Controllers, TTL, MaxTime, MaxToken, MaxVersion

States == {"pending", "running", "completed", "failed"}
NoOne == "none"

VARIABLES now, lease, fence, seen, stage, granted, lostUpdate, staleWrite

vars == <<now, lease, fence, seen, stage, granted, lostUpdate, staleWrite>>

TypeOK ==
    /\ now \in 0..MaxTime
    /\ lease.owner \in Controllers \cup {NoOne}
    /\ lease.token \in 0..MaxToken
    /\ fence \in [Controllers -> 0..MaxToken]
    /\ stage.state \in States
    /\ stage.product \in Controllers \cup {NoOne}

Init ==
    /\ now = 0
    /\ lease = [owner |-> NoOne, token |-> 0, expires |-> 0]
    /\ fence = [c \in Controllers |-> 0]
    /\ seen = [c \in Controllers |-> 0]
    /\ stage = [state |-> "pending", version |-> 0, product |-> NoOne]
    /\ granted = 0
    /\ lostUpdate = FALSE
    /\ staleWrite = FALSE

Live == lease.owner # NoOne /\ lease.expires > now

\* The check every mutation makes before it changes anything.
Fenced(c) == lease.owner = c /\ lease.token = fence[c] /\ fence[c] > 0

Grant(c) ==
    /\ lease.token < MaxToken
    /\ lease' = [owner |-> c, token |-> lease.token + 1, expires |-> now + TTL]
    /\ fence' = [fence EXCEPT ![c] = lease.token + 1]
    /\ granted' = lease.token + 1

Acquire(c) ==
    /\ \/ /\ Live /\ lease.owner = c
          /\ lease' = [lease EXCEPT !.expires = now + TTL]
          /\ fence' = [fence EXCEPT ![c] = lease.token]
          /\ UNCHANGED granted
       \/ /\ ~Live
          /\ Grant(c)
    /\ UNCHANGED <<now, seen, stage, lostUpdate, staleWrite>>

TakeOver(c) ==
    /\ lease.owner # NoOne /\ lease.owner # c
    /\ Grant(c)
    /\ UNCHANGED <<now, seen, stage, lostUpdate, staleWrite>>

Renew(c) ==
    /\ Fenced(c)
    /\ lease' = [lease EXCEPT !.expires = now + TTL]
    /\ UNCHANGED <<now, fence, seen, stage, granted, lostUpdate, staleWrite>>

Release(c) ==
    /\ Fenced(c)
    /\ lease' = [lease EXCEPT !.expires = now]
    /\ UNCHANGED <<now, fence, seen, stage, granted, lostUpdate, staleWrite>>

Read(c) ==
    /\ seen' = [seen EXCEPT ![c] = stage.version]
    /\ UNCHANGED <<now, lease, fence, stage, granted, lostUpdate, staleWrite>>

Transition(c, s) ==
    /\ Fenced(c)
    /\ stage.version = seen[c]
    /\ stage.version < MaxVersion
    /\ stage.state # "completed"
    /\ stage' = [state |-> s, version |-> stage.version + 1,
                 product |-> IF s = "completed" THEN c ELSE NoOne]
    /\ lostUpdate' = (lostUpdate \/ seen[c] # stage.version)
    /\ staleWrite' = (staleWrite \/ fence[c] # granted)
    /\ UNCHANGED <<now, lease, fence, seen, granted>>

Tick ==
    /\ now < MaxTime
    /\ now' = now + 1
    /\ UNCHANGED <<lease, fence, seen, stage, granted, lostUpdate, staleWrite>>

Next ==
    \/ Tick
    \/ \E c \in Controllers :
        \/ Acquire(c) \/ TakeOver(c) \/ Renew(c) \/ Release(c) \/ Read(c)
        \/ \E s \in {"running", "completed", "failed"} : Transition(c, s)

Spec == Init /\ [][Next]_vars

CompletedHasProduct == stage.state = "completed" => stage.product # NoOne
NoLostUpdate == ~lostUpdate
NoStaleWrite == ~staleWrite

CompletedNeverChanges == [][stage.state = "completed" => stage' = stage]_stage
=============================================================================
