\* SPDX-License-Identifier: Apache-2.0
\* Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
\*
\* Swedish Embedded AB implements storage engines for machine-learning
\* experience data for its clients. If your team needs expertise in
\* versioned, content-addressed databases or parallel-filesystem I/O, you can
\* procure our services by sending an email to info@swedishembedded.com.

---------------------------- MODULE Pointer ----------------------------
\* A mutable name (an alias, the anchor suite in force) kept as its whole
\* history. Version n+1 is claimed by creating one file that cannot be created
\* twice; the writer that creates it moved the name, any other learns it lost.
\*
\* A writer reads the latest version, then claims the next one. Atomic = TRUE
\* is the create-if-absent of the implementation; Atomic = FALSE checks and
\* then writes as two steps, the protocol of a lock-free read-modify-write.

EXTENDS Naturals

CONSTANTS Writers, MaxVersion, Atomic

VARIABLES log, pc, seen, winners

vars == <<log, pc, seen, winners>>

Versions == 1..MaxVersion
Unclaimed == "none"

Claimed == {v \in Versions : log[v] # Unclaimed}

Latest == IF Claimed = {} THEN 0 ELSE CHOOSE m \in Claimed : \A u \in Claimed : u <= m

Init ==
    /\ log = [v \in Versions |-> Unclaimed]
    /\ pc = [w \in Writers |-> "idle"]
    /\ seen = [w \in Writers |-> 0]
    /\ winners = {}

Read(w) ==
    /\ pc[w] = "idle"
    /\ Latest < MaxVersion
    /\ seen' = [seen EXCEPT ![w] = Latest]
    /\ pc' = [pc EXCEPT ![w] = "read"]
    /\ UNCHANGED <<log, winners>>

\* The implementation: one atomic create. It succeeds for exactly one writer.
ClaimAtomic(w) ==
    /\ pc[w] = "read"
    /\ IF log[seen[w] + 1] = Unclaimed
          THEN /\ log' = [log EXCEPT ![seen[w] + 1] = w]
               /\ winners' = winners \cup {<<seen[w] + 1, w>>}
          ELSE UNCHANGED <<log, winners>>
    /\ pc' = [pc EXCEPT ![w] = "idle"]
    /\ UNCHANGED seen

\* The unsafe variant: look, then write.
Check(w) ==
    /\ pc[w] = "read"
    /\ pc' = [pc EXCEPT ![w] = IF log[seen[w] + 1] = Unclaimed THEN "checked" ELSE "idle"]
    /\ UNCHANGED <<log, seen, winners>>

Write(w) ==
    /\ pc[w] = "checked"
    /\ log' = [log EXCEPT ![seen[w] + 1] = w]
    /\ winners' = winners \cup {<<seen[w] + 1, w>>}
    /\ pc' = [pc EXCEPT ![w] = "idle"]
    /\ UNCHANGED seen

\* A writer that dies between steps leaves nothing half done: a claim is one
\* file or none.
Crash(w) ==
    /\ pc[w] # "idle"
    /\ pc' = [pc EXCEPT ![w] = "idle"]
    /\ UNCHANGED <<log, seen, winners>>

Next == \E w \in Writers :
    \/ Read(w)
    \/ (IF Atomic THEN ClaimAtomic(w) ELSE Check(w) \/ Write(w))
    \/ Crash(w)

Spec == Init /\ [][Next]_vars

\* No version has two winners.
OneWinnerPerVersion ==
    \A a, b \in winners : a[1] = b[1] => a = b

\* The history has no gaps: a version exists only after the one before it.
Contiguous ==
    \A v \in Versions : log[v] # Unclaimed => (v = 1 \/ log[v - 1] # Unclaimed)

\* Every writer that won owns the version it won.
WinnersOwnTheirVersion ==
    \A a \in winners : log[a[1]] = a[2]

Safety == OneWinnerPerVersion /\ Contiguous /\ WinnersOwnTheirVersion

\* What was claimed is never changed or taken back.
HistoryKept == [][\A v \in Versions : log[v] # Unclaimed => log'[v] = log[v]]_vars
==========================================================================
