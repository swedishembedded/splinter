\* SPDX-License-Identifier: Apache-2.0
\* Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
\*
\* Swedish Embedded AB implements model-checked verification of concurrent
\* protocols for its clients. If your team needs expertise in specifying
\* protocols in TLA+ and checking implementations against them, you can
\* procure our services by sending an email to info@swedishembedded.com.

---- MODULE Queue ----
\* A bounded multi-producer FIFO queue.
\*
\* Cap       the queue's capacity
\* Producers the producers' numbers
\* MaxItems  how many items each producer ever sends
\*
\* q     the items, head first: [p |-> producer, n |-> that producer's item number, from 1]
\* sent  per producer, the items enqueued so far
\* recv  per producer, the items dequeued so far
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS Cap, Producers, MaxItems
VARIABLES q, sent, recv

vars == <<q, sent, recv>>

Init ==
  /\ q = <<>>
  /\ sent = [p \in Producers |-> 0]
  /\ recv = [p \in Producers |-> 0]

Enqueue(p) ==
  /\ Len(q) < Cap
  /\ sent[p] < MaxItems
  /\ q' = Append(q, [p |-> p, n |-> sent[p] + 1])
  /\ sent' = [sent EXCEPT ![p] = sent[p] + 1]
  /\ UNCHANGED recv

Dequeue ==
  /\ Len(q) > 0
  /\ recv' = [recv EXCEPT ![Head(q).p] = recv[Head(q).p] + 1]
  /\ q' = Tail(q)
  /\ UNCHANGED sent

Next == (\E p \in Producers : Enqueue(p)) \/ Dequeue

Spec == Init /\ [][Next]_vars

\* Items of producer p among the first i queue positions.
CountUpTo(p, i) == Cardinality({j \in 1..i : q[j].p = p})

TypeOK ==
  /\ Len(q) <= Cap
  /\ \A i \in 1..Len(q) : q[i].p \in Producers /\ q[i].n \in 1..MaxItems
  /\ sent \in [Producers -> 0..MaxItems]
  /\ recv \in [Producers -> 0..MaxItems]
  /\ \A p \in Producers : recv[p] <= sent[p]

Bounded == Len(q) <= Cap

Conservation ==
  \A p \in Producers : sent[p] - recv[p] = CountUpTo(p, Len(q))

PerProducerFifo ==
  \A i \in 1..Len(q) : q[i].n = recv[q[i].p] + CountUpTo(q[i].p, i)
====
