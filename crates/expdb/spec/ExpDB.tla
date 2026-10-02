---------------------------- MODULE ExpDB ----------------------------
(***************************************************************************)
(* The storage protocol of splinter-expdb: immutable object files named by *)
(* content, immutable manifests that only add and remove files, refs that  *)
(* point at manifest heads, pinned snapshots, compaction, catalog merging, *)
(* absorbing idle writer refs, and garbage collection.                     *)
(*                                                                         *)
(* An object is the set of records it holds, so it is content-addressed:   *)
(* two objects with the same records are the same object. A manifest is    *)
(* the record of its contents (parents, adds, removes, nonce), so identical *)
(* manifests are the same manifest.                                        *)
(*                                                                         *)
(* Each safeguard of the implementation is a switch, so a counterexample   *)
(* can be produced for the protocol without it and a proof of safety with  *)
(* it. The feature switches (checkpoints, pins, absorbing) scope a run.    *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
    Records,          \* the records writers can create
    Writers,          \* writer processes, each with its own ref chain
    Compactors,       \* compactor processes, which may overlap
    MaxCompactions,   \* bound on compactions per run
    MaxPins,          \* bound on pins per run
    MaxManifests,     \* bound on manifests created
    ExcludeSelf,      \* a compaction does not remove an input equal to its output
    TouchOnDedupe,    \* re-publishing an existing file restarts its grace period
    SafeAbsorb,       \* idle refs are retired, folded into the catalog, then discarded
    TrashGC,          \* GC sets candidates aside, re-reads pins, restores, then deletes
    GraceHolds,       \* a writer publishes, and a collection ends, within the grace period
    EnableCheckpoint, \* include catalog checkpoints
    EnablePins,       \* include pins
    EnableAbsorb,     \* include absorbing idle writer refs
    EnableGC          \* include time and collection

NoMan == [p |-> {}, a |-> {}, r |-> {}, s |-> FALSE, n |-> 0 - 1]
Objects == {o \in SUBSET Records : o # {}}

VARIABLES
    disk, young,      \* object files, and those still inside the grace period
    mdisk, myoung,    \* manifest files, and those still inside the grace period
    wref,             \* each writer's own ref: a manifest or NoMan
    orefs,            \* refs made by publish_once: a set of manifests
    cats,             \* the catalog: a set of immutable head refs, only ever added to or trimmed
    rref,             \* refs being retired: still heads until folded into the catalog
    chk,              \* unsafe absorb only: the head a writer's ref held when it was checked
    fold,             \* the folder: idle, or a retired head with a merge in hand
    pins, pinok,      \* pinned snapshots, and the ones whose validation passed
    npin,
    committed,        \* ground truth: records a writer has published
    used,             \* records already written into some object
    pend,             \* each writer's sealed, not yet published, object
    nonce, nmade,     \* distinguishes published manifests; manifests created
    comp, ncomp,      \* each compactor's state, and compactions published
    gc                \* garbage collector state

vars == <<disk, young, mdisk, myoung, wref, orefs, cats, rref, chk, fold, pins, pinok,
          npin, committed, used, pend, nonce, nmade, comp, ncomp, gc>>

-------------------------------------------------------------------------
\* Manifests and resolution.

Mk(parents, add, remove, squash, n) ==
    [p |-> parents, a |-> add, r |-> remove, s |-> squash, n |-> n]

Step(S) == S \cup UNION {m.p : m \in {x \in S : ~x.s}}
StepAll(S) == S \cup UNION {m.p : m \in S}

RECURSIVE Iter(_, _)
Iter(S, k) == IF k = 0 THEN S ELSE Iter(Step(S), k - 1)
RECURSIVE IterAll(_, _)
IterAll(S, k) == IF k = 0 THEN S ELSE IterAll(StepAll(S), k - 1)

\* Everything reachable from `S`; resolution stops at a checkpoint, which
\* carries the whole state. `AncAll` walks through checkpoints too.
Anc(S)    == Iter(S, MaxManifests)
AncAll(S) == IterAll(S, MaxManifests)

Adds(S) == UNION {m.a : m \in Anc(S)}
Rems(S) == UNION {m.r : m \in Anc(S)}
Live(S) == Adds(S) \ Rems(S)

JobHeads == ({wref[w] : w \in Writers} \ {NoMan}) \cup orefs \cup rref
Heads    == JobHeads \cup cats

PinnedLive == UNION {Live({p}) : p \in pins}
PinnedMans == UNION {Anc({p}) : p \in pins}

\* Writing a file that may already exist. Content addressing makes the write a
\* no-op on a copy that is already there; TouchOnDedupe makes it also restart
\* that copy's grace period.
PutObject(o) ==
    /\ disk' = disk \cup {o}
    /\ young' = IF o \in disk THEN (IF TouchOnDedupe THEN young \cup {o} ELSE young)
                ELSE young \cup {o}

PutM(m) ==
    /\ mdisk' = mdisk \cup {m}
    /\ myoung' = IF m \in mdisk THEN (IF TouchOnDedupe THEN myoung \cup {m} ELSE myoung)
                 ELSE myoung \cup {m}
    /\ nmade' = nmade + 1

CanMake(k) == nmade + k <= MaxManifests

MergeParents(ps) == IF Cardinality(ps) = 1 THEN CHOOSE x \in ps : TRUE
                    ELSE Mk(ps, {}, {}, FALSE, 0)

Absorbed(m) == m \in AncAll(cats)

\* Catalog heads that another head already contains may be dropped.
Trim(h) == {x \in cats : x # h /\ x \in AncAll({h})}

-------------------------------------------------------------------------
Init ==
    /\ disk = {} /\ young = {} /\ mdisk = {} /\ myoung = {}
    /\ wref = [w \in Writers |-> NoMan]
    /\ orefs = {} /\ cats = {} /\ rref = {}
    /\ chk = [w \in Writers |-> NoMan]
    /\ fold = [stage |-> "idle"]
    /\ pins = {} /\ pinok = {} /\ npin = 0
    /\ committed = {} /\ used = {}
    /\ pend = [w \in Writers |-> {}]
    /\ nonce = 1 /\ nmade = 0
    /\ comp = [c \in Compactors |-> [stage |-> "idle"]] /\ ncomp = 0
    /\ gc = [stage |-> "idle"]

-------------------------------------------------------------------------
\* Writers: seal a segment, then publish a manifest that names it.

WriteSegment(w) ==
    /\ pend[w] = {}
    /\ \E o \in Objects :
         /\ o \subseteq Records \ used
         /\ PutObject(o)
         /\ used' = used \cup o
         /\ pend' = [pend EXCEPT ![w] = o]
    /\ UNCHANGED <<mdisk, myoung, wref, orefs, cats, rref, chk, fold, pins, pinok, npin,
                   committed, nonce, nmade, comp, ncomp, gc>>

WriterCrash(w) ==       \* the process dies between sealing and publishing
    /\ pend[w] # {}
    /\ pend' = [pend EXCEPT ![w] = {}]
    /\ UNCHANGED <<disk, young, mdisk, myoung, wref, orefs, cats, rref, chk, fold, pins,
                   pinok, npin, committed, used, nonce, nmade, comp, ncomp, gc>>

PublishSegment(w) ==
    /\ pend[w] # {}
    /\ CanMake(1)
    /\ LET m == Mk(IF wref[w] = NoMan THEN {} ELSE {wref[w]}, {pend[w]}, {}, FALSE, nonce)
       IN /\ PutM(m)
          /\ wref' = [wref EXCEPT ![w] = m]
    /\ committed' = committed \cup pend[w]
    /\ pend' = [pend EXCEPT ![w] = {}]
    /\ nonce' = nonce + 1
    /\ UNCHANGED <<disk, young, orefs, cats, rref, chk, fold, pins, pinok, npin, used,
                   comp, ncomp, gc>>

-------------------------------------------------------------------------
\* Compaction: read the live set, write the union of some inputs, publish a
\* manifest that adds it and removes the inputs, on a ref of its own.

CompactChoose(c) ==
    /\ comp[c].stage = "idle" /\ ncomp < MaxCompactions
    /\ \E ins \in SUBSET Live(Heads) :
         /\ Cardinality(ins) >= 2
         /\ comp' = [comp EXCEPT ![c] = [stage |-> "read", ins |-> ins, out |-> UNION ins]]
    /\ UNCHANGED <<disk, young, mdisk, myoung, wref, orefs, cats, rref, chk, fold, pins,
                   pinok, npin, committed, used, pend, nonce, nmade, ncomp, gc>>

CompactWrite(c) ==
    /\ comp[c].stage = "read"
    /\ IF comp[c].ins \subseteq disk
       THEN /\ PutObject(comp[c].out)
            /\ comp' = [comp EXCEPT ![c].stage = "written"]
       ELSE /\ comp' = [comp EXCEPT ![c] = [stage |-> "idle"]]   \* an input was collected: the read fails
            /\ UNCHANGED <<disk, young>>
    /\ UNCHANGED <<mdisk, myoung, wref, orefs, cats, rref, chk, fold, pins, pinok, npin,
                   committed, used, pend, nonce, nmade, ncomp, gc>>

CompactPublish(c) ==
    /\ comp[c].stage = "written"
    /\ CanMake(1)
    /\ LET removed == IF ExcludeSelf THEN comp[c].ins \ {comp[c].out} ELSE comp[c].ins
           m == Mk({}, {comp[c].out}, removed, FALSE, nonce)
       IN /\ PutM(m)
          /\ orefs' = orefs \cup {m}
    /\ nonce' = nonce + 1
    /\ ncomp' = ncomp + 1
    /\ comp' = [comp EXCEPT ![c] = [stage |-> "idle"]]
    /\ UNCHANGED <<disk, young, wref, cats, rref, chk, fold, pins, pinok, npin, committed,
                   used, pend, gc>>

CompactCrash(c) ==
    /\ comp[c].stage # "idle"
    /\ comp' = [comp EXCEPT ![c] = [stage |-> "idle"]]
    /\ UNCHANGED <<disk, young, mdisk, myoung, wref, orefs, cats, rref, chk, fold, pins,
                   pinok, npin, committed, used, pend, nonce, nmade, ncomp, gc>>

-------------------------------------------------------------------------
\* Catalog: merge every job head the catalog does not yet hold (a no-op if it
\* holds them all); checkpoint the whole state into one manifest.

\* Writing the merged head adds a file; dropping the heads it contains is a
\* separate step. Nothing is overwritten, so a concurrent merge cannot lose
\* what another absorbed.
MergeCatalog ==
    /\ EnableCheckpoint \/ EnableAbsorb
    /\ LET known == AncAll(cats)
           fresh == JobHeads \ known
           ps == fresh \cup cats
       IN /\ ps # {}
          /\ (fresh # {} \/ Cardinality(cats) > 1)
          /\ CanMake(1)
          /\ LET m == MergeParents(ps)
             IN /\ PutM(m)
                /\ cats' = cats \cup {m}
    /\ UNCHANGED <<disk, young, wref, orefs, rref, chk, fold, pins, pinok, npin, committed,
                   used, pend, nonce, comp, ncomp, gc>>

TrimCatalog ==
    /\ EnableCheckpoint \/ EnableAbsorb
    /\ \E h \in cats : Trim(h) # {}
    /\ \E h \in cats : cats' = cats \ Trim(h)
    /\ UNCHANGED <<disk, young, mdisk, myoung, wref, orefs, rref, chk, fold, pins, pinok,
                   npin, committed, used, pend, nonce, nmade, comp, ncomp, gc>>

Checkpoint ==
    /\ EnableCheckpoint
    /\ Heads # {}
    /\ CanMake(2)
    /\ LET h == MergeParents(Heads)
           sq == Mk({h}, Live({h}), Rems({h}), TRUE, nonce)
       IN /\ mdisk' = mdisk \cup {h, sq}
          /\ myoung' = myoung \cup ({h, sq} \ mdisk) \cup (IF TouchOnDedupe THEN {h, sq} ELSE {})
          /\ cats' = cats \cup {sq}
    /\ nonce' = nonce + 1
    /\ nmade' = nmade + 2
    /\ UNCHANGED <<disk, young, wref, orefs, rref, chk, fold, pins, pinok, npin, committed,
                   used, pend, comp, ncomp, gc>>

-------------------------------------------------------------------------
\* Absorbing idle refs into the catalog, so finished writers leave no refs.
\*
\* Safe form (what the code does): move the ref atomically aside, so the value
\* it held at that instant is known; fold that head into the catalog, reading
\* the catalog again and repeating until the head is there to stay; only then
\* discard the retired ref. A retired ref still counts as a head.
\*
\* Unsafe form: check that the ref's head is in the catalog, and later delete
\* the ref whatever it holds by then.

AbsorbRetire(w) ==
    /\ EnableAbsorb /\ SafeAbsorb
    /\ wref[w] # NoMan
    /\ rref' = rref \cup {wref[w]}
    /\ wref' = [wref EXCEPT ![w] = NoMan]
    /\ UNCHANGED <<disk, young, mdisk, myoung, orefs, cats, chk, fold, pins, pinok, npin,
                   committed, used, pend, nonce, nmade, comp, ncomp, gc>>

FoldPrepare ==
    /\ EnableAbsorb /\ SafeAbsorb
    /\ fold.stage = "idle"
    /\ \E r \in rref :
         IF Absorbed(r)
         THEN /\ fold' = [stage |-> "prepared", r |-> r, m |-> r]
              /\ UNCHANGED <<mdisk, myoung, nmade>>
         ELSE /\ CanMake(1)
              /\ LET m == MergeParents({r} \cup cats)
                 IN /\ PutM(m)
                    /\ fold' = [stage |-> "prepared", r |-> r, m |-> m]
    /\ UNCHANGED <<disk, young, wref, orefs, cats, rref, chk, pins, pinok, npin, committed,
                   used, pend, nonce, comp, ncomp, gc>>

FoldCommit ==       \* adds a head; never replaces one
    /\ fold.stage = "prepared"
    /\ cats' = cats \cup {fold.m}
    /\ fold' = [fold EXCEPT !.stage = "committed"]
    /\ UNCHANGED <<disk, young, mdisk, myoung, wref, orefs, rref, chk, pins, pinok, npin,
                   committed, used, pend, nonce, nmade, comp, ncomp, gc>>

FoldDiscard ==      \* discard the retired ref only if the catalog holds its head now
    /\ fold.stage = "committed"
    /\ IF Absorbed(fold.r) THEN rref' = rref \ {fold.r} ELSE rref' = rref
    /\ fold' = [stage |-> "idle"]
    /\ UNCHANGED <<disk, young, mdisk, myoung, wref, orefs, cats, chk, pins, pinok, npin,
                   committed, used, pend, nonce, nmade, comp, ncomp, gc>>

AbsorbCheck(w) ==
    /\ EnableAbsorb /\ ~SafeAbsorb
    /\ wref[w] # NoMan /\ Absorbed(wref[w])
    /\ chk' = [chk EXCEPT ![w] = wref[w]]
    /\ UNCHANGED <<disk, young, mdisk, myoung, wref, orefs, cats, rref, fold, pins, pinok,
                   npin, committed, used, pend, nonce, nmade, comp, ncomp, gc>>

AbsorbDeleteUnsafe(w) ==
    /\ EnableAbsorb /\ ~SafeAbsorb
    /\ chk[w] # NoMan /\ wref[w] # NoMan
    /\ wref' = [wref EXCEPT ![w] = NoMan]
    /\ chk' = [chk EXCEPT ![w] = NoMan]
    /\ UNCHANGED <<disk, young, mdisk, myoung, orefs, cats, rref, fold, pins, pinok, npin,
                   committed, used, pend, nonce, nmade, comp, ncomp, gc>>

AbsorbPruneOnce ==       \* a ref nobody will write again: always safe to drop once absorbed
    /\ EnableAbsorb
    /\ \E m \in orefs : /\ Absorbed(m)
                        /\ orefs' = orefs \ {m}
    /\ UNCHANGED <<disk, young, mdisk, myoung, wref, cats, rref, chk, fold, pins, pinok, npin,
                   committed, used, pend, nonce, nmade, comp, ncomp, gc>>

-------------------------------------------------------------------------
\* Pins. Pinning an older snapshot writes the pin, then validates that the
\* snapshot's manifests and objects still exist; if not the pin is withdrawn.

PinWrite ==
    /\ EnablePins /\ npin < MaxPins
    /\ \E m \in mdisk :
         /\ pins' = pins \cup {m}
         /\ npin' = npin + 1
    /\ UNCHANGED <<disk, young, mdisk, myoung, wref, orefs, cats, rref, chk, fold, pinok,
                   committed, used, pend, nonce, nmade, comp, ncomp, gc>>

PinValidate ==
    /\ EnablePins
    /\ \E p \in pins \ pinok :
         IF Anc({p}) \subseteq mdisk /\ Live({p}) \subseteq disk
         THEN pinok' = pinok \cup {p} /\ pins' = pins
         ELSE pinok' = pinok /\ pins' = pins \ {p}
    /\ UNCHANGED <<disk, young, mdisk, myoung, wref, orefs, cats, rref, chk, fold, npin,
                   committed, used, pend, nonce, nmade, comp, ncomp, gc>>

Unpin ==
    /\ EnablePins
    /\ \E p \in pins : pins' = pins \ {p} /\ pinok' = pinok \ {p}
    /\ UNCHANGED <<disk, young, mdisk, myoung, wref, orefs, cats, rref, chk, fold, npin,
                   committed, used, pend, nonce, nmade, comp, ncomp, gc>>

-------------------------------------------------------------------------
\* Time: a file leaves its grace period. With GraceHolds a writer's or the
\* compactor's file does not age before it is published, and no file ages
\* during a collection (the grace period is far longer than one collection).

Protected(o) == (\E w \in Writers : pend[w] = o)
                \/ (\E c \in Compactors : comp[c].stage = "written" /\ comp[c].out = o)

AgeObject ==
    /\ EnableGC
    /\ ~GraceHolds \/ gc.stage = "idle"
    /\ \E o \in young : (~GraceHolds \/ ~Protected(o)) /\ young' = young \ {o}
    /\ UNCHANGED <<disk, mdisk, myoung, wref, orefs, cats, rref, chk, fold, pins, pinok, npin,
                   committed, used, pend, nonce, nmade, comp, ncomp, gc>>

\* A merge manifest the folder has written but not yet pointed the catalog at is
\* protected for the same reason: the code writes it and sets the ref in one call.
ProtectedM(m) == fold.stage = "prepared" /\ fold.m = m

AgeManifest ==
    /\ EnableGC
    /\ ~GraceHolds \/ gc.stage = "idle"
    /\ \E m \in myoung : (~GraceHolds \/ ~ProtectedM(m)) /\ myoung' = myoung \ {m}
    /\ UNCHANGED <<disk, young, mdisk, wref, orefs, cats, rref, chk, fold, pins, pinok, npin,
                   committed, used, pend, nonce, nmade, comp, ncomp, gc>>

-------------------------------------------------------------------------
\* Garbage collection. It reads the refs and pins once, then removes files
\* nothing it read needs and that have left their grace period. TrashGC
\* instead moves candidates aside, reads the refs and pins again, restores what
\* is needed now (or has become young) and only then deletes the rest.

GCStart ==
    /\ EnableGC
    /\ gc.stage = "idle"
    /\ gc' = [stage |-> "read", needO |-> Live(Heads) \cup PinnedLive,
              needM |-> Anc(Heads) \cup PinnedMans, trashO |-> {}, trashM |-> {}]
    /\ UNCHANGED <<disk, young, mdisk, myoung, wref, orefs, cats, rref, chk, fold, pins,
                   pinok, npin, committed, used, pend, nonce, nmade, comp, ncomp>>

GCDeleteObject ==
    /\ gc.stage = "read"
    /\ \E o \in (disk \ young) \ gc.needO :
         /\ disk' = disk \ {o}
         /\ gc' = IF TrashGC THEN [gc EXCEPT !.trashO = @ \cup {o}] ELSE gc
    /\ UNCHANGED <<young, mdisk, myoung, wref, orefs, cats, rref, chk, fold, pins, pinok,
                   npin, committed, used, pend, nonce, nmade, comp, ncomp>>

GCDeleteManifest ==
    /\ gc.stage = "read"
    /\ \E m \in (mdisk \ myoung) \ gc.needM :
         /\ mdisk' = mdisk \ {m}
         /\ gc' = IF TrashGC THEN [gc EXCEPT !.trashM = @ \cup {m}] ELSE gc
    /\ UNCHANGED <<disk, young, myoung, wref, orefs, cats, rref, chk, fold, pins, pinok,
                   npin, committed, used, pend, nonce, nmade, comp, ncomp>>

GCFinish ==
    /\ gc.stage = "read"
    /\ IF TrashGC
       THEN LET needO == Live(Heads) \cup PinnedLive
                needM == Anc(Heads) \cup PinnedMans
            IN /\ disk' = disk \cup (gc.trashO \cap (needO \cup young))
               /\ mdisk' = mdisk \cup (gc.trashM \cap (needM \cup myoung))
       ELSE UNCHANGED <<disk, mdisk>>
    /\ gc' = [stage |-> "idle"]
    /\ UNCHANGED <<young, myoung, wref, orefs, cats, rref, chk, fold, pins, pinok, npin,
                   committed, used, pend, nonce, nmade, comp, ncomp>>

-------------------------------------------------------------------------
Next ==
    \/ \E w \in Writers : WriteSegment(w) \/ WriterCrash(w) \/ PublishSegment(w)
                          \/ AbsorbRetire(w) \/ AbsorbCheck(w) \/ AbsorbDeleteUnsafe(w)
    \/ \E c \in Compactors : CompactChoose(c) \/ CompactWrite(c) \/ CompactPublish(c) \/ CompactCrash(c)
    \/ MergeCatalog \/ Checkpoint
    \/ FoldPrepare \/ FoldCommit \/ FoldDiscard \/ AbsorbPruneOnce
    \/ PinWrite \/ PinValidate \/ Unpin
    \/ AgeObject \/ AgeManifest
    \/ GCStart \/ GCDeleteObject \/ GCDeleteManifest \/ GCFinish

Spec == Init /\ [][Next]_vars

Sym == Permutations(Writers) \cup Permutations(Records) \cup Permutations(Compactors)

-------------------------------------------------------------------------
\* What must always hold.

TypeOK ==
    /\ disk \subseteq Objects /\ young \subseteq disk
    /\ committed \subseteq Records

\* Every file a reader could need exists: the catalog, the refs, every
\* manifest and object reachable from them, and no published record is lost.
\* While a collection has candidates set aside a needed file may be briefly
\* absent; it is restored before the collection ends, so these are stated for
\* when none is running. A collection without the set-aside step loses such a
\* file for good.
ManifestsExist == gc.stage = "idle" => Anc(Heads) \subseteq mdisk
ObjectsExist   == gc.stage = "idle" => Live(Heads) \subseteq disk
NoLostRecord   == gc.stage = "idle" => committed \subseteq UNION Live(Heads)

\* A pin that passed validation keeps its whole snapshot readable.
PinnedReadable ==
    gc.stage = "idle" => \A p \in pinok : Anc({p}) \subseteq mdisk /\ Live({p}) \subseteq disk

\* A file is never both added and removed by one manifest.
NoSelfRemoval == \A m \in mdisk : m.a \cap m.r = {}

Safety == TypeOK /\ ManifestsExist /\ ObjectsExist /\ NoLostRecord /\ PinnedReadable /\ NoSelfRemoval
=========================================================================
