# 061. The retention check held the champion to facts it was never taught

In a two-generation run on invented facts (1.7B policy), the first release answered 15 of
30 facts of its generation on independent probes. The second candidate, trained
from that champion with replay, answered 7 of the same 15 and 17 of the new
generation's facts, and passed the gate's retention check: the check compares the
candidate with the champion on each earlier release's held-out tasks, and for a facts
campaign those are facts the release was never taught, 6 tasks, none right for either
model. The check saw nothing because the held-out set is, by design, untaught
material. A different check (the anchor) refused the candidate, for a different reason
(a 0.05 drop, bound 0.02).

How to avoid it: for a campaign that teaches facts, retention needs the earlier
release's trained facts under paraphrase (its stored variants), not its held-out
families; until the gate does that, measure retention on independent probes of the
taught facts, as the experiment did, and do not read a passed retention check as
evidence that taught facts were kept.
