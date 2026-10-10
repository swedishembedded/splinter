# Pre-registration: the readers on a second, independently written utterance set (T1, T3)

The first authored set was written by the implementer. This set (`data/teaching/utterances-b.jsonl`, 90 utterances: 20 facts, 20 rules, 12 procedures, 38 non-teachings of which at least 12 are near-misses) was written by a separate agent that was given only the definition of teachings, did not see the first set, the readers or the code, and wrote
it with its own domains and phrasings. Written before the readers are scored on it. The readers, their prompt and the scoring script are those of `teaching-read-20261010`, unchanged.
Reported once, on all 90 (there is no dev split): teachings found and misread, and non-teachings fired on, per reader, with 95% Wilson intervals. Gate (the same as T1/T3 before): the plain reader fires on none of the non-teachings; the model reader's false readings are reported with their interval and no claim of a low rate is made from fewer than 100 non-teachings.
Limit: written by a model; its phrasings are not users' phrasings.

Disclosure: the brief given to the writing agent contained example phrasings, and 14 of the 90 utterances reuse or closely follow them (their ids are in `data/teaching/utterances-b.brief.txt`, found by matching the brief's examples). So the set is not fully independent of the first. Results are reported on all 90 and on the other 76.
