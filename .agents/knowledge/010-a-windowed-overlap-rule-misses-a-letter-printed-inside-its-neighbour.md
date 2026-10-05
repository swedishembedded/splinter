# 010. A windowed overlap rule misses a letter printed inside its neighbour

Texts were grouped as one print by the eight-word runs sampled from their
first seven hundred words. On a corpus of 3,005 letters from several
editions of one writer, comparing whole texts found 93 groups the windowed
rule had left apart, 21 of them pairs sharing fifty or more distinct runs:
whole letters, printed by one edition inside a neighbour whose heading the
parser had not cut at, so the shared text began hundreds of words in. Of
the 1,625 files a run had been pointed at, 51 shared a passage of eight
runs or more with a letter the same split held out for its independent
exam, which that exam then scored as unseen.

Compare every run of every text, not a window at the start; keep ignoring
runs held by many texts (a formula of the period's letters); and set the
merge threshold by what two unrelated texts can share by idiom (a 12-word
phrase is five runs and must not merge) against what a reused passage
shares (a 40-word passage is 33). Whole-text comparison of the 3,005
letters took three seconds, so the window bought nothing.
