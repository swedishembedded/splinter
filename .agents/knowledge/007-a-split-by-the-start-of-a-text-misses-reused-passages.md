# 007. A split by the start of a text misses reused passages

Held-out documents were grouped with training documents by comparing the
start of each text. Thirty-seven of 89 held-out documents nevertheless shared
an eight-word run with a training document, some dozens of them, because a
passage was reused deep inside a longer text. Group by shared runs anywhere
in the text, ignore runs held by many documents (boilerplate), and re-freeze
before any model has read the split. Editor footnotes inside a body make two
documents look related only because both cite the same book: strip apparatus
before comparing or training, but compare and hash the stored text so the
digests stay stable.
