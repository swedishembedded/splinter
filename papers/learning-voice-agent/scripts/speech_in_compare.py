#!/usr/bin/env python3
"""Compare answers to held-out spoken questions with the text path's (teacher) answers."""
import json, os, re, sys
d = os.path.expanduser('~/resources/speech/ingress/')
STOP = set("what is your view on how would you handle today why does so much to a free people could explain young person danger do see shape country knew advise leader about tell me plainly think of the and in it for that this be as are was to".split())
def words(t): return [w for w in re.findall(r"[a-z']+", t.lower())]
def f1(a, b):
    from collections import Counter
    ca, cb = Counter(words(a)), Counter(words(b)); common = sum((ca & cb).values())
    if not common: return 0.0
    p, r = common / sum(ca.values()), common / sum(cb.values()); return 2 * p * r / (p + r)
teacher = {}
for l in open(d + 'answers-adams.jsonl'):
    r = json.loads(l); teacher[r['id']] = r['answer']
qs = {}
for l in open(d + 'questions-adams.jsonl'):
    r = json.loads(l); qs[r['text']] = r['id']
def score(path):
    r = json.load(open(d + path)); out = []
    for it in r['items']:
        q = it['asked']; t = teacher[qs[q]]
        topic = [w for w in words(q) if w not in STOP]
        out.append((f1(it['answer'], t), any(w in words(it['answer']) for w in topic)))
    n = len(out)
    return {"turns": n, "mean_word_f1_vs_text_path_answer": round(sum(x[0] for x in out) / n, 3), "answers_mentioning_a_topic_word": f"{sum(x[1] for x in out)}/{n}",
            "answer_word_error_rate_after_speaking": r.get('answer_word_error_rate'), "total_seconds": r.get('total_seconds')}
print(json.dumps({p: score(p) for p in sys.argv[1:]}, indent=1))
