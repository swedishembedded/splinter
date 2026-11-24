#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements measurement of agent evaluations that cannot be flattered by trials that never
# finished, for its clients. If your team needs expertise in telling what an agent fails at from its trajectories,
# you can procure our services by sending an email to info@swedishembedded.com.
"""What an evaluation run's arms did, read from its trials.

usage: run-metrics.py RUN_DIR [--json] [--by-kind]

For every evaluation arm of the run (a `splinter run` or `fork` output), the counted slots, the passes, the
slots that never counted, the physical attempts the infrastructure retried, and from the trajectories:

  - repeat_after_error: of the tool calls that followed a call whose result showed a Vim error, the share that
    were identical to it (a model that does not read its errors repeats them);
  - loop_stops: trials a loop watchdog ended;
  - http_400: model requests the server refused (a request over the context window is one);
  - thinking: replies that carry reasoning (an arm run in no-think mode must have none);
  - emptied_files: results that report a file written empty.

An arm run with several attempts per task also reports the tasks solved within the first k attempts.

With `--by-kind` the passes are also split by the kind of task, the word its name starts with after the split
(`train/composite-007-x` is `composite`; a name that starts with a number is `numbered`).

The pass rate over all slots counts a slot that never counted as not passed: the rate over counted slots alone
drops exactly the trials a failing policy loses.
"""
import collections
import glob
import json
import os
import sys


def products(run):
    for path in sorted(glob.glob(os.path.join(run, 'products', '*.json'))):
        product = json.load(open(path))
        name = product.get('name', '')
        if '_arms' in name and any(f['path'] == 'attempts.jsonl' for f in product['files']):
            yield name.replace('/_arms', ''), product


def blob(run, product, path):
    for f in product['files']:
        if f['path'] == path:
            full = os.path.join(run, 'blobs', 'sha256', f['digest'].split(':')[1])
            return open(full, errors='replace').read()
    return None


def kind_of(task):
    name = task.split('/', 1)[-1]
    word = name.split('-', 1)[0]
    return 'numbered' if word.isdigit() else word


def arm_metrics(run, name, product):
    files = {f['path'] for f in product['files']}
    last, physical = {}, collections.Counter()
    slots = set()
    for line in blob(run, product, 'attempts.jsonl').splitlines():
        trial = json.loads(line)['trial']
        key = (trial['task'], trial['slot']['slot'])
        slots.add(key)
        physical[trial['grade']['status']] += 1
        if trial['grade']['counted']:
            last[key] = trial
    after_error = repeats = loops = h400 = thinking = emptied = 0
    for trial in last.values():
        trace = trial.get('trace') or {}
        path = trace.get('path')
        if not path or path not in files:
            continue
        steps = json.loads(blob(run, product, path))['steps']
        agent = [s for s in steps if s['source'] == 'agent']
        calls = [json.dumps(c['arguments'], sort_keys=True) for s in agent for c in (s.get('tool_calls') or [])]
        results = [str(o.get('content')) for s in steps for o in (s.get('observation') or {}).get('results', [])]
        for i in range(1, min(len(calls), len(results) + 1)):
            if 'Error detected' in results[i - 1]:
                after_error += 1
                repeats += calls[i] == calls[i - 1]
        emptied += sum(' 0L, 0B written' in r for r in results)
        thinking += sum(1 for s in agent if s.get('reasoning_content'))
        directory = path.rsplit('/agent/', 1)[0]
        gateway = f'{directory}/gateway/requests.jsonl'
        if gateway in files:
            h400 += sum(1 for l in blob(run, product, gateway).splitlines()
                        if l.strip() and json.loads(l).get('status') == 400)
        stderr = f'{directory}/agent/stderr.txt'
        if stderr in files and 'loop:' in blob(run, product, stderr):
            loops += 1
    passed = sum(1 for t in last.values() if t['grade']['passed'])
    kinds = collections.defaultdict(lambda: [0, 0])
    for key in slots:
        kind = kind_of(key[0])
        kinds[kind][1] += 1
        kinds[kind][0] += bool(last.get(key) and last[key]['grade']['passed'])
    # Of an arm run with several attempts per task: pass@1 is the first attempt, solved-within-k a pass in
    # any of the first k attempts (a slot that never counted is not a pass).
    tasks = {key[0] for key in slots}
    attempts = max(key[1] for key in slots) + 1 if slots else 0
    within = [sum(1 for t in tasks if any(last.get((t, a)) and last[(t, a)]['grade']['passed'] for a in range(k)))
              for k in range(1, attempts + 1)]
    return {
        'tasks': len(tasks), 'solved_within_k': within,
        'by_kind': {k: {'passed': v[0], 'slots': v[1]} for k, v in sorted(kinds.items())},
        'arm': name, 'slots': len(slots), 'counted': len(last), 'passed': passed,
        'never_counted': len(slots) - len(last),
        'pass_rate_counted': round(passed / len(last), 4) if last else None,
        'pass_rate_all_slots': round(passed / len(slots), 4) if slots else None,
        'physical_attempts': sum(physical.values()), 'infra_timeouts': physical.get('infra_timeout', 0),
        'budget_exhausted': physical.get('budget_exhausted', 0),
        'repeat_after_error': {'calls_after_error': after_error, 'identical': repeats,
                               'share': round(repeats / after_error, 3) if after_error else None},
        'loop_stops': loops, 'http_400': h400, 'reasoning_replies': thinking, 'emptied_files': emptied,
    }


def main(argv):
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    run = argv[1]
    rows = [arm_metrics(run, name, product) for name, product in products(run)]
    if '--json' in argv:
        print(json.dumps(rows, indent=1))
        return 0
    for r in rows:
        share = r['repeat_after_error']['share']
        print(f"{r['arm']:28} slots {r['slots']:3d} counted {r['counted']:3d} passed {r['passed']:3d} "
              f"(rate {r['pass_rate_all_slots']}, counted-only {r['pass_rate_counted']}) never-counted {r['never_counted']:3d} "
              f"attempts {r['physical_attempts']:3d} infra-timeouts {r['infra_timeouts']:3d} "
              f"repeat-after-error {share} http400 {r['http_400']} loops {r['loop_stops']} reasoning {r['reasoning_replies']}")
        if r['solved_within_k'] and len(r['solved_within_k']) > 1:
            print(f"    solved within k=1..{len(r['solved_within_k'])} of {r['tasks']} tasks: {r['solved_within_k']}")
        if '--by-kind' in argv:
            print('    ' + ', '.join(f"{k} {v['passed']}/{v['slots']}" for k, v in r['by_kind'].items()))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
