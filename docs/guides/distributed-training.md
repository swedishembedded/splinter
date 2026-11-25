<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Running a training job over a cluster

`splinter jobs run-served` runs one `brain/train` action (`speech_sft`, `full_train`, `train`, ...) as written in a
params file, on brain servers, as one job over every card they have. There is no launch script to write: inside a
Slurm allocation Splinter starts one `brain serve` on each node as a task of ONE job step (on an HPE Slingshot fabric
the network is granted per step, so this keeps every rank on the fabric), with the training workload, the shared
artifact store, a fresh job secret and the job's coordinator on the first server; it then talks to them only through
their API and stops them when the job ends. Outside an allocation it starts one server here with every card of the
machine, and with `--brain-endpoint` it uses servers that are already running.

## The request

The params file is one JSON object, the params of the action as `GET /v1/capabilities` lists them. Datasets are not
inlined: `--artifact PARAM=FILE` publishes a file to the servers' artifact store and sets `PARAM` to its digest.
`cards` and `job` are not written by hand: when the action takes them the job is spread over every card of every
server (one rank per card, ranks numbered server by server); when it takes only `cards` it runs on the first server's
cards; otherwise on one card, and Splinter says so. A `seed` left out is drawn and printed, so the run can be repeated.

```json
{"base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
 "languages": "{\"english\": {\"new\": false}}", "voices": "{}",
 "steps": 30, "batch_utterances": 32, "grad_accum": 4, "seed": 20261125}
```

## On the nodes of an allocation

From the first node of the allocation (here inside an Apptainer image; the state root, the model store and the wrapper
scripts on a filesystem every node sees, the node-local scratch bound so the model is staged there):

```bash
step="scripts/cluster/apptainer-step --image $HOME/images/workstation.sif --bind /nobackup --bind /scratch/local --"
splinter --cluster-nodes 'n[80,105,460]' --backend cuda \
    --cluster-exec-prefix "$step scripts/cluster/rank-env --env $HOME/nccl-env.sh --" \
    jobs run-served --action speech_sft --params speech.json \
    --artifact dataset_digest=train.bttshrd --artifact held_out_digest=held-out.bttshrd \
    --out checkpoint
```

Rank 0's step frames are printed on stdout as JSON lines while it trains; the report follows (with `--json`, as one
more JSON line, the one that carries `run`). Each file the job publishes (the checkpoint's `files`, an adapter, a
training record) is fetched by digest into `--out`, checked against it, with the result as `result.json`. Brain's
`parallel` report is checked against the spread asked for, and any difference is printed as a warning. Ctrl-C, or a
failure of any server's ranks, cancels the job on every server.

A node that brings fewer cards than the others is named with its cards: `--cluster-node-cards n80=0,1` serves and
trains on gpu0 and gpu1 of n80 only, and the job's world size counts them (2 + 4 + 4 = 10 ranks above).

## On servers that are already running

`splinter serve` starts the same servers and prints what a client needs. Another process then names them:

```bash
splinter --brain-endpoint https://n105:41000/v1 --brain-endpoint https://n460:41000/v1 \
    --brain-key-file fleet/key --brain-ca-file fleet/ca.pem --brain-artifact-store state/brain-service/artifacts \
    --brain-train-cards 4,4 --brain-train-job-secret fleet/train-job.secret --brain-train-coordinator 10.0.0.5:29517 \
    jobs run-served --action speech_sft --params speech.json --artifact dataset_digest=train.bttshrd
```

The servers must share the artifact store the job's files are published to: a server that cannot find a digest refuses
the job by name.
