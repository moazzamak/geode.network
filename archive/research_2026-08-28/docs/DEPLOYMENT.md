# GEODE deployment guide

This document tells you how to deploy the GEODE system.
It explains each part from the start.
It does not refer to internal history.
It uses Simplified Technical English.

## 1. What the system is

GEODE is a service that answers machine-learning tasks.
It does this with a set of frozen learning models, not one large
model.

The system has these parts:

- **Frozen learning models.** Each model is a fixed function. It
  turns an input into a vector of numbers. A model does not change
  after it is registered. A change is a new model, not an edit.
  (See the explanation after this list.)
- **Fits.** For each task, the system computes one exact solution. The
  solution is a set of weights. The system computes the weights with
  one linear-algebra solve. There is no training loop.
- **A registry.** The registry holds the models. Each entry has a
  content hash. The hash proves what was registered.
- **A router.** The router selects the best model for each task.
  It uses measured accuracy, not opinion.
- **Guards.** Guards block inputs that are out of range. Guards block
  arms that have no measured safety coverage.
- **A ledger.** The ledger records every decision. The records are
  chained by hashes. No record can change without breaking the chain.
- **An evidence rule.** Every measured number must be reproducible.
  The number must match a stored hash. If it does not match, the
  number is void.

### What a frozen learning model is

A **learning model** is a program that has learned from examples. It
turns an input, such as a picture, into an answer or a list of
numbers.

Inside the model are many small dials. The dials hold the model's
knowledge. **Learning** means turning the dials. The program tries
many examples. Each try turns the dials a little. After enough tries,
the dials are set, and the program gives good answers.

**Frozen** means the dials are fixed. The program can no longer
change itself. It can only give answers.

Think of a book. Writing the book is learning. Reading the book is
using the model. A printed book does not change when you read it.

We freeze the models for three reasons:

1. **Measure once, and the measurement stays true.** A model that
   changed between Tuesday and Wednesday cannot have a trusted
   measurement from Monday.
2. **The model behaves the same every day.** The same input always
   gives the same output.
3. **We can hash the model like a file.** The hash is the model's
   identity. A different model has a different hash.

Many services use models that keep learning while they run. Those
models change from day to day. This system does not do that. If we
want a better model, we build a new one and register it. The old one
stays in the registry.

## 2. What you deploy

You deploy these items:

- The GEODE package. It is a Python wheel.
- The API service. It gives access over HTTP.
- The command-line tool. It is named `geode`.
- The container image, if you use containers.

## 3. Conditions before you start

You must have these items:

- **Hardware.** One AMD GPU with ROCm support. The GPU must have
  at least 32 GB of RAM on the machine, and 63 GB is better.
  The exact solve needs the memory.
- **Operating system.** Windows or Linux with Python 3.10 or newer.
- **Disk.** Enough space for the model files and the evidence
  files. Model files can be several gigabytes.
- **A cache directory.** The system needs one directory for models
  and data. Set the environment variable `GEODE_CACHE_DIR` to this
  directory before you start any process.
- **Signed contracts.** If the system pays contributors, the legal
  review must be complete first. Without it, run the API on the
  local machine only.

## 4. Deploy, step by step

### 4.1 Install the package

Run this command in the project directory:

```bash
pip install .[api]
```

This installs the package and the API service.

To test the install, run:

```bash
geode version
```

The command must print a version number.

### 4.2 Put the artifacts in the store

The system loads model weights and model files from an artifact
store. The store identifies each file by a hash of its contents.

Do this for each file:

1. Publish the file to the store. The store returns a reference with
   the file hash and the file size.
2. Record the reference with the arm that uses the file.

Before the system uses a file, it checks the hash and the size. If
either does not match, the system refuses the file. A wrong file is
never loaded.

To check one file by hand:

```bash
geode artifacts verify --path <file> --digest <hash>
```

### 4.3 Seed the registry

Register the arms that the system will serve. Each arm needs:

- a unique name,
- a fingerprint (a short vector that describes the tasks it fits),
- its measured accuracy,
- its safety coverage (measured tags),
- its artifact references.

Use only measured numbers. Do not use declared numbers. An arm with
declared-only safety coverage is provisional. A provisional arm
cannot serve safety-flagged tasks.

**What an arm is.** An arm is a frozen learning model plus its
measured record. It is not a model in training. It is a fixed
function with a file of evidence attached, like a library entry in a
catalogue. An arm never changes after registration. An update is a
new arm, not an edit of the old one.

**What a fingerprint is.** A fingerprint is a short vector of numbers
(for example 16 numbers). It describes the kind of task an arm fits.
It is the arm's address in task space. Arms that fit similar tasks
get similar vectors. The router finds an arm by looking for the
nearest fingerprint.

**How to make a fingerprint.** Describe the task with attributes from
a fixed list: the input kind, the output kind, the label count, and
so on. Each attribute has a small learned vector. The fingerprint is
the sum of those vectors, with length one. Because it is a sum:

- a combination never seen before still lands at a defined point,
- changing one attribute is vector arithmetic,
- the same description always gives the same vector, and
- the fingerprint is never trained on task data, only on the
  attribute description.

**How to make an arm.**

1. **Freeze the model.** Hash its code and its weights. The hash
   is its identity.
2. **Measure it.** Fit it on held-out data and record the accuracy,
   with the evidence file and its hash.
3. **Measure its safety coverage.** Run it through the safety probes.
   Several independent verifiers must agree before a tag counts. If
   no measurement exists, the arm is provisional.
4. **Publish its files** to the artifact store and record the
   digests (section 4.2).
5. **Register.** The arm enters with all of the above.

**How the router uses them.**

- A route scores each arm's fingerprint against the task's. The
  nearest wins. Ties break by measured accuracy, then by price.
- A safety-flagged task removes arms that lack the measured tags for
  its requirements. A provisional arm can never serve it.
- A change is a new arm entry. The old entry stays in the registry.

Save the registry state to a snapshot:

```bash
geode route --fp <fingerprint> --snapshot <snapshot.json>
```

The snapshot lets you restore the registry after a restart.

### 4.4 Start the service

Run the API service:

```bash
geode serve --host 127.0.0.1 --port 8000
```

Or start the container:

```bash
docker run --device=/dev/kfd --device=/dev/dri -p 8000:8000 geode-ml
```

Check the service:

```bash
curl http://127.0.0.1:8000/health
```

The reply must show `status: ok`.

## 5. Run the system

### 5.1 Route a task

```bash
geode route --fp 0.9,0.3,0.2,0.1
```

The command prints the selected arm and its score.

For a safety-flagged task, add the required tags:

```bash
geode route --fp 0.9,0.3,0.2,0.1 --tags refusal
```

The system then removes every arm that lacks the measured tags. If no
arm qualifies, the route is empty. An empty route means: escalate, do
not guess.

### 5.2 Guard the inputs

The input guard decides which inputs are safe to answer at all. It
works before any model sees the input.

**What it does.** The guard compares each new input to the inputs the
system knows. If the new input is too different, the guard stops it.
The route returns empty. The caller must escalate. The system never
guesses on a stopped input.

This matters because models behave well only on the kind of input
they were measured on. An input far outside that range can make a
model give a wrong or harmful answer with high confidence. The
guard turns that case into "no answer" instead of a bad answer.

**How it works, step by step.**

1. **Learn the reference range.** Give the guard a set of known,
   trusted input vectors from the real traffic, before you serve it.
   The guard stores the center (the average vector) and the spread
   (how much each position in the vector varies).
2. **Score a new input.** The guard measures how far the new vector
   is from the center, in units of the spread. Close inputs get a low
   score. Far inputs get a high score.
3. **Compare to the limit.** A score at or below the limit passes.
   A score above the limit is out of range. The route is empty.

Two more rules make the guard safe by construction:

- **It fails closed.** If the guard was never fit, it stops every
  input. An unguarded path is not a safety path.
- **No input, no answer.** If a guard is attached to a route but the
  input is missing, the route is empty.

**A small picture.** The reference range is like a circle around the
known inputs:

```text
        .  .     input inside  -> route proceeds
      .        .
     .   center .
      .        .
        .  .
            x   input outside -> empty route, escalate
```

**The three guards.** The system has three guards, and they work at
different points:

1. The input guard — stops inputs outside the known range (this
   section).
2. The safety-tag guard — removes arms that lack measured safety
   coverage for a safety-flagged task (section 5.1).
3. The freeze — stops all routing and admission in an emergency, for
   a limited time (section 5.3).

The input guard is the first line. The tag guard is the second. The
freeze is the emergency stop.

**How to use it in practice.**

1. Collect real inputs from your traffic before you serve it. This is
   the reference set.
2. Fit the guard on that set.
3. Pick the limit. Start with the default. If the guard stops too
   many good inputs, raise the limit a little and re-test. If bad
   inputs pass, lower it.
4. Attach the guard to every route call. Never serve traffic without
   it.

### 5.3 Stop the system in an emergency

A freeze stops all routing and admission. It needs attestation from
several operators. It expires by itself. A freeze can never be
permanent.

```bash
geode freeze --attest op1,op2 --ttl 1000 --reason "incident"
```

### 5.4 Record a human intervention

If an operator changes a decision by hand, the system records the
change. The record must have a reason and the answer the system would
have given.

```bash
geode override --actor op1 --action kill_switch \
    --justification "power failure" \
    --counterfactual '{"would_have": "route to arm A"}'
```

The system rejects a record with an empty reason.

### 5.5 Verify evidence

To check a set of evidence files:

```bash
geode verify --evidence <directory>
```

The command prints the evidence chain and the content hash.

## 6. What can go wrong

Use this table when something fails.

| Problem                | What you see                       | What to do                                    |
| ---------------------- | ---------------------------------- | --------------------------------------------- |
| Wrong file             | The artifact check fails           | Use the correct hash. Never bypass the check. |
| Corrupt data           | The fit anchor does not match      | Stop. Rebuild the data from the sealed copy.  |
| Bad input              | The route is empty                 | Escalate the request. Do not force an answer. |
| Arm drift              | The arm score moves past its bound | Gate the arm update. Re-measure the arm.      |
| Service down           | The health check fails             | Restore from the snapshot. Check the logs.    |
| Chain broken           | The ledger verify fails            | Stop writes. Restore the last good snapshot.  |
| Latency up             | The metrics show high p99          | Check the GPU. Check the artifact store.      |
| Unauthorized change    | A record has no reason             | The system rejects it. Review the ledger.     |
| Dispute about a number | Two operators disagree             | Use the dispute proof. The wrong side pays.   |

## 7. Safety rules

These rules are always on:

1. Only measured facts move the system. Declared facts never count.
2. Every number must reproduce from its hash. If not, it is void.
3. Safety-flagged tasks never use unvetted arms.
4. Out-of-range inputs get an empty route.
5. A freeze expires. No one can stop the system forever.
6. Every human intervention is recorded with a reason.
7. Several independent verifiers must agree before a fact is used.

## 8. Final checklist

Before you open traffic:

- [ ] The package installs and `geode version` works.
- [ ] All artifact references match their hashes.
- [ ] The registry uses measured numbers only.
- [ ] The out-of-range guard is fit on the real input data.
- [ ] The health check returns `ok`.
- [ ] One freeze drill ran and expired.
- [ ] One override drill ran and was recorded.
- [ ] The ledger verifies clean.
