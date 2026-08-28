# M176c remote runbook — better-code arm on a rented NVIDIA GPU

User approved the ≤ $300 budget (17 Aug 2026). RunPod account created
and funded ($200 added by the user). This runbook is the frozen plan;
the driver script ships separately and is registered before any run.

## Pod stability playbook (registered 18 Aug 2026, after the pivot local)

The rental's failures were diagnosed and are avoidable:

1. **The fatal kill was HOST-level, not ours**: the pod's own cgroup
   never OOM'd (max_usage 150 MB, oom_kill=0) while the shared host
   showed ~12 GB free of 124 GB. NOTE (user, 18 Aug): the pod was
   already on RunPod SECURE cloud — so even "secure" hosts see
   host-level kills; the levers that remain are (a) `free -h` at pod
   start and re-roll for a quieter host, (b) a bigger RAM tier, and
   (c) the retry loop. Local remains the reliable option at our
   scale.
2. **Bigger RAM tier** (100 GB) makes the ~40 GB peak a smaller
   target for host-level kills.
3. **Retry loop**: any multi-hour job runs inside the registered
   3-attempt / 60 s-backoff loop on exit 137 (the `run_anchor_chain`
   pattern).
4. **Transfers**: scp/sftp do not work through RunPod's proxy. Use
   `runpodctl send-file` (API key stays on the user's machine) or
   Jupyter drag-drop. Best: a **read-only GitHub deploy key** for
   `moazzamak/geode-ml` with its private half pasted onto the pod by
   the user themselves — then the pod clones/pulls directly and no
   manual uploads are ever needed.
5. The pod's injected `RUNPOD_API_KEY` enables the self-stop call;
   always end unattended work with the GraphQL podStop.

## Pod sizing (registered 17 Aug 2026)

**Root cause of the whole rental saga (user, 18 Aug 2026):** when
creating the pod, RunPod exposes additional RAM/HDD spec fields that
**default to 30 GB disk / 50 GB RAM** — the user had never noticed
them. The pod was therefore running on the defaults, but the
container's `free`/`df` readings did not reflect those fields (the
container sees something else), so the mismatch was invisible from
inside the pod. Lesson: set the RAM/HDD spec fields explicitly at pod
creation and verify `free -h`/`df -h` inside before any run; do not
trust the container's readings alone.

**A single RTX 4090 (24 GB). No cluster, no multi-GPU pod.** The
candidate ladder is sequential with closed-form fits; multi-GPU would
only parallelize candidates, complicating the per-candidate budget
discipline for no measured gain.

GPU-choice rule (registered 17 Aug 2026, after the A6000-vs-4090
question): the workload is DINOv2 inference-bound, and the 4090 is
≈2x the A6000's FP32/tensor throughput. Take the **4090 unless the
A6000's hourly rate is less than half the 4090's** (the metric is
cost per work done, not cost per hour). An A6000 is acceptable in any
case — it roughly doubles the wall-clock estimates below and still
lands ≈ $10-30, far under budget. 48 GB VRAM is NOT needed at
DINOv2-S/base@224 (24 GB is ample); if a later gate fires for
DINOv2-large@518, that cell gets its own pod.

Staged wall-clock estimate on one
4090: setup + data ≈ 0.5-1 h; anchor reproduction ≈ 1-2 h; candidate
1 (deep-patch SPM) ≈ 2-4 h; candidate 2 (Fisher) ≈ 3-6 h; candidate 3
(small encoder, if reached) ≈ 6-12 h. Total ≈ **12-24 h ≈ $4-18** at
$0.34-0.74/h (double these hours on an A6000). The $200 funded
balance is ~10x headroom: do not spend
it — stop the pod between stages and when the ladder completes, and
keep the balance for follow-up cells (higher resolution or
DINOv2-base/large variants only if a gate fires).

## Disk plan (registered 17 Aug 2026 — the pod has 80 GB)

**80 GB is workable; no volume needed.** The constraint that makes it
work: the sealed code array is (409,832 × 40,383) float32 = 66.2 GB —
it must NEVER be persisted on the rental. Rules:

1. **Anchor reproduction streams in chunks.** The anchor fit needs
   X'X (40,383×40,383 ≈ 6.5 GB), X'y, and label stats — all
   accumulated chunk-wise over the re-encoded rows (the M151 chunked-
   Gram precedent), then discarded. The 66 GB codes file is never
   written; peak transient ≈ 10 GB.
2. **Deep-patch encodes stream too.** decode → DINOv2 → SPM aggregate
   → discard the patch tensors (persisting DINOv2-S@224 patch codes
   for 410k images would be ≈ 160 GB). Only the aggregated codes
   (≈ 3-20 GB depending on atom count) persist.
3. **HF cache discipline.** The parquet download doubles if the
   datasets cache is not cleaned; point the cache at the same
   directory or delete it after the first stage. If the container
   disk ever fills, the parquet cache is the first deletion target.

Peak usage estimate: environment + torch ≈ 12-15 GB, parquets ≈ 17 GB,
Gram ≈ 6.5 GB, aggregated codes ≈ 3-20 GB → ≈ 45-55 GB peak, inside
80 GB with margin.

## What the rental must do

Measure the better-code candidates against the registered per-MAC gate:
**beat the sealed dense ladder per-MAC (r70 0.3118 / r98 0.4476) or
serve a task axis no frozen arm serves.** Candidate order (registered
§7): (1) deep-patch SPM — SPM bins over DINOv2 patch tokens, no
training; (2) Fisher vectors on deep patches; (3) from-scratch small
encoder (last resort; M113's learned-dictionary negative registered).
Candidates 1-2 enter as cited comparison baselines (arXiv:1603.09046,
arXiv:2012.12509, arXiv:1912.10804), not claims.

## RunPod setup (the user does this; no secrets ever reach the model)

1. **Billing**: Settings → Billing → add a payment method / funds
   (≈ $50 to start is plenty; cap the rest).
2. **SSH key** (optional but recommended): generate locally
   (`ssh-keygen -t ed25519`), add the PUBLIC key under
   Settings → SSH Public Keys. Keep the private key to yourself.
3. **Create a Pod**:
   - Template: **RunPod PyTorch** (latest).
   - GPU: **RTX 4090** (or A6000/A5000 — anything ≥ 24 GB VRAM).
   - Container disk: **100 GB** (the DomainNet codes are 61.7 GB; the
     parquet route is ~17 GB — the driver's data step chooses).
   - Region: any.
   - Start the pod; open the web terminal (Jupyter or SSH).
   - Billing note: 4090 pods bill per started hour — stop the pod
     whenever a stage is waiting on a verdict.
4. **Nothing else**. Do NOT paste any API key anywhere the model can
   see it; the box only needs the repo and the data.

## On the box (commands come from the driver script)

1. `git clone` the repo at the registered commit.
2. Data step (registered, first): pull the DomainNet parquets from the
   local F: cache or re-download from HF; the SPM dictionary/ZCA
   artifacts ship with the repo where possible.
3. Environment: install the frozen `requirements.txt` on the pod's
   PyTorch image (cu124).
4. Anchors first: reproduce the sealed DomainNet anchors (0.2605 raw
   / 0.2274 @138k, tol registered) on the rental BEFORE any candidate
   number is admissible — this proves the rental reproduces the
   sealed environment.
5. Candidates in order; each candidate's fit-and-report goes through
   the smoke → gates → evidence writer; per-MAC gate verdict recorded.
6. Download `logs/results/v24/m176c_*/evidence.json` back to the
   workspace; the local verification (hashes, artifact index, verdict
   registration in §12) happens here, not on the rental.

## Cost guardrails

- Burn-down stops at $300 or on the registered trigger; each candidate
  is budget-capped individually (60% / 25% / 15% of the burn-down).
- A candidate that fails its anchors or its per-MAC gate is VOID or a
  scoped negative — never re-run to chase the gate.
