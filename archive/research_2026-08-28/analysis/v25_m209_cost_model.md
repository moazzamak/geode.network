# M209 — The privacy cost envelope (cost model)

Registered 19 Aug 2026 (§4.15, user constraint: privacy must not make
GEODE materially more expensive than company datacenters; assumed
willingness-to-pay is a 10–20% premium, definitely not 5×). This
document records the per-query cost accounting the pricing study
(M186) and the M195 trigger must respect.

## 1. Base serving cost — GEODE is cheap in absolute terms

The sealed additive recipe is 175.2M MACs per query (0.2786 accuracy)
on 32×32 inputs. At ~1 GFLOP/s per CPU core or ~10 TFLOPs/s on a
consumer GPU, one query is microseconds of compute — fractions of a
cent even at poor utilization. A company datacenter serving a
CLIP/ViT-class vision model spends 25–100× more compute per query.
The premium multiple applies to a SMALL base.

## 2. The privacy stack, priced per query

| Component                                 | Overhead                      | Notes                                                                                         |
| ----------------------------------------- | ----------------------------- | --------------------------------------------------------------------------------------------- |
| TLS 1.3 + AES-256-GCM payload             | <1%                           | symmetric crypto is free at this scale                                                        |
| Hybrid X25519 + ML-KEM-1024 handshake     | ~0 (amortized)                | once per session; µs-scale                                                                    |
| Local encode (P1 stage 0)                 | ~0                            | the architecture's load-bearing choice                                                        |
| MPC over the linear head (P1 stage 1)     | ~1.1–1.2× on the head portion | head = 53,627×345 matmul, tiny; round trips add ms latency                                    |
| Redundant sampling (ρ = 0.05)             | ×1.05 serving cost            | poisoning detection (M201)                                                                    |
| Liveness probes                           | ~0                            | cheap, registered (M201)                                                                      |
| Ledger/attribution + L2 settlement        | cents per batch               | never per session                                                                             |
| Dev fund 2.5% + validator 5%              | 7.5% of revenue               | a share, not a cost multiple                                                                  |
| **Full-encoder FHE/MPC (M195, deferred)** | **10–1000×**                  | Iron (NeurIPS'22), BOLT (HPCA'24), CryptoNets measurements; convnets only, transformers worse |

## 3. Where the real multiples live (not crypto)

1. **Utilization gap.** Decentralized hosts run at 30–60% GPU
   utilization vs 80–90% in a hyperscaler — up to 2–3× per unit of
   compute.
2. **Failover spare capacity.** Ordered failover chains (H8) need
   1.2–1.5× provisioned capacity to absorb downtime.
3. **Orchestration.** Routing, selection, and attribution bookkeeping
   — bounded, amortized per session, not per query.

Default-path estimate: 1.05 (sampling) × 1.2–1.5 (utilization/failover)
× ~1.02 (crypto) ≈ **1.3–1.9× the plaintext datacenter reference** —
inside the user's tolerance at comparable scale, and the gap shrinks
with scale (utilization is the lever, not cryptography).

## 4. The honest market boundary

- Users who need privacy are not choosing between GEODE and plaintext
  datacenters; they are choosing between GEODE and on-prem or
  nothing. The premium is nevertheless priced against the PLAINTEXT
  reference, per the user's constraint.
- The 5×-killer scenario is ONLY the full-encoder FHE/MPC default —
  which is why it is deferred behind the M195 trigger.

## 5. Registered gates (M209)

- **Default-path cost gate:** measured end-to-end cost per query
  ≤ 1.2× the registered reference datacenter cost at a registered
  scale (a measurable pass/fail, run before any launch).
- **M195 trigger gate:** the private-encoder tier may be built only
  if it measures ≤ 10× its plaintext path AND has registered demand
  evidence.
- **Pricing coupling (M186):** the posted-price band must be set
  against reference datacenter unit prices, not against GEODE's own
  costs.
