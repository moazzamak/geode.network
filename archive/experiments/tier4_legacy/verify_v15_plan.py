"""Verify every figure quoted in RESEARCH_IMPLEMENTATION_PLAN_v15.md and
CLAIM_LEDGER_v15.md against the sealed evidence files they cite.

The v15 plan is a pre-registration, and §5.10 requires that superseded text be
contradicted in place rather than deleted. That convention makes the document
grow rather than shrink, and a growing document is exactly where quoted figures
drift away from the evidence they came from. This script exists so that drift is
caught mechanically rather than by rereading.

Three kinds of check run:

* **Figure checks** — every number the plan or the ledger quotes is recomputed
  from the sealed evidence JSON, or from arithmetic the plan states, and
  compared with a tolerance of half a unit in the last quoted place.
* **Document checks** — every registration the plan's own rules require to be
  present is matched by pattern, including the prohibitions and the kill
  switches.
* **Structural checks** — cross-references and milestone section pins, each
  with a negative control proving the checker fires.

Usage::

    python experiments/tier4/verify_v15_plan.py
    python experiments/tier4/verify_v15_plan.py --negative-control

The second form corrupts one figure at a time in a scratch copy of the plan and
requires that the corresponding check fails, then restores the plan and compares
its hash. A check that cannot fail proves nothing.

Exit code is 0 only when every check passes.
"""
import json
import pathlib
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
PLAN = ROOT / 'analysis' / 'RESEARCH_IMPLEMENTATION_PLAN_v15.md'


# --- negative control -------------------------------------------------------
# A check that cannot fail proves nothing. In this mode the script corrupts one
# figure at a time in the plan, re-runs itself against the corrupted copy, and
# requires that the corresponding check reports a failure. The plan is restored
# byte for byte and its hash compared before the script exits.
#
# Each case is (label, exact text to corrupt, replacement, substring that must
# appear in the failing output). Corruption targets are chosen to be unique in
# the document so that replacing the first occurrence hits the intended site.
NEGATIVE_CONTROLS = [
    ('2.9.1 ensemble delta',
     '| **0.6257** *(\u22120.0065)*', '| **0.6257** *(\u22120.0500)*',
     '2.9.1 ensemble delta at 32 dims'),
    ('2.9.3 learning gain',
     '| **\u22120.0116** |', '| **\u22120.0900** |',
     '2.9.3'),
    ('2.9.2 relative compute',
     '| 80.86 | 13.33\u00d7 |', '| 80.86 | 44.00\u00d7 |',
     '2.9.2 table relative cell for dinov2-large'),
    ('2.9.2 INT8 void disclosure',
     '**The dinov2-base row is void, not negative.**',
     '**The dinov2-base row is perfectly fine.**',
     '2.9.2 states base is void not negative'),
    ('2.9.3 single-seed disclosure',
     '(i) Single seed', '(i) Many seeds',
     '2.9.3 discloses single-seed status'),
    ('M103 registration',
     '### 7.9 M103 \u2014 is a grown dictionary better',
     '### 7.9 M103 \u2014 something else entirely',
     'M103 registered'),
    ('M99 second regate',
     '**Second regate.', '**Not a regate at all.',
     'M99 second regate registered'),
    ('prohibition 23',
     'Quote any \u00a72.9 figure as evidence', 'Quote anything you like',
     'prohibition 23 on 2.9 figures present'),
    ('M102 ordering deviation',
     '**Registered deviation from this table.', '**Nothing to see here.',
     'M102 ordering deviation disclosed'),
    ('dangling cross-reference',
     'through M103 (\u00a77.9)', 'through M103 (\u00a77.31)',
     'broken cross-references'),
    ('M102 refuted verdict in the ledger',
     '| verdict | **refuted** |', '| verdict | **confirmed** |',
     'ledger records the refuted verdict'),
    ('arm (d) void marking in the ledger',
     '*(\u22121.9%)* **void \u2014 see C102.2**', '*(\u22121.9%)*',
     'ledger voids the arm (d) width'),
    # --- second amendment ---
    ('2.9.4 discriminative gain',
     '| **0.5813** | **+0.0292** | 0.0078 |',
     '| **0.5813** | **+0.0900** | 0.0078 |',
     '2.9.4 quoted gain at 128 atoms'),
    ('2.9.4 seed spread',
     '| **0.6041** | **+0.0160** | 0.0053 |',
     '| **0.6041** | **+0.0160** | 0.0400 |',
     '2.9.4 quoted seed spread at 256 atoms'),
    ('2.9.4 atom-norm mechanism row',
     '| k-means centroids | 1.209 | 1.179 | **0.481** | **2.064** |',
     '| k-means centroids | 1.209 | 1.179 | **0.111** | **2.064** |',
     '2.9.4 atom-norm row present for k-means centroids'),
    ('2.9.4 training-cost disclosure',
     'a *training* cost the random arm does not pay at all',
     'a cost that is not worth mentioning',
     '2.9.4 discloses the unpaid training cost of selection'),
    ('2.9.4 pool-artefact limitation',
     'its advantage may be an artefact of pool size',
     'its advantage is definitely real',
     '2.9.4 registers the candidate-pool artefact limitation'),
    ('2.9.5 re-verified Thiry figure',
     '| 2K patches, linear head | **0.8232** |',
     '| 2K patches, linear head | **0.8690** |',
     '2.9.5 quotes the re-verified figure for 2K patches, linear head'),
    ('8.5 unconfirmed marking on the Thiry bar',
     '**0.869** *(unconfirmed \u2014 \u00a72.9.5)*', '**0.869**',
     '8.5 Thiry bar marked unconfirmed'),
    ('2.9.3 Thiry prior-art disclosure',
     'or even a learning procedure', 'or even something else',
     '2.9.3 discloses Thiry prior art for random-beats-learned'),
    ('M103 prior reversal',
     '**[recorded after execution \u2014 the prior on this milestone has',
     '**[a paragraph that says nothing in particular and the prior has',
     'M103 prior reversal recorded in place'),
    ('M103 candidate-pool matching restriction',
     '6. **Candidate-pool matching.', '6. **Something unrelated.',
     'M103 candidate-pool matching restriction registered'),
    ('10.2 prediction contradicted in place',
     'the prediction in the paragraph above did not',
     'the prediction in the paragraph above certainly did',
     '10.2 gate-loosening prediction contradicted in place'),
    ('second amendment in the status header',
     '**Second amendment, recorded after the \u00a72.9 audit.**',
     '**Nothing further was recorded.**',
     'status header records the second amendment'),
    ('2.9.4 gain-to-spread band in the prose',
     'by 3.0\u20133.7\u00d7 the random arm', 'by 9.0\u20139.7\u00d7 the random arm',
     '2.9.4 lower end of the quoted band'),
    # --- third amendment ---
    ('third amendment in the status header',
     '**Third amendment, recorded after the M103 instrumentation run.**',
     '**Nothing further was recorded after that.**',
     'status header records the third amendment'),
    ('2.9.6 section header',
     '### 2.9.6 The M103 instrumentation run',
     '### 2.9.6 An unrelated aside',
     '2.9.6 registered'),
    ('2.9.6 discriminative accuracy at 1024 atoms',
     '| (c) discriminative selection | **0.6951** |',
     '| (c) discriminative selection | **0.9951** |',
     '2.9.6 table row present for (c) discriminative selection'),
    ('2.9.6 k-means accuracy at 1024 atoms',
     '| (b) k-means | 0.6856 |', '| (b) k-means | 0.5856 |',
     '2.9.6 table row present for (b) k-means'),
    ('2.9.6 gain quoted in the prose',
     '**+0.0133** over random', '**+0.9133** over random',
     '2.9.6 quotes the gain in prose'),
    ('2.9.6 discriminative atom-norm row',
     '| (c) discriminative | **1.734** | 1.683 | **1.032** | 2.683 |',
     '| (c) discriminative | **1.734** | 1.683 | **0.102** | 2.683 |',
     '2.9.6 atom-norm row present for (c) discriminative'),
    ('2.9.6 instrumentation-run disclaimer',
     'It is an instrumentation run,', 'It is a sealed milestone result,',
     '2.9.6 declares itself an instrumentation run, not a milestone'),
    ('2.9.6 single-seed disclosure',
     'single seed, single budget, unsealed, inadmissible',
     'many seeds, many budgets, sealed, admissible',
     '2.9.6 discloses single seed and single budget'),
    ('2.9.6 record of the 2.9.3 reversal',
     "**Second \u2014 \u00a72.9.3's ordering reverses at full scale.**",
     '**Second \u2014 everything held up nicely.**',
     '2.9.6 records the 2.9.3 reversal'),
    ('2.9.6 record of the instrument-check defect',
     'registered instrument check in \u00a77.9 design item 4 cannot be',
     'registered instrument check in \u00a77.9 design item 4 can readily be',
     '2.9.6 records the instrument-check defect'),
    ('2.9.6 statement of the feature-count mismatch',
     'Coates anchor is **0.796 at 4000 features**',
     'Coates anchor is **0.796 at 1024 features**',
     '2.9.6 names the 4000-feature mismatch'),
    ('7.9 in-place correction marker',
     '**[corrected after execution \u2014 this check as written cannot be '
     'satisfied by',
     '**[quietly rewritten because the old one was inconvenient and cannot be '
     'satisfied by',
     '7.9 item 4 corrected in place rather than deleted'),
    ('7.9 monotonicity condition',
     "* **(i) Monotonicity.** Arm (b)'s accuracy must rise with atom count",
     "* **(i) Vibes.** Arm (b)'s accuracy must feel about right",
     '7.9 corrected check registers monotonicity'),
    ('7.9 internal floor condition',
     "* **(ii) A floor set by this program's own weaker instrument.**",
     '* **(ii) A floor set by a paper we did not read.**',
     '7.9 corrected check registers the internal floor'),
    ('7.9 anchor demotion',
     '* **(iv) The Coates figure is reported beside the curve as an anchor**',
     '* **(iv) The Coates figure decides whether M103 is read at all**',
     '7.9 corrected check demotes Coates to a non-gating anchor'),
    ('prohibition 23 extension to 2.9.6',
     "\u00a72.9.6's figures are **single-seed**",
     "\u00a72.9.6's figures are **fully replicated**",
     'prohibition 23 extended to 2.9.6'),
    # --- fourth amendment: the M103 ledger entry ---
    ('M103 ledger confirmed verdict',
     '| verdict | **confirmed** |', '| verdict | **refuted** |',
     'M103 ledger states the confirmed verdict'),
    ('M103 ledger accuracy row at 512 atoms',
     '| 512 | 0.6720 | 0.6705 | **0.6908** | 0.6476 | **+0.0188** |',
     '| 512 | 0.6720 | 0.6705 | **0.7908** | 0.6476 | **+0.0188** |',
     'M103 ledger row present at 512 atoms'),
    ('M103 ledger accuracy row at 1024 atoms',
     '| 1024 | 0.6879 | 0.6839 | **0.6983** | 0.6684 | **+0.0104** |',
     '| 1024 | 0.6879 | 0.6839 | **0.6983** | 0.6684 | **+0.0904** |',
     'M103 ledger row present at 1024 atoms'),
    ('M103 ledger Q2 chain marking',
     '**Q2** (efficiency), on the M103 \u2192 M99 chain',
     '**Q1** (the frontier point), on the M103 \u2192 M99 chain',
     'M103 ledger states it is a Q2 milestone'),
    ('M103 ledger seed-11 pair',
     '0.6862 vs 0.6818 (seed 11)', '0.6862 vs 0.6118 (seed 11)',
     'M103 ledger quotes the seed 11 pair'),
    ('M103 ledger 15-of-15 claim',
     'Arm (c) beat arm (a) in **15 of 15**',
     'Arm (c) beat arm (a) in **9 of 15**',
     'M103 ledger quotes 15 of 15'),
    ('M103 ledger 1024-atom ratio row',
     '| 1024 | +0.0104 | 0.0101 | **1.03** |',
     '| 1024 | +0.0104 | 0.0101 | **9.03** |',
     'M103 ledger quotes the 1024-atom ratio row'),
    ('M103 ledger narrowing-margin restriction',
     '**C103.3 travels with C103.1.**', '**C103.3 is optional.**',
     'M103 ledger binds C103.3 to C103.1'),
    ('M103 ledger convergence disclosure',
     'did not converge, and all five are in the arms that lose',
     'converged everywhere, so there is nothing at all to report',
     'M103 ledger discloses the convergence defect'),
    ('M103 ledger single-seed-artifact record',
     'single-seed artifact', 'fully replicated finding',
     'M103 ledger records the single-seed artifact'),
    ('M103 ledger arm (a) inference total',
     '| **89,165,584** |', '| **12,165,584** |',
     'M103 ledger quotes the arm (a) inference total'),
    ('M103 ledger break-even figure',
     'stated plainly: 132,244 inferences', 'stated plainly: 12 inferences',
     'M103 ledger quotes the break-even figure'),
    ('M103 ledger anchor demotion',
     'reported here as an **anchor and not an',
     'reported here as the deciding **operand and not an',
     'M103 ledger states the anchor is not an operand'),
    ('M103 ledger void-rung marking',
     'The **2048 rung is void, not negative**',
     'The **2048 rung is our best result**',
     'M103 ledger records the void rung as not read'),
    ('M103 ledger novelty disclaimer',
     'not a new idea', 'an entirely new idea',
     'M103 ledger disclaims novelty'),
    ('M103 ledger corpus-mixing restriction',
     '**No CIFAR-to-DomainNet comparison**',
     '**Comparison with DomainNet is fine**',
     'M103 ledger forbids CIFAR/DomainNet comparison'),
    ('plan 2.9.6 finding 2 contradiction',
     '**[contradicted after execution \u2014 this finding was single-seed and did',
     '**[this finding stands entirely unchallenged and did',
     'plan 2.9.6 finding 2 is contradicted in place'),
    # --- fifth amendment: the prior-art audit and the M104-M106 pivot ---
    ('fifth amendment in the status header',
     '**Fifth amendment, recorded after a prior-art audit of M103.**',
     '**Nothing at all happened after M103.**',
     'status header records the fifth amendment'),
    ('fifth amendment statement that the audit went against M103',
     'and **the audit went against', 'and **the audit went entirely for',
     'fifth amendment states the audit went against M103'),
    ('2.9.7 section header',
     '### 2.9.7 Effective rank as a sizing instrument',
     '### 2.9.7 An unrelated digression',
     '2.9.7 registered'),
    ('2.9.7 probe 1 row at 64 atoms',
     '| 64 | 256 | 37.194 | 0.14529 |',
     '| 64 | 256 | 97.194 | 0.14529 |',
     '2.9.7 probe 1 quotes the 64-atom row'),
    ('2.9.7 probe 1 row at 2048 atoms',
     '| 2048 | 8192 | 177.822 | 0.02171 |',
     '| 2048 | 8192 | 177.822 | 0.92171 |',
     '2.9.7 probe 1 quotes the 2048-atom row'),
    ('2.9.7 probe 1 rank exponent',
     'exponent on `log rank` versus `log atoms` is **0.4553**',
     'exponent on `log rank` versus `log atoms` is **0.9553**',
     '2.9.7 quotes the rank exponent'),
    ('2.9.7 probe 1 doubling band',
     'Rank rises by \u00d71.319\u20131.475 per doubling',
     'Rank rises by \u00d79.319\u20139.475 per doubling',
     '2.9.7 quotes the per-doubling band'),
    ('2.9.7 probe 1 collapse factor',
     '**6.69\u00d7** across the sweep', '**1.69\u00d7** across the sweep',
     '2.9.7 quotes the collapse factor'),
    ('2.9.7 probe 1 refutation record',
     '**This refutes the hypothesis that rank saturation',
     '**This confirms the hypothesis that rank saturation',
     '2.9.7 records that probe 1 refuted the saturation hypothesis'),
    ('2.9.7 probe 2 class specialisation ratio',
     'a ratio of **0.9107**', 'a ratio of **0.1907**',
     '2.9.7 quotes the class specialisation ratio'),
    ('2.9.7 probe 3 quickdraw row',
     '| **quickdraw** | **8.752** | **0.185** |',
     '| **quickdraw** | **48.752** | **0.185** |',
     '2.9.7 quotes the quickdraw row'),
    ('2.9.7 probe 3 domain spread',
     'infograph to quickdraw is **6.32\u00d7**',
     'infograph to quickdraw is **1.32\u00d7**',
     '2.9.7 quotes the domain spread'),
    ('2.9.7 probe 3 mean-statistic disclaimer',
     '**which conceals the finding entirely**',
     '**which states the finding perfectly well**',
     '2.9.7 disowns the mean as the wrong statistic'),
    ('2.9.7 probe 3 resolution limitation',
     'no milestone may quote 6.32', 'any milestone may quote 6.32',
     '2.9.7 forbids quoting the spread as resolution-independent'),
    ('2.9.7 probe 4 intrinsic router accuracy',
     '**4 intrinsic scalars \u2192', '**9 intrinsic scalars \u2192',
     '2.9.7 quotes both router accuracies'),
    ('2.9.7 probe 4 quickdraw alpha row',
     '| quickdraw | 30.76 | **\u22121.247** | 0.972 | **0.693** |',
     '| quickdraw | 30.76 | **\u22124.247** | 0.972 | **0.693** |',
     '2.9.7 quotes the quickdraw alpha row'),
    ('2.9.7 probe 4 drift table',
     '| intrinsic fingerprints, 4-d | 0.96664 | **3.336%** |',
     '| intrinsic fingerprints, 4-d | 0.96664 | **9.336%** |',
     '2.9.7 quotes the intrinsic drift row'),
    ('2.9.7 probe 4 stability advantage',
     'is **2.80\u00d7 more stable**', 'is **9.80\u00d7 more stable**',
     '2.9.7 quotes the stability advantage'),
    ('2.9.7 within-versus-across-image distinction',
     '**Expert sizing is set by the', '**Expert sizing is set by no',
     '2.9.7 records the within-versus-across-image distinction'),
    ('2.9.7 statement that the probes measure no accuracy',
     'None of them measures accuracy', 'All of them measure accuracy',
     '2.9.7 states the probes measure no accuracy'),
    ('M104 section header',
     '### 7.10 M104 \u2014 does sizing an expert to its sub-population',
     '### 7.10 M104 \u2014 something else altogether',
     'M104 registered'),
    ('M104 structure-matched null',
     'the structure-matched null R5 requires**',
     'an optional extra that we may skip**',
     'M104 registers the structure-matched random-partition null'),
    ('M104 oracle-routing restriction',
     '**M104 therefore measures an upper', '**M104 therefore measures a lower',
     'M104 declares its routing an oracle upper bound'),
    ('M104 mechanism-failure clause',
     '**If the margin is uniform across all six domains, the stated mechanism',
     '**If the margin is uniform across all six domains, the stated conclusion',
     'M104 registers that a uniform margin refutes its mechanism'),
    ('M104 kill switch 2',
     '**Kill switch 2.** If arm (d)', '**A curiosity.** If arm (d)',
     'M104 kill switch 2 registered'),
    ('M104 per-expert sample floor',
     "expert, on that expert's own rows**", 'corpus, on all its rows at once**',
     'M104 binds the sample floor per expert'),
    ('M105 section header',
     '### 7.11 M105 \u2014 does the intrinsic router survive contact',
     '### 7.11 M105 \u2014 an unrelated matter',
     'M105 registered'),
    ('M105 routing-tax binding',
     '**The routing tax is reported with the headline.**',
     '**The routing tax may be omitted if inconvenient.**',
     'M105 binds the routing tax to its headline'),
    ('M105 random-routing null',
     '**(d) Random routing** \u2014 **the null**',
     '**(d) Random routing** \u2014 **an afterthought**',
     'M105 registers the random-routing null'),
    ('M106 section header',
     '### 7.12 M106 \u2014 does the construction actually compose additively',
     '### 7.12 M106 \u2014 a different question entirely',
     'M106 registered'),
    ('M106 no-refit procedure',
     '**without refitting the first four experts and without re-measuring',
     '**while refitting the first four experts and re-measuring',
     'M106 forbids refitting earlier experts'),
    ('M106 quadratic failure mode',
     'total construction cost is **O(K\u00b2)**',
     'total construction cost is **O(K)**',
     'M106 names the O(K squared) failure mode'),
    ('M106 continual-learning prohibition',
     '**no M106 figure may be described as demonstrating',
     '**every M106 figure may be described as demonstrating',
     'M106 forbids continual-learning language'),
    ('7.13 dense-comparator defect',
     '**Neither compared anything to a dense network**',
     '**Both compared themselves to dense networks**',
     '7.13 records the missing dense comparator'),
    ('6.1 P2 in-place correction marker',
     '**[corrected in place after the M103 prior-art audit. Retained per ',
     '**[silently replaced because it was inconvenient. Not retained per ',
     '6.1 P2 hardness citation corrected in place'),
    ('8.10 section header',
     '### 8.10 Effective rank, random features and mixtures of experts',
     '### 8.10 A short note on nothing in particular',
     '8.10 registered'),
    ('8.10.1 matching lower bound',
     ', **with a matching', ', **with no matching',
     '8.10.1 records the matching lower bound'),
    ('8.10.1 prediction of C103.3',
     "**C103.3's narrowing margin is *predicted*, not merely volunteered.**",
     "**C103.3's narrowing margin was a complete surprise to everyone.**",
     '8.10.1 records that C103.3 was predicted by the theory'),
    ('8.10.1 label-free reading',
     '**The strongest published results are label-free.**',
     '**The strongest published results all use labels.**',
     '8.10.1 records that the published mechanism is label-free'),
    ('8.10.2 RankMe attribution',
     '\u00a72.9.7 uses, unmodified.**', '\u00a72.9.7 built from scratch.**',
     '8.10.2 attributes RankMe to its authors'),
    ('8.10.3 prohibition on rebuilding constructive networks',
     '**This program may not build a constructive',
     '**This program should now build a constructive',
     '8.10.3 records that constructive white-box networks already exist'),
    ('8.10.4 uniform expert sizing in the literature',
     '  remain **uniformly sized**.', '  are **individually sized**.',
     '8.10.4 records that published experts are uniformly sized'),
    ('8.10.4 search-failure framing',
     'disclosure under \u00a78.6 and not a novelty claim**',
     'disclosure under \u00a78.6 and also a novelty claim**',
     '8.10.4 states the search failure without claiming novelty'),
    ('8.10.4 correct reading of a failed search',
     '*"this program did not find it"*, never *"it does not exist."*',
     '*"it does not exist"*, and that is that.',
     '8.10.4 states the correct reading of a failed search'),
    ('8.10.5 Blum & Rivest correction',
     '**That result is about *training* a',
     '**That result is exactly about routing and not *training* a',
     '8.10.5 corrects the Blum & Rivest misuse'),
    ('8.10.5 closed-set versus open-set separation',
     'measured a **closed-set** domain probe at',
     'measured a **open-set** domain probe at',
     '8.10.5 separates closed-set routing from open-set rejection'),
    ('prohibition 25',
     '25. **Present effective-rank measurement, constructive white-box networks',
     '25. **Feel free to claim effective-rank measurement and white-box networks',
     "prohibition 25 on presenting borrowed instruments as this program's"),
    ('prohibition 26',
     '26. **State C103.1 without its prior art',
     '26. **State C103.1 however you like',
     'prohibition 26 binds the C103.1 prior-art disclosure'),
    ('prohibition 27',
     '27. **Claim an efficiency result against dense networks from M104, M105 or '
     'M106',
     '27. **Claim whatever efficiency result you like from M104, M105 or M106',
     'prohibition 27 forbids a dense-network efficiency claim from M104-M106'),
    ('M103 ledger prior-art amendment',
     '**The phenomenon C103.1 measures is already published, in a stronger form',
     '**The phenomenon C103.1 measures has never been published, in any form',
     'M103 ledger discloses the prior art that subsumes C103.1'),
    ('M103 ledger prior-art restriction',
     '8. **The C103.1 prior-art disclosure travels with C103.1.',
     '8. **The C103.1 prior-art disclosure is entirely optional.',
     'M103 ledger binds the prior-art disclosure to C103.1'),
    ('M103 ledger record that no figure is withdrawn',
     '**What does not change.** No figure in this entry is withdrawn',
     '**What changes.** Every figure in this entry is withdrawn',
     'M103 ledger states no figure is withdrawn by the audit'),
    ('ledger records M104-M106 as registered and not run',
     '(\u00a77.12) are registered and **not yet run**',
     '(\u00a77.12) have all run and been confirmed**',
     'ledger records M104-M106 as registered but not run'),
    # --- M104 execution-time amendments -----------------------------------
    ('7.10 amendment block header',
     '**Execution-time amendments, registered before any M104 figure was '
     'computed.**',
     '**Execution-time amendments, added once the M104 results were in.**',
     '7.10 registers its execution-time amendments before measurement'),
    ('7.10 amendments predate the run',
     '**before the sealed run was started**',
     '**after the sealed run had finished**',
     '7.10 states the amendments were made before the run started'),
    ('7.10 amendment direction',
     'Each is written so that a reader can see whether it makes the milestone',
     'Each is written so that a reader cannot see whether it makes the milestone',
     '7.10 states every amendment makes the milestone harder'),
    ('7.10 row-weighted MAC derivation',
     '`\u03a3_e f_e\u00b7A_e`, where `f_e` is domain *e*\'s share of rows.',
     '`\u03a3_e A_e`, where `f_e` is domain *e*\'s share of rows.',
     '7.10 derives the MAC match as the row-weighted atom sum'),
    ('7.10 parameter excess is reported not matched',
     '**reported rather than matched away**',
     '**matched away rather than reported**',
     '7.10 reports the parameter excess rather than matching it away'),
    ('7.10 design item 2 governs',
     'GOVERNS.]**',
     'YIELDS.]**',
     '7.10 marks design item 2 as governing over design item 1'),
    ('7.10 both generalists are run',
     '**Both are run**: arm (c1) at the',
     '**Only one is run**: arm (c1) at the',
     '7.10 runs both generalists rather than choosing one'),
    ('7.10 kill switch 4 registration',
     '**Kill switch 4.** **[added before execution, with arm (e).]**',
     '**Kill switch 4.** **[added after execution, with arm (e).]**',
     '7.10 registers kill switch 4'),
    ('7.10 kill switch 4 arbitrage form',
     '**traffic-weighted MAC arbitrage**',
     '**a genuine effective-rank effect**',
     '7.10 states kill switch 4 in the arbitrage form'),
    ('7.10 per-class reading binds every arm equally',
     'it is met by no arm *equally*',
     'it is met by arm (b) alone',
     '7.10 states the per-class reading binds every arm equally'),
    ('7.10 cap binds against the nulls',
     '**The floor therefore makes',
     '**The floor therefore fails to make',
     '7.10 records the sample-floor cap as binding against the nulls'),
    ('7.10 restriction 7 head change',
     '7. **The head is a multi-output ridge, and that is a change from M103.**',
     '7. **The head is unchanged from M103 in every respect.**',
     '7.10 restriction 7 records the head change from M103'),
    ('7.10 restriction 7 constant chosen on the null',
     '**once, on the null arm (a), at the first seed**',
     '**once, on the treatment arm (b), at the first seed**',
     '7.10 restriction 7 states the constant is chosen on the null arm'),
    ('7.10 uniform atom spend figure',
     '**3,072**',
     '**3,071**',
     '7.10 quotes the 3072-atom uniform spend'),
    ('7.10 rank-sized atom spend figure',
     '**3,455**',
     '**3,456**',
     '7.10 quotes the 3455-atom rank-sized spend'),
    ('7.10 parameter excess percentage',
     '12.5% more parameters',
     '12.6% more parameters',
     '7.10 quotes the 12.5 percent parameter excess'),
    ('7.10 domain size ratio',
     'differ in size by **3.6\u00d7**',
     'differ in size by **3.7\u00d7**',
     '7.10 quotes the 3.6x domain size ratio'),
    ('7.10 quickdraw traffic share',
     '**29.46%**',
     '**29.47%**',
     '7.10 quotes quickdraw at 29.46 percent of rows'),
    ('7.10 clipart cap figure',
     'clipart at **838** atoms',
     'clipart at **839** atoms',
     '7.10 quotes the clipart cap of 838'),
    ('7.10 infograph cap figure',
     'infograph at **900**',
     'infograph at **901**',
     '7.10 quotes the infograph cap of 900'),
    ('7.10 corpus row counts',
     '`real` holds 120,906 train rows, `clipart` 33,525',
     '`real` holds 120,907 train rows, `clipart` 33,525',
     '7.10 quotes the real and clipart row counts'),
    ('ledger M104 amendment block',
     '**M104 execution-time amendments, recorded before the sealed run '
     'started.**',
     '**M104 execution-time amendments, recorded after the sealed run '
     'finished.**',
     'ledger records the M104 execution-time amendments'),
    ('ledger M104 amendments predate any figure',
     '**in place, before any M104 figure existed**',
     '**in place, once the first M104 figures were in**',
     'ledger states the amendments predate the run'),
    ('ledger M104 arm (e) and kill switch 4',
     '**arm (e)** and **kill switch 4** are added',
     '**arm (e)** and **kill switch 4** were considered and dropped',
     'ledger names arm (e) and kill switch 4'),
    ('ledger M104 head change cross-reference',
     '\u00a77.10 restriction 7 records the head change from M103',
     '\u00a77.10 restriction 7 records that the head is unchanged from M103',
     'ledger records the head change and its restriction'),
    ('7.10 amendment 5 refits on every row',
     '5. **The reported model is refitted on every row the expert owns, and the',
     '5. **The reported model is fitted on the 90% left after validation, and the',
     '7.10 amendment 5 refits the reported model on every row'),
    ('7.10 amendment 5 names the defect the smoke run exposed',
     'execution, after the smoke run exposed the defect it fixes.]**',
     'execution, on general principle and without any run behind it.]**',
     '7.10 amendment 5 attributes the defect to the smoke run'),
    ('7.10 amendment 5 records two voided experts',
     '(b)\u0027s six experts were voided anyway**. A guard that does not guard',
     '(b)\u0027s six experts stayed adequate**. A guard that does not guard',
     '7.10 amendment 5 records the two voided experts'),
    ('7.10 amendment 5 costs no extra encode',
     '**no additional encode pass and no additional memory**',
     '**one additional encode pass over the held-out rows**',
     '7.10 amendment 5 states it costs no extra encode pass'),
    ('7.10 amendment 5 makes the floor exact',
     'is now enforced **exactly** by the cap rather than',
     'is now enforced **approximately** by the cap rather than',
     '7.10 amendment 5 states the floor is now exactly enforced'),
    ('7.10 the 3,455 is an illustration not a prediction',
     '**That figure illustrates the size of the parameter excess the MAC match',
     '**That figure predicts the size of the parameter excess the MAC match',
     '7.10 marks the 3,455 as an illustration and not a prediction'),
    ('7.10 the sealed run re-measures rank',
     'consume them: it re-measures',
     'consume them: it imports',
     '7.10 states the sealed run re-measures rank rather than importing it'),
    ('7.14 registered before any M104 accuracy',
     '**[new — registered while M104 was running, before any M104 accuracy '
     'existed]**',
     '**[new — registered once M104 had finished and its accuracies were '
     'known]**',
     '7.14 records that M107 was registered before any M104 accuracy existed'),
    ('7.14 prediction is against the program thesis',
     'dominates the sparse ladder in accuracy at every MAC budget where the two',
     'is dominated by the sparse ladder at every MAC budget where the two',
     '7.14 registers a prediction against the program own thesis'),
    ('7.14 names the prediction as against interest',
     '**This prediction is against the program\u0027s',
     '**This prediction is in line with the program\u0027s',
     '7.14 marks its prediction as against interest'),
    ('7.14 kill switch 1 refutes Q2 on a dense win',
     '**\u00a73.2 Q2\u0027s efficiency claim is refuted at this',
     '**\u00a73.2 Q2\u0027s efficiency claim is merely unproven at this',
     '7.14 kill switch 1 refutes Q2 if the dense curve dominates'),
    ('7.14 kill switch 1 forbids footnoting the refutation',
     'It may not be reported as a footnote, a limitation, or future work.',
     'It may be reported as a limitation or as future work.',
     '7.14 forbids reporting kill switch 1 as a footnote'),
    ('7.14 LVD-142M caveat',
     '**LVD-142M**, 142 million curated images',
     '**LVD-142M**, 142 thousand curated images',
     '7.14 records the LVD-142M training asymmetry'),
    ('7.14 resolution asymmetry runs in dense favour',
     '**The resolution asymmetry is real, runs in dense\u0027s favour, and is '
     'measured',
     '**The resolution asymmetry is minor, runs in the sparse side\u0027s favour, '
     'and is argued',
     '7.14 records the resolution asymmetry as favouring dense'),
    ('7.14 arm d5 is the information-matched control',
     '**(d1) minus (d5) is what the extra pixels are worth;',
     '**(d1) minus (d5) is what the architecture is worth;',
     '7.14 defines arm d5 as the information-matched control'),
    ('7.14 restriction 7 binds d1 and d5 together',
     '7. **Arm (d1) and arm (d5) are reported together or not at all**',
     '7. **Arm (d1) may be reported without arm (d5)**',
     '7.14 restriction 7 binds arms d1 and d5 together'),
    ('7.14 restriction 5 forbids wall-clock comparison',
     '5. **Analytic MACs only**, per design item 4; no wall-clock comparison',
     '5. **Wall-clock timings only**, per design item 4; no analytic comparison',
     '7.14 restriction 5 forbids a wall-clock comparison between families'),
    ('7.14 head constant chosen on the sparse side',
     'chosen **once, on the sparse generalist at',
     'chosen **once, on the dense generalist at',
     '7.14 chooses the head constant on the sparse side'),
    ('7.13 records that 7.14 closes the defect',
     '**[The comparator is now designed, in \u00a77.14, and registered while M104 '
     'was still',
     '**[The comparator was designed in \u00a77.14 after M104 finished, once M104 '
     'was',
     '7.13 cross-references 7.14 as the measurement that closes it'),
    ('7.13 keeps prohibition 27 in force until 7.14 runs',
     'stays in force until \u00a77.14 has actually been run.]**',
     'is lifted as soon as \u00a77.14 has been designed.]**',
     '7.13 keeps prohibition 27 in force until 7.14 has run'),
    ('ledger M107 entry',
     '**M107 (plan \u00a77.14), registered while M104 was running and before any '
     'M104',
     '**M107 (plan \u00a77.14), registered after M104 had finished and its',
     'ledger records M107 and when it was registered'),
    ('ledger M107 is unconditional',
     '**unconditional** \u2014 it does not depend on M104\u0027s outcome',
     '**conditional** on M104 surviving all of its kill switches',
     'ledger records M107 as unconditional on M104'),
    ('7.14 amendment 1 exists',
     '1. **The resolution sweep gains 28 and 56.** Design item 2 registers',
     '1. **The resolution sweep is left as registered.** Design item 2 registers',
     '7.14 amendment 1 adds two resolutions before measurement'),
    ('7.14 amendment 1 is against interest',
     '   make the sparse side\u0027s job easier at any budget that was already',
     '   make the sparse side\u0027s job harder at every budget that was already',
     '7.14 amendment 1 states the overlap is two budgets wide'),
    ('7.14 amendment 2 voids rather than reports',
     '2. **The \u00a75.3 floor voids an arm rather than being reported beside it.**',
     '2. **The \u00a75.3 floor is reported beside each arm.**',
     '7.14 amendment 2 makes the sample floor void an arm'),
    ('7.14 amendment 2 aborts on a void selection arm',
     '   constant on is itself void, because every other arm inherits that constant.',
     '   constant on is itself void, which is reported as a warning only.',
     '7.14 amendment 2 aborts if the selection arm is itself void'),
    ('7.14 amendment 3 proves the pixel identity',
     '3. **The instrument proves, rather than asserts, that the two families see the',
     '3. **The instrument assumes, reasonably, that the two families see the',
     '7.14 amendment 3 proves the pixel identity rather than asserting it'),
    ('7.14 amendment 3 is bitwise',
     'parquet and requires them **bitwise** equal to the cached tensors; a mismatch',
     'parquet and requires them **approximately** equal to the cached tensors; a mismatch',
     '7.14 amendment 3 requires bitwise equality'),
    ('7.14 amendment 4 is not a tightening',
     '   this does not make the milestone harder**, and it is recorded as a limitation',
     '   this makes the milestone harder too**, and it is recorded as a tightening',
     '7.14 amendment 4 records the single seed as not making it harder'),
    ('ledger records the M107 amendments',
     '**M107 execution-time amendments, recorded before the run and before any M107',
     '**M107 execution-time amendments, recorded after the run and after the first M107',
     'ledger records the M107 execution-time amendments'),
    ('ledger records the M107 seed limitation',
     'does **not** make the milestone harder and is recorded as a limitation of every',
     'does **also** make the milestone harder and is recorded as a tightening of every',
     'ledger records the M107 single seed as not making it harder'),
    ('7.10 amendment 7 exists',
     '   the opposite was nearly done.** **[registered during execution.]** M104 was',
     '   the opposite was obviously wrong.** **[registered after execution.]** M104 was',
     '7.10 amendment 6 quotes the measured core occupancy'),
    ('7.10 amendment 7 protects the sealed numerics block',
     '   block for scheduling convenience is not a trade this program makes**, and the',
     '   block for scheduling convenience is a routine adjustment**, and the',
     '7.10 amendment 7 refuses to edit a sealed numerics block'),
    ('7.10 amendment 7 quotes the pinned thread count',
     '   at **16**, so running them together would place over thirty spinning threads',
     '   at **32**, so running them together would place over thirty spinning threads',
     '7.10 amendment 7 names the thread counts that forced the decision'),
    ('7.10 amendment 6 prohibition text',
     '   quoted as one.** **[registered during execution, on discovering the',
     '   quoted freely.** **[registered during execution, on discovering the',
     '7.10 amendment 6 forbids quoting the seconds field'),
    ('7.10 amendment 6 exists',
     "6. **The `seconds` field in M104's evidence is NOT an operand and may never be",
     "6. **The `seconds` field in M104's evidence is a supporting efficiency figure",
     '7.10 amendment 6 names the seconds field as not an operand'),
    ('7.10 amendment 6 keeps M104 MAC counting analytic',
     '   **analytic** \u2014 `training_macs` and `rank_measurement_macs` are counted, not',
     '   **measured** \u2014 `training_macs` and `rank_measurement_macs` are counted, not',
     '7.10 amendment 6 keeps M104 MAC counting analytic'),
    ('7.10 amendment 6 says what would have forced a rerun',
     '   wall-clock, the correct response would have been to stop and rerun it alone',
     '   wall-clock, the correct response would still have been to disclose and go on',
     '7.10 amendment 6 states what would have forced a rerun'),
    ('ledger records the M104 timing contamination',
     'evidence **is not an operand and may never be quoted as one**: the M107 pixel',
     'evidence is a supporting efficiency figure and may be quoted: the M107 pixel',
     'ledger records the M104 seconds field as not an operand'),
    ('ledger records the rejected concurrency',
     '**7.1 of 16** cores, and filling the rest with M107 was rejected because both',
     '**7.1 of 16** cores, and filling the rest with M107 was adopted because both',
     'ledger records why M107 was not run beside M104'),
    ('7.10 result headline',
     'spread. Under \u00a711.1 this is the headline of M104 and is not reportable as a',
     'spread. Under \u00a711.1 this is a minor note in M104 and is not reportable as a',
     '7.10 result: kill switch 1 is the headline, not a footnote'),
    ('7.10 result refutation',
     'switches fired. Rank-sizing is refuted.** **[written after the sealed run;',
     'switches fired. Rank-sizing is promising.** **[written after the sealed run;',
     '7.10 result: rank-sizing is refuted'),
    ('7.10 result inversion',
     '**The registered mechanism is not merely unsupported \u2014 it is inverted.** The',
     '**The registered mechanism is only partly unsupported here.** The',
     '7.10 result: the mechanism inverted rather than merely failing'),
    ('7.10 result mechanism statement',
     'is a property of the input distribution; the capacity a domain needs is a',
     'is a property of the label structure; the capacity a domain needs is a',
     '7.10 result: rank measures inputs, capacity follows labels'),
    ('7.10 result closes M105 and M106',
     'not. **M105 and M106 do not proceed.** Pursuing a router for a partition that',
     'not. **M105 and M106 proceed anyway.** Pursuing a router for a partition that',
     '7.10 result: M105 and M106 are closed, not pending'),
    ('7.10 result spares the representation',
     '**What survives.** Nothing in this refutes sparse dictionaries as a',
     '**What survives.** Nothing at all survives this, including sparse dictionaries as a',
     '7.10 result: sparse representation survives the refutation'),
    ('7.10 result quickdraw starvation',
     'rule handed it **~104 atoms**, about',
     'rule handed it **~1040 atoms**, about',
     '7.10 result: the quickdraw starvation is quantified'),
    ('ledger M104 refutation',
     '**M104 result (plan \u00a77.10). Outcome letter: R \u2014 refuted. Four of five kill',
     '**M104 result (plan \u00a77.10). Outcome letter: S \u2014 supported. Four of five kill',
     'ledger records the M104 refutation'),
    ('ledger closes M105 and M106',
     '**M105 and M106 are closed, not pending.** Both were registered conditional on',
     '**M105 and M106 remain pending.** Both were registered conditional on',
     'ledger closes M105 and M106'),
    ('ledger supersession marker',
     'three are Q2. **[superseded in place, \u00a75.10: M104 has since run to completion.',
     'three are Q2. **[still current: M104 has since run to completion.',
     'ledger marks the stale not-yet-run claim superseded'),
    ('7.14 amendment 6 exists',
     '   `logs/results/v15/m107_dense/` \u2014 the sealed directory. Nothing had read it',
     '   `logs/results/v15/m107_smoke/` \u2014 a scratch directory. Nothing had read it',
     '7.14 amendment 6 records the sealed-directory contamination'),
    ('7.14 amendment 6 guard',
     '   **refuses to start** when a config declaring itself inadmissible is pointed',
     '   **warns on startup** when a config declaring itself inadmissible is pointed',
     '7.14 amendment 6 makes the runner refuse to start'),
    ('7.14 amendment 6 evidence stamp',
     '   `admissible_as_evidence` and the `config_file` that produced it, so a reader',
     '   `admissible_as_evidence` and the `config_file` that produced it, so nobody',
     '7.14 amendment 6 stamps the evidence with its admissibility'),
    ('ledger records the M107 sealed-directory contamination',
     '**2,760-row** `evidence.json` into the **sealed** M107 directory, where the',
     '**2,760-row** `evidence.json` into a scratch M107 directory, where the',
     'ledger records the M107 sealed-directory contamination'),
    ('7.14 amendment 5 exists',
     '5. **The mixture ladder runs only at 128 and 256 atoms, and \u00a75.3 is why.**',
     '5. **The mixture ladder runs at every budget the generalist does.**',
     '7.14 amendment 5 stops the mixture ladder at 256 atoms'),
    ('7.14 result headline',
     '**+1.80 pp**. Under \u00a711.1 this',
     '**+1.80 pp**. As a minor aside this',
     '7.14 result: kill switch 2 is the headline, not a footnote'),
    ('7.14 result prediction failed',
     'prediction is refuted.** **[written after the sealed run;',
     'prediction is confirmed.** **[written after the sealed run;',
     '7.14 result: the registered prediction failed'),
    ('7.14 result prediction was against the thesis',
     'precisely so that it could not be quietly softened afterwards. **It failed.**',
     'precisely so that it could be revisited afterwards. **It held.**',
     '7.14 result: the prediction was registered against the thesis'),
    ('7.14 result bound 1 void budgets',
     '1. **Two of six budgets are decidable; the other four are void, not won.** The',
     '1. **Two of six budgets are decidable, and the other four went the same way.** The',
     '7.14 result: four budgets are void rather than won'),
    ('7.14 result bound 2 deployability',
     '2. **The crossings are at accuracies nobody would deploy.** **20.61%** and',
     '2. **The crossings are at deployable accuracies.** **20.61%** and',
     '7.14 result: the crossing accuracies are not deployable'),
    ('7.14 result bound 3 the window closes',
     '3. **Dense passes the sparse ceiling for 1.44\u00d7 the cost.** The sparse ladder tops',
     '3. **Dense never passes the sparse ceiling at any cost.** The sparse ladder tops',
     '7.14 result: the window closes at 1.44x the sparse ceiling'),
    ('7.14 result bound 4 resolution starvation',
     '   its bottom end.** `d4a_small_28` and `d4b_small_42` feed DINOv2 28\u00d728 and',
     '   its bottom end.** `d4a_small_28` and `d4b_small_42` feed DINOv2 128\u00d7128 and',
     '7.14 result: the dense ladder is resolution-starved at its bottom'),
    ('7.14 result reconciles M104',
     '**Kill switch 3 did not fire, and it reconciles with M104 rather than',
     '**Kill switch 3 did not fire, and it contradicts M104 rather than',
     '7.14 result: kill switch 3 reconciles with M104 rather than contradicting it'),
    ('7.14 result per-parameter versus per-MAC',
     'Partitioning buys nothing per *parameter* \u2014 M104 \u2014 and a great deal per',
     'Partitioning buys a great deal per *parameter* \u2014 M104 \u2014 and nothing per',
     '7.14 result: partitioning is per-MAC and not per-parameter'),
    ('7.14 result names the oracle subsidy',
     'the oracle. Both figures are oracle figures and neither survives without a router',
     'the oracle. Both figures are robust and each survives without a router',
     '7.14 result: the oracle subsidy is named as the reason'),
    ('7.14 result measures the resolution asymmetry',
     'therefore worth **14.89 pp** to the dense side on this corpus \u2014 a measured',
     'therefore worth **1.89 pp** to the dense side on this corpus \u2014 a measured',
     '7.14 result: the resolution asymmetry is finally measured'),
    ('7.14 result discloses the missing information-matched arm',
     "no information-matched dense arm inside the crossing window, so kill switch 2's",
     "an information-matched dense arm inside the crossing window, so kill switch 2's",
     '7.14 result: no information-matched arm sits inside the window'),
    ('7.14 result the ladder was truncated by the corpus',
     'does not know where the sparse curve goes next. That is a registered question for',
     'does not know where the sparse curve goes next. That is a settled matter for',
     '7.14 result: the sparse ladder was truncated by the corpus'),
    ('7.14 result the two capacity currencies',
     '**The two families do not pay for capacity in the same currency**: the',
     '**The two families pay for capacity in the same currency**: the',
     '7.14 result: the two families buy capacity in different currencies'),
    ('7.14 result discharges prohibition 27 narrowly',
     '**What M107 licenses, and what it does not.** Prohibition 27 is discharged for',
     '**What M107 licenses.** Prohibition 27 is discharged in full for',
     '7.14 result: prohibition 27 is discharged only for this comparison'),
    ('ledger M107 outcome letter',
     '**M107 result (plan \u00a77.14). Outcome letter: P \u2014 the registered prediction is',
     '**M107 result (plan \u00a77.14). Outcome letter: S \u2014 the registered prediction is',
     'ledger 7.14 result: the outcome letter is P'),
    ('ledger M107 refutation favours the thesis',
     "refuted, and the refutation favours this program's thesis.** Evidence:",
     "refuted, and the refutation settles this program's thesis.** Evidence:",
     'ledger 7.14 result: the refutation favours the thesis'),
    ('ledger M107 independent recomputation',
     'own gate by `experiments/tier4/report_v15_m107_gate.py`, which agrees with it on',
     'own gate by reading the gate field directly, which agrees with it on',
     'ledger 7.14 result: the gate was recomputed independently'),
    ('ledger M107 bounds are not optional',
     '(**+1.80 pp**), both under the LVD-142M asymmetry. **This claim is admissible',
     '(**+1.80 pp**), both under the LVD-142M asymmetry. **This claim is broadly admissible',
     'ledger 7.14 result: the crossing margins are quoted with their asymmetry'),
    ('7.14 result bound 5 exists',
     '5. **The comparison rule lets the sparse arm outspend its opponent, and the',
     '5. **The comparison rule holds both arms to the same budget exactly, and the',
     '7.14 result: bound 5 admits the sparse arm outspends its opponent'),
    ('7.14 result bound 5 registered not omitted',
     '   because it bounds a figure this document already quotes.]** Gate item 4',
     '   because it seemed worth a mention.]** Gate item 4',
     '7.14 result: bound 5 was added after the run rather than omitted'),
    ('7.14 result bound 5 survives interpolation',
     'interpolation at the upper one**, and any successor milestone should place a',
     'interpolation at every budget**, and any successor milestone should place a',
     '7.14 result: bound 5 states the crossing survives interpolation'),
    ('7.14 result bound count is five',
     '**The five bounds that travel with that headline, none of them optional.** The',
     '**The four bounds that travel with that headline, none of them optional.** The',
     '7.14 result: there are five bounds, not four'),
    ('ledger M107 five bounds',
     'only with its five registered bounds**, all recorded in \u00a77.14: **four of the six**',
     'only with its four registered bounds**, all recorded in \u00a77.14: **four of the six**',
     'ledger 7.14 result: the five bounds are called non-optional'),
    ('ledger M107 bound 5 nearly undoes the headline',
     "**M107's fifth bound was found after the run and is the one that most nearly",
     "**M107's fifth bound was found after the run and is a minor technicality that barely",
     'ledger 7.14 result: bound 5 is recorded as nearly undoing the headline'),
    ('ledger M107 interpolation is arithmetic not evidence',
     'budget, which is not a measured arm and is recorded as arithmetic rather than',
     'budget, which is an equally valid arm and is recorded as evidence rather than',
     'ledger 7.14 result: bound 5 calls interpolation arithmetic, not evidence'),
    ('ledger M107 bound 5 names the successor experiment',
     'correct successor experiment places a **measured** dense arm inside the window',
     'correct successor experiment interpolates a dense arm inside the window',
     'ledger 7.14 result: bound 5 names the successor experiment'),
    ('ledger M107 prohibition 27 stays narrow',
     '\u00a77.14 restriction 6 and remains in force everywhere else.',
     '\u00a77.14 restriction 6 and is lifted everywhere else too.',
     'ledger 7.14 result: prohibition 27 is discharged narrowly'),
    ('ledger M107 reconciles M104',
     'contradicting it.** At matched inference MACs and under **oracle** routing the',
     'contradicting it.** At matched inference MACs and under learned routing the',
     'ledger 7.14 result: kill switch 3 reconciles M104'),
    ('ledger M107 two budget readings',
     'budgets differ: **partitioning buys nothing per parameter and a great deal per',
     'budgets differ: **partitioning buys a great deal per parameter and nothing per',
     'ledger 7.14 result: the per-parameter and per-MAC readings differ'),
    ('ledger M107 resolution asymmetry',
     'sparse arms see scores **38.86%**, so the asymmetry is worth **14.89 pp** to the',
     'sparse arms see scores **38.86%**, so the asymmetry is worth **1.89 pp** to the',
     'ledger 7.14 result: the resolution asymmetry is quantified'),
    ('ledger M107 information parity gap',
     'crossing window**, so kill switch 2 cannot be re-decided at information parity;',
     'crossing window**, so kill switch 2 still holds at information parity;',
     'ledger 7.14 result: the crossing cannot be re-decided at information parity'),
    ('ledger M107 corpus truncation',
     'registered question, not a claim.** Rows per fitted dimension falls from',
     'settled finding, not a question.** Rows per fitted dimension falls from',
     'ledger 7.14 result: the ladder was truncated by the corpus'),
    ('ledger M107 curve above the ceiling is unmeasured',
     '3,072 atoms is unmeasured**, and no sentence in this program may assume it',
     '3,072 atoms is well understood**, and any sentence in this program may assume it',
     'ledger 7.14 result: the curve above 3,072 atoms is unmeasured'),
    ('ledger M107 prohibition 27 resolution marker',
     'comparison M107 measures. **[resolved in place, \u00a75.10: M107 has since run to',
     'comparison M107 measures. **[still open: M107 has since run to',
     'ledger 7.14: prohibition 27 is marked resolved in place'),
    ('7.14 amendment 5 discloses the narrowing of kill switch 3',
     '   rather than discovered in the output \u2014 and every sentence reporting kill',
     '   rather than discovered in the output \u2014 and no sentence need mention kill',
     '7.14 amendment 5 requires kill switch 3 to disclose the narrowing'),
    ('7.14 amendment 5 quotes clipart at 11,224 rows',
     'showed `clipart` holds **11,224** of the 138,000 train rows',
     'showed `clipart` holds **21,224** of the 138,000 train rows',
     '7.14 amendment 5 quotes the clipart row count'),
    ('7.14 amendment 5 quotes the floor readings',
     'ten rows per fitted dimension at 256 atoms (**10.96**) and fails at 512',
     'ten rows per fitted dimension at 256 atoms (**20.96**) and fails at 512',
     '7.14 amendment 5 quotes the 10.96 and 5.48 floor readings'),
    ('7.14 amendment 5 rejects the M104 cap-and-redistribute',
     '   its own floor and redistributing, as \u00a77.10 does, was considered and',
     '   its own floor and redistributing, as \u00a77.10 does, is adopted here and',
     '7.14 amendment 5 rejects capping and redistributing'),
    ('ledger records the M107 mixture narrowing',
     '**mixture ladder therefore runs only at 128 and 256**',
     '**mixture ladder therefore runs at all six budgets**',
     'ledger records the M107 mixture ladder narrowing'),
    ('7.14 quotes the 255 million sparse ceiling',
     'the sparse ladder tops out at **255 million** MACs while the registered',
     'the sparse ladder tops out at **355 million** MACs while the registered',
     '7.14 amendment 1 quotes the 255-million sparse ceiling'),
    ('7.14 quotes the 216 million registered floor',
     "sweep's cheapest point sits at **216 million**, so exactly **one** sparse",
     "sweep's cheapest point sits at **116 million**, so exactly **one** sparse",
     '7.14 amendment 1 quotes the 216-million registered floor'),
    ('7.14 states the one-point anecdote',
     'budget would have had a dense reference at or below it. A comparison with one',
     'budget would have had a dense reference at or below it. A comparison with two',
     '7.14 amendment 1 quotes the 108 and 368 million additions'),
    ('7.14 refuses a degenerate dense opponent',
     'would have to sit below one patch of image, and handing the sparse side a',
     'would have to sit below one patch of image, and that point is added too, a',
     '7.14 amendment 1 refuses a degenerate dense opponent'),
    ('7.14 quotes the 64 verified rows',
     'runner decodes the first **64** selected rows of each split straight',
     'runner decodes the first **6** selected rows of each split straight',
     '7.14 amendment 3 quotes the 64 verified rows'),
    ('7.14 quotes the 3450-atom floor headroom',
     '**3,450** atoms, which is above the top',
     '**4,350** atoms, which is above the top',
     '7.14 quotes the 3,450-atom floor headroom'),
    ('ledger marks the 3,455 as an anchor',
     '**Provenance of the 3,455.**',
     '**Prediction of the 3,455.**',
     'ledger marks the 3,455 as an anchor and not a prediction'),
    ('7.10 amendment count is five, not four',
     'exposed a fifth. All five are recorded here, in the plan',
     'exposed a fifth. All four are recorded here, in the plan',
     '7.10 records five execution-time amendments'),
    ('7.10 separates the direction of amendment 5 from the first four',
     'The first four make it **harder**; the fifth makes the sample floor',
     'All five of them make it **harder**; the fifth makes the sample floor',
     '7.10 does not claim amendment 5 makes the milestone harder'),
    ('ledger M104 fifth amendment',
     'A **fifth** amendment was forced by running the instrument once on a smoke',
     'A **fifth** amendment was forced by reading the sealed run\u0027s own output',
     'ledger records the M104 fifth amendment'),
    ('ledger M104 fifth amendment is disclosed not repaired quietly',
     'defect is recorded rather than quietly repaired because the guard it broke',
     'defect is repaired without comment because the guard it broke',
     'ledger states the defect is disclosed rather than quietly repaired'),
]


def run_negative_control():
    import hashlib
    import os
    import subprocess

    originals = {p: p.read_bytes() for p in (PLAN, LEDGER_PATH)}
    before = {p: hashlib.sha256(b).hexdigest() for p, b in originals.items()}
    problems = []
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    try:
        for label, find, repl, expect in NEGATIVE_CONTROLS:
            source = LEDGER_PATH if 'ledger' in label else PLAN
            base = source.read_bytes()
            body = base.decode('utf-8')
            if body.count(find) != 1:
                problems.append(
                    f'{label}: corruption target appears {body.count(find)} '
                    f'times in {source.name}, expected exactly 1')
                continue
            source.write_bytes(body.replace(find, repl).encode('utf-8'))
            try:
                out = subprocess.run(
                    [sys.executable, str(pathlib.Path(__file__).resolve())],
                    capture_output=True, text=True, encoding='utf-8',
                    errors='replace', env=env).stdout or ''
            finally:
                source.write_bytes(base)
            fired = any(expect in line for line in out.splitlines()
                        if line.startswith('MISMATCH')
                        or 'MISSING' in line or 'STRUCTURAL' in line
                        or 'BROKEN' in line)
            print(f'{"DETECTED" if fired else "NOT DETECTED":>13}  {label}')
            if not fired:
                problems.append(f'{label}: verifier did not fire')
    finally:
        for path, blob in originals.items():
            path.write_bytes(blob)

    restored = True
    for path, digest in before.items():
        after = hashlib.sha256(path.read_bytes()).hexdigest()
        same = after == digest
        restored &= same
        print(f'\n{path.name}\n  sha256 before {digest}\n'
              f'  sha256 after  {after}\n  restored byte-identical: {same}')
    if problems or not restored:
        for problem in problems:
            print(f'  FAIL {problem}')
        sys.exit(1)
    print(f'\nAll {len(NEGATIVE_CONTROLS)} negative controls fired, '
          f'and both documents were restored byte-identically.')
    sys.exit(0)


LEDGER_PATH = ROOT / 'analysis' / 'CLAIM_LEDGER_v15.md'
if '--negative-control' in sys.argv:
    run_negative_control()

text = PLAN.read_text(encoding='utf-8')

m80 = json.load(open(ROOT / 'logs/results/v13/m80_sparse_dictionary/evidence.json'))
m81 = json.load(open(ROOT / 'logs/results/v13/m81_sparse_head/evidence.json'))

w128 = next(x for x in m81['seeds'][0]['widths'] if x['width'] == 'i5_128')
arms = {a['arm']: a for a in w128['arms']}
cells = {(c['dictionary_size'], c['active_atoms']): c for c in m80['cells']}

checks = []


def check(label, quoted, actual, places=6):
    if isinstance(quoted, bool) or isinstance(actual, bool):
        ok = quoted == actual
    elif isinstance(quoted, str) or isinstance(actual, str):
        ok = quoted == actual
    else:
        # Tolerance of half a unit in the last quoted place, so that exact
        # rounding ties (0.5078125 -> 0.507812 or 0.507813) both pass.
        ok = abs(float(quoted) - float(actual)) <= 0.5 * 10 ** (-places) + 1e-12
    checks.append((ok, label, quoted, actual))
    return ok


# --- M81 128-way accuracies quoted in section 2.1 ---
for arm, quoted in [
    ('metric_field_shrinkage_1.0', 0.663452),
    ('knn', 0.661255),
    ('metric_field_shrinkage_0.5', 0.660889),
    ('mlp_integrated_gradients', 0.660522),
    ('mlp_expected_gradients', 0.660522),
    ('metric_field_shrinkage_0.1', 0.636841),
    ('sparse_linear_l1_0.0', 0.607178),
    ('sparse_linear_l1_0.3', 0.573975),
    ('rbf_nystroem', 0.569092),
    ('sparse_linear_l1_0.03', 0.520020),
    ('sparse_linear_l1_0.1', 0.507813),
    ('sparse_linear_budget_1024', 0.441284),
    ('sparse_linear_budget_512', 0.351563),
    ('sparse_linear_budget_256', 0.225098),
    ('decision_list', 0.146362),
]:
    check(f'M81 acc {arm}', quoted, arms[arm]['balanced_accuracy'])

# --- explanation lengths quoted in section 2.1 ---
for arm, atoms, frac in [
    ('knn', 6.72, 1.0),
    ('sparse_linear_budget_1024', 5.16, 0.9655),
    ('sparse_linear_l1_0.3', 15.60, 0.0607),
    ('rbf_nystroem', 2047.96, 0.0),
    ('decision_list', 0.22, 1.0),
]:
    el = arms[arm]['explanation_length']
    check(f'M81 atoms {arm}', atoms, el['mean_active_atoms'], places=2)
    check(f'M81 budgetfrac {arm}', frac, el['fraction_of_decisions_within_budget'], places=4)

# --- the registered floor and the two derived gaps ---
floor = m81['gate']['i5_128']['comparable_accuracy_floor_per_seed'][0]
check('I5-128 comparability floor', 0.6112548828125, floor, places=12)
check('floor == knn - 0.05', floor, arms['knn']['balanced_accuracy'] - 0.05, places=12)
check('shortfall of best atom arm (pt)', 0.4077,
      (floor - arms['sparse_linear_l1_0.0']['balanced_accuracy']) * 100, places=3)
check('shortfall at budget (pt)', 16.997,
      (floor - arms['sparse_linear_budget_1024']['balanced_accuracy']) * 100, places=3)
check('nonlinear cluster spread (pt)', 0.293,
      (arms['metric_field_shrinkage_1.0']['balanced_accuracy']
       - arms['mlp_integrated_gradients']['balanced_accuracy']) * 100, places=3)
check('seeds with no admissible atom arm', 3,
      m81['gate']['i5_128']['seeds_with_no_admissible_atom_arm'])
check('H100 kill threshold = best + 0.01', 0.673452,
      arms['metric_field_shrinkage_1.0']['balanced_accuracy'] + 0.01, places=6)

# --- M80 grid quoted in section 2.3 ---
raw = m80['gate']['raw_feature_probe_balanced_accuracy']
check('M80 raw probe bar', 0.613037109375, raw, places=12)
for (mm, kk), probe, rand, ent in [
    ((2048, 16), 0.526367, 0.400513, 3.872),
    ((2048, 32), 0.535156, 0.478271, 4.819),
    ((2048, 64), 0.552734, 0.535645, 5.521),
    ((4096, 16), 0.549927, 0.457153, 3.069),
    ((4096, 32), 0.561523, 0.538086, 4.159),
    ((4096, 64), 0.574951, 0.584351, 4.998),
    ((8192, 16), 0.583130, 0.496338, 2.285),
    ((8192, 32), 0.607910, 0.570190, 3.381),
    ((8192, 64), 0.612061, 0.612305, 4.338),
]:
    c = cells[(mm, kk)]
    check(f'M80 probe m={mm} k={kk}', probe, c['codes_probe_balanced_accuracy'])
    check(f'M80 rand  m={mm} k={kk}', rand, c['random_control_probe_balanced_accuracy'])
    check(f'M80 entropy m={mm} k={kk}', ent, c['mean_atom_label_entropy_bits'], places=3)

for (mm, kk), margin in [((2048, 16), 12.585), ((8192, 16), 8.679), ((8192, 64), -0.024)]:
    c = cells[(mm, kk)]
    check(f'M80 margin m={mm} k={kk}', margin,
          (c['codes_probe_balanced_accuracy']
           - c['random_control_probe_balanced_accuracy']) * 100, places=3)

check('M80 shuffled-label entropy', 5.1312,
      m80['cells'][0]['shuffled_label_entropy_bits'], places=4)
check('M80 config hash quoted', True,
      m80['configuration_hash'] in text)
check('M81 config hash quoted', True,
      m81['configuration_hash'] in text)
check('corpus index hash quoted', True, m80['corpus']['index_sha256'] in text)

# --- v15 planning probe numbers quoted in section 2.4 ---
sf = pathlib.Path.home() / '.copilot/session-state/808d9a7b-43f9-438f-a1db-0d68ed1a8b6a/files'
if (sf / 'probe_tree_verification.json').exists():
    tv = json.load(open(sf / 'probe_tree_verification.json'))
    check('probe RF', 0.646851, tv['random_forest_400']['eval'])
    check('probe HGB lr0.05', 0.577393, tv['hgb_lr0.05_iter100_leaf31']['eval'])
    check('probe logistic train', 0.846313, tv['logistic_reference']['train'])
    check('probe logistic eval', 0.568115, tv['logistic_reference']['eval'])
    hs_raw = json.load(open(sf / 'probe_hierarchical_sparse_results.json'))['arms']
    hs = hs_raw[0] if isinstance(hs_raw, list) else hs_raw
    check('probe 10-NN', 0.659790, hs['knn_10']['accuracy'])
    check('probe dense 384d', 0.583618, hs['dense_probe_384d']['accuracy'])
    check('probe top-32 ambient', 0.522583, hs['flat_ovr_k32']['accuracy'])
    check('probe top-16 ambient', 0.451050, hs['flat_ovr_k16']['accuracy'])
    check('probe top-8 ambient', 0.335205, hs['flat_ovr_k8']['accuracy'])
    best_hier = max(v['accuracy'] for k, v in hs.items() if k.startswith('hier_'))
    check('probe best hierarchy arm', 0.112549, best_hier)

# --- section 2.6.1 parameter costs, new in the efficiency revision ---
for arm, quoted_params in [
    ('metric_field_shrinkage_1.0', 2_097_152),
    ('knn', 25_165_824),
    ('mlp_integrated_gradients', 262_784),
    ('sparse_linear_l1_0.0', 1_048_704),
    ('sparse_linear_budget_1024', 131_200),
    ('sparse_linear_budget_256', 32_896),
    ('decision_list', 67),
]:
    check(f'2.6.1 active_parameters {arm}', quoted_params,
          arms[arm]['active_parameters'], places=0)

# the dense-probe comparison the section is built on
check('2.6.1 dense linear probe parameters', 49_280, 384 * 128 + 128, places=0)
check('2.6.1 sparse/dense parameter ratio', 21.3,
      round(arms['sparse_linear_l1_0.0']['active_parameters'] / 49_280, 1), places=1)

# --- section 2.6.2 MAC derivation for the frozen ViT-S/14 trunk ---
N, D, LAYERS = 257, 384, 12
per_layer = 3 * N * D * D + N * D * D + 2 * N * N * D + 2 * N * D * 4 * D
trunk_macs = per_layer * LAYERS
check('2.6.2 trunk MACs per image', 6_065_759_232, trunk_macs, places=0)
check('2.6.2 dense head MACs', 49_152, 384 * 128, places=0)
check('2.6.2 sparse head MACs', 3_149_824, 384 * 8192 + 32 * 128, places=0)
check('2.6.2 dense head share of trunk %', 0.00081,
      round(100 * (384 * 128) / trunk_macs, 5), places=5)
check('2.6.2 sparse head share of trunk %', 0.0519,
      round(100 * (384 * 8192 + 32 * 128) / trunk_macs, 4), places=4)

# --- section 2.7 tier6 sequence figures ---
t6path = ROOT / 'logs/results/tier6_locked_window5_confirmation.json'
t6 = json.load(open(t6path))['results']['window']
check('2.7 tier6 geometric head', 0.3036, t6['test_acc_final'], places=4)
check('2.7 tier6 linear head', 0.3464, t6['linear_acc'], places=4)
check('2.7 tier6 matched n-gram', 0.445, t6['ngram_matched_acc'], places=3)
check('2.7 tier6 best practical n-gram', 0.47606901725431355,
      t6['ngram_best_practical_acc'], places=12)
check('2.7 tier6 unigram floor', 0.19215, t6['unigram_acc'], places=5)
check('2.7 tier6 perplexity', 14.246250235966526, t6['ppl_final'], places=12)
check('2.7 tier6 classes below sample floor', 56,
      t6['sample_adequacy']['below_minimum'], places=0)
check('2.7 tier6 class count in adequacy block', 82,
      t6['sample_adequacy']['class_count'], places=0)
check('2.7 tier6 min_seed', 299, t6['sample_adequacy']['min_seed'], places=0)
check('2.7 tier6 EM iterations actually run', 0,
      json.load(open(t6path))['config']['n_em_iters'], places=0)

# --- section 2.8: the v15 planning probe. These are NOT sealed figures; the
# plan marks them inadmissible (prohibition 21). They are verified anyway so
# that the plan cannot drift from the artifact it cites.
SESSION_FILES = pathlib.Path(
    r'C:\Users\mak\.copilot\session-state'
    r'\808d9a7b-43f9-438f-a1db-0d68ed1a8b6a\files')
probe_path = next((p for p in (ROOT / 'probe_difficulty_skew.json',
                               SESSION_FILES / 'probe_difficulty_skew.json')
                   if p.exists()), None)
if probe_path is None:
    print('WARNING: probe_difficulty_skew.json not found; skipping 2.8 checks')
else:
    probe = json.load(open(probe_path))
    check('2.8 probe full-kNN (must differ from sealed 0.661255)',
          0.6357, probe['knn_balanced_accuracy'], places=4)
    curves = {k: {round(r['deferral_rate'], 2): r for r in v}
              for k, v in probe['cascade_curves'].items()}
    for dims, rate, oracle, conf, null, recov in [
        ('16', 0.75, 0.6535, 0.5722, 0.5340, 0.320),
        ('32', 0.50, 0.5990, 0.5414, 0.4934, 0.454),
        ('32', 0.75, 0.6638, 0.6063, 0.5677, 0.401),
        ('64', 0.40, 0.5980, 0.5485, 0.5020, 0.485),
        ('64', 0.50, 0.6432, 0.5762, 0.5228, 0.444),
        ('64', 0.75, 0.6649, 0.6220, 0.5792, 0.499),
    ]:
        row = curves[dims][rate]
        check(f'2.8 NCM({dims}) p={rate} oracle', oracle, row['oracle'], places=4)
        check(f'2.8 NCM({dims}) p={rate} confidence', conf, row['confidence'], places=4)
        check(f'2.8 NCM({dims}) p={rate} null', null, row['random_null'], places=4)
        check(f'2.8 NCM({dims}) p={rate} recovered fraction', recov,
              (row['confidence'] - row['random_null'])
              / (row['oracle'] - row['random_null']), places=3)
    # The claim that carries the redirect: oracle at 50% deferral beats the
    # full model. If this ever stops holding, section 2.8 Reading 1 is void.
    check('2.8 oracle at p=0.50 exceeds full kNN', True,
          curves['64'][0.50]['oracle'] > probe['knn_balanced_accuracy'])
    # H110's bar must sit above the measured baseline, or it is not a bar.
    check('4.3 H110 bar (0.60) exceeds measured recovery', True,
          0.60 > (curves['64'][0.50]['confidence'] - curves['64'][0.50]['random_null'])
          / (curves['64'][0.50]['oracle'] - curves['64'][0.50]['random_null']))

    # Section 2.8.2 Reading 2 is a claim about DIRECTION across stage-1
    # capacity, and an earlier draft of it asserted the opposite of what the
    # artifact says. Recompute the directions rather than trust the prose.
    def frac(d, rate):
        row = curves[d][rate]
        return ((row['confidence'] - row['random_null'])
                / (row['oracle'] - row['random_null']))

    dims = ['8', '16', '32', '64']
    for rate, quoted in [(0.25, [0.964, 0.731, 0.634, 0.528]),
                         (0.40, [0.749, 0.620, 0.533, 0.485]),
                         (0.50, [0.491, 0.496, 0.454, 0.444]),
                         (0.75, [0.317, 0.320, 0.401, 0.499])]:
        for d, q in zip(dims, quoted):
            check(f'2.8.2 NCM({d}) p={rate} recovered', q, frac(d, rate),
                  places=3)
    for rate in (0.25, 0.40):
        f = [frac(d, rate) for d in dims]
        check(f'2.8.2 recovered fraction FALLS with capacity at p={rate}',
              True, f == sorted(f, reverse=True))
    for rate in (0.25, 0.40, 0.50):
        g = [curves[d][rate]['oracle'] - curves[d][rate]['confidence']
             for d in dims]
        check(f'2.8.2 absolute gap WIDENS with capacity at p={rate}',
              True, g == sorted(g))
    f50 = [frac(d, 0.50) for d in dims]
    check('2.8.2 recovered fraction non-monotonic at p=0.50', True,
          f50 != sorted(f50) and f50 != sorted(f50, reverse=True))
    f75 = [frac(d, 0.75) for d in dims]
    check('2.8.2 recovered fraction RISES with capacity at p=0.75', True,
          f75 == sorted(f75))
    # The 3.5x accuracy claim that makes Reading 2 interesting.
    ncm = {str(k): v for k, v in probe['ncm_balanced_accuracy'].items()}
    check('2.8.2 NCM(8) accuracy', 0.1172, ncm['8'], places=4)
    check('2.8.2 NCM(64) accuracy', 0.4150, ncm['64'], places=4)
    check('2.8.2 stage-1 accuracy grows >3x from 8 to 64 dims', True,
          ncm['64'] / ncm['8'] > 3.0)

# --- M102 Tier A: every figure quoted in the plan and the claim ledger must
# come from the evidence file, and the ledger's verdict must be the artifact's.
LEDGER = ROOT / 'analysis' / 'CLAIM_LEDGER_v15.md'
m102path = ROOT / 'logs/results/v15/m102_abstention/evidence.json'
if not m102path.exists():
    print('WARNING: M102 evidence not found; skipping M102 checks')
else:
    m102 = json.load(open(m102path))
    ledger = LEDGER.read_text(encoding='utf-8')
    gate = m102['gate']
    recovered = gate['recovered_fraction']

    check('M102 verdict is refuted', 'refuted', gate['verdict'])
    check('M102 Tier B does not open', False, gate['tier_b_opens'])
    check('M102 bar is the registered 0.60', 0.60,
          gate['recovered_fraction_bar'], places=4)
    check('M102 primary dims', 32, gate['primary_pca_dimensions'], places=0)
    check('M102 primary deferral rate', 0.50,
          gate['primary_deferral_rate'], places=4)
    for arm, quoted in [('b_sparse_margin', 0.306),
                        ('b_prime_sparse_temperature', 0.277),
                        ('a_ncm_margin', 0.225),
                        ('c_sparse_selective', 0.136),
                        ('d_dense_gate', -0.019)]:
        check(f'M102 ledger recovered {arm}', quoted,
              recovered[arm]['mean'], places=3)
    # The claims that carry the ledger's two headline readings.
    check('M102 joint objective did NOT beat the margin gate', True,
          recovered['c_sparse_selective']['mean']
          < recovered['b_sparse_margin']['mean'])
    check('M102 temperature scaling did NOT beat the raw margin', True,
          recovered['b_prime_sparse_temperature']['mean']
          < recovered['b_sparse_margin']['mean'])
    check('M102 dense supervised gate is at or below random', True,
          recovered['d_dense_gate']['mean'] <= 0.0)
    check('M102 best arm falls short of the 0.60 bar', True,
          max(v['mean'] for v in recovered.values()) < 0.60)

    # Instrument checks: these are the properties that make the numbers mean
    # anything, and one of them failed in the first implementation.
    every_point = [p for s in m102['seeds'] for w in s['widths']
                   for a in w['arms'] for p in a['points']]
    check('M102 oracle is an upper bound at EVERY point', True,
          all(p['oracle_is_upper_bound'] for p in every_point))
    check('M102 no degenerate (constant) gate', False,
          any(p['gate']['degenerate'] for p in every_point))
    check('M102 all gate scores finite', True,
          all(p['gate']['all_finite'] for p in every_point))

    # The 64-dimensional width must be void, because the plan's probe took its
    # headline from it and section 2.8.5 says so.
    widths = {w['pca_dimensions']: w for w in m102['seeds'][0]['widths']}
    check('M102 d=64 is void below the sample floor',
          'void_below_sample_floor', widths[64]['status'])
    check('M102 d=64 fit samples per dimension', 7.0,
          widths[64]['fit_samples_per_fitted_dimension'], places=1)
    check('M102 d=32 is measured, not void', 'measured', widths[32]['status'])
    check('M102 d=32 fit samples per dimension', 14.0,
          widths[32]['fit_samples_per_fitted_dimension'], places=1)

    # Reading 1, corrected: the oracle beats the full model at every rate.
    stage_two = float(np.mean([s['stage_two_balanced_accuracy']
                               for s in m102['seeds']])) if False else None
    stage_two_values = [s['stage_two_balanced_accuracy'] for s in m102['seeds']]
    stage_two = sum(stage_two_values) / len(stage_two_values)
    check('M102 stage-2 alone (ledger quotes 0.6322)', 0.6322, stage_two,
          places=4)
    best_oracle_32 = max(
        sum(next(p['oracle_balanced_accuracy']
                 for p in next(a for a in
                               next(w for w in s['widths']
                                    if w['pca_dimensions'] == 32)['arms']
                               if a['arm'] == arm)['points']
                 if p['deferral_rate'] == rate)
            for s in m102['seeds']) / len(m102['seeds'])
        for arm in ('a_ncm_margin', 'b_sparse_margin', 'c_sparse_selective')
        for rate in (0.25, 0.4, 0.5, 0.6, 0.75))
    check('M102 best oracle at 32 dims (ledger quotes 0.6976)', 0.6976,
          best_oracle_32, places=4)
    check('M102 oracle beats the full model at 32 dims', True,
          best_oracle_32 > stage_two)

    # The ledger must not describe Tier A in the language of compute saving.
    check('M102 evidence records that Tier A saves no compute', True,
          m102['tier_a_saves_no_compute'])
    forbidden = []
    check('M102 trunk MACs recorded', 6065759232,
          m102['trunk_macs_per_input_all_arms'], places=0)
    forbidden = [phrase for phrase in
                 ('compute saving', 'saves compute', 'reduces compute',
                  'faster inference', 'speedup')
                 if phrase in ledger.lower().replace(
                     'no compute saving is claimed', '')]
    # Tier A must never be described as saving compute. A bare substring test
    # fires on the ledger's own denial, so each occurrence is required to sit
    # inside a negating clause. Whitespace is normalised first because the
    # negations wrap across lines in the source.
    negations = ('no compute saving is claimed', 'forbids reporting',
                 'does not', 'no arm saved', 'never', 'no compute')
    offending = []
    lowered = re.sub(r'\s+', ' ', ledger.lower())
    for phrase in ('compute saving', 'saves compute', 'reduces compute',
                   'faster inference', 'speedup'):
        start = 0
        while (found := lowered.find(phrase, start)) != -1:
            window = lowered[max(0, found - 200): found + len(phrase) + 60]
            if not any(n in window for n in negations):
                offending.append(phrase)
            start = found + len(phrase)
    check('ledger uses no compute-saving language for Tier A',
          '', ', '.join(sorted(set(offending))))
    check('ledger records the refuted verdict', True,
          'verdict | **refuted**' in ledger)
    check('ledger discloses the oracle defect', True,
          'C102.5' in ledger and '141.4%' in ledger)
    check('ledger states no v15 outcome letter', True,
          'no v15 outcome letter' in re.sub(r'\s+', ' ', ledger))

    # --- arm (d) sample adequacy: the correction C102.2 records -------------
    # Arm (d) is the only arm fitted on the calibration split, so its adequacy
    # is its own and is not the width-level figure the evidence records for the
    # other four. The ledger quotes these ratios; recompute them from the
    # evidence rather than trusting the prose.
    calibration_rows = m102['seeds'][0].get('calibration_rows', 8192)
    check('M102 calibration split size', 8192, calibration_rows, places=0)
    armd_expected = {8: 12.78, 16: 7.10, 32: 3.76, 64: 1.94}
    for dims, expected in armd_expected.items():
        w = widths[dims]
        armd = next((a for a in w['arms'] if a['arm'].startswith('d_')), None)
        check(f'M102 arm (d) exists at d={dims}', True, armd is not None)
        if armd is None:
            continue
        params = armd['gate_parameters']
        check(f'M102 arm (d) parameters at d={dims}', dims * 64 + 129, params,
              places=0)
        check(f'M102 arm (d) rows per parameter at d={dims}', expected,
              calibration_rows / params, places=2)
    check('M102 arm (d) adequate only at d=8', 'True,False,False,False',
          ','.join(str(calibration_rows / (d * 64 + 129) >= 10.0)
                   for d in (8, 16, 32, 64)))
    check('ledger voids the arm (d) width', True,
          'void — see C102.2' in ledger)

    # --- the d=8 readings the ledger now leads with -------------------------
    # d=8 is the only width where every arm, including (d), clears the floor.
    def recovered(dims, arm_prefix, rate):
        vals = []
        for s in m102['seeds']:
            w = next(x for x in s['widths'] if x['pca_dimensions'] == dims)
            a = next((x for x in w['arms'] if x['arm'].startswith(arm_prefix)),
                     None)
            if a is None:
                return None
            p = next(x for x in a['points'] if x['deferral_rate'] == rate)
            vals.append(p['recovered_fraction'])
        return sum(vals) / len(vals)

    for prefix, expected in [('a_', 0.075), ('b_sparse_margin', 0.094),
                             ('b_prime', 0.133), ('c_', 0.251), ('d_', 0.026)]:
        got = recovered(8, prefix, 0.5)
        check(f'M102 d=8 recovered fraction, arm {prefix} at 0.50 deferral',
              expected, got, places=3)
    check('M102 arm (d) is the worst arm at d=8, 0.50 deferral', True,
          recovered(8, 'd_', 0.5) == min(
              recovered(8, p, 0.5)
              for p in ('a_', 'b_sparse_margin', 'b_prime', 'c_', 'd_')))
    check('ledger retracts the representation reading', True,
          'sample-expensive' in ledger)

# --- M103: the ledger entry must be recomputed from the sealed evidence ----
# Every figure C103.1-C103.8 quotes is derived here from evidence.json rather
# than trusted. A ledger that disagrees with its own artifact is the failure
# mode this whole file exists to catch.
m103path = ROOT / 'logs/results/v15/m103_atoms/evidence.json'
if not m103path.exists():
    print('WARNING: M103 evidence not found; skipping M103 checks')
else:
    m103 = json.load(open(m103path))
    ledger = LEDGER.read_text(encoding='utf-8')
    ascii_ledger = ledger.replace('\u2212', '-')
    g103 = m103['gate']
    acc = {}
    conv = {}
    for s in m103['seeds']:
        for b in s['budgets']:
            for a in b['arms']:
                acc[(a['arm'], b['atoms'], s['seed'])] = a['test_accuracy']
                conv[(a['arm'], b['atoms'], s['seed'])] = a['converged']
    seeds103 = [s['seed'] for s in m103['seeds']]
    readable = g103['readable_budgets']

    def mean103(arm, atoms):
        return sum(acc[(arm, atoms, s)] for s in seeds103) / len(seeds103)

    check('M103 verdict is confirmed', 'confirmed', g103['verdict'])
    check('M103 ledger states the confirmed verdict', True,
          '| verdict | **confirmed** |' in ledger)
    check('M103 seeds are the registered three', '11,23,37',
          ','.join(str(s) for s in seeds103))
    check('M103 readable budgets', '64,128,256,512,1024',
          ','.join(str(b) for b in readable))
    check('M103 2048 rung is void', '2048',
          ','.join(str(b) for b in g103['void_budgets']))
    check('M103 reference budget is the registered 1024', 1024,
          g103['reference_budget'], places=0)

    # C103.1: the accuracy table, every cell recomputed.
    for atoms, a_q, b_q, c_q, d_q in [
            (64, 0.5730, 0.5746, 0.6065, 0.5349),
            (128, 0.6211, 0.6143, 0.6414, 0.5765),
            (256, 0.6520, 0.6477, 0.6746, 0.6167),
            (512, 0.6720, 0.6705, 0.6908, 0.6476),
            (1024, 0.6879, 0.6839, 0.6983, 0.6684)]:
        for arm, quoted in [('a_random_patches', a_q), ('b_kmeans', b_q),
                            ('c_discriminative', c_q),
                            ('d_random_projections', d_q)]:
            check(f'M103 ledger accuracy {arm} at {atoms} atoms', quoted,
                  mean103(arm, atoms), places=4)
        check(f'M103 ledger row present at {atoms} atoms', True,
              f'| {atoms} | {a_q:.4f} | {b_q:.4f} | **{c_q:.4f}** | '
              f'{d_q:.4f} | **+{c_q - a_q:.4f}** |' in ascii_ledger)

    # The headline: efficiency at matched accuracy.
    reference = mean103('a_random_patches', 1024)
    check('M103 reference accuracy quoted in the ledger', 0.6879, reference,
          places=4)
    check('M103 arm (c) reaches the reference at the 512 rung', 512,
          g103['atoms_to_reach_reference']['c_discriminative']['rung'],
          places=0)
    check('M103 arm (c) at 512 really does clear the reference', True,
          mean103('c_discriminative', 512) >= reference)
    check('M103 interpolated atom count quoted as 465.8', 465.8,
          g103['atoms_to_reach_reference']['c_discriminative'][
              'interpolated'], places=1)
    check('M103 rung reduction quoted as 2.0x', 2.0, 1024 / 512, places=1)
    check('M103 interpolated reduction quoted as 2.2x', 2.2,
          1024 / g103['atoms_to_reach_reference']['c_discriminative'][
              'interpolated'], places=1)
    for arm in ('b_kmeans', 'd_random_projections'):
        check(f'M103 arm {arm} never reaches the reference', True,
              g103['atoms_to_reach_reference'][arm]['rung'] is None)

    # Per-seed, which is the claim the ledger actually makes.
    check('M103 every seed reaches the reference at 512', '512,512,512',
          ','.join(str(r) for r in g103['c_reaches_reference_per_seed_rung']))
    for seed, c_q, a_q in [(11, 0.6862, 0.6818), (23, 0.6950, 0.6919),
                           (37, 0.6913, 0.6900)]:
        check(f'M103 seed {seed} c@512 quoted', c_q,
              acc[('c_discriminative', 512, seed)], places=4)
        check(f'M103 seed {seed} a@1024 quoted', a_q,
              acc[('a_random_patches', 1024, seed)], places=4)
        check(f'M103 seed {seed} beats its own reference', True,
              acc[('c_discriminative', 512, seed)]
              >= acc[('a_random_patches', 1024, seed)])
        check(f'M103 ledger quotes the seed {seed} pair', True,
              f'{c_q:.4f} vs {a_q:.4f} (seed {seed})' in ledger)

    cells = [(b, s) for b in readable for s in seeds103]
    wins = sum(1 for b, s in cells
               if acc[('c_discriminative', b, s)]
               > acc[('a_random_patches', b, s)])
    check('M103 arm (c) wins all 15 seed-budget cells', 15, wins, places=0)
    check('M103 ledger quotes 15 of 15', True, '**15 of 15**' in ledger)
    check('M103 kill switch 1 did not fire', True,
          g103['kill_switch_1_c_beats_a_at_every_seed'])
    check('M103 kill switch 2 did not fire', False,
          g103['kill_switch_2_d_matches_a'])

    # C103.2: arm (d) below arm (a) at every readable rung.
    deltas_ad = [mean103('a_random_patches', b)
                 - mean103('d_random_projections', b) for b in readable]
    check('M103 arm (d) is below arm (a) at every readable rung', True,
          all(d > 0 for d in deltas_ad))
    check('M103 ledger quotes the arm (a)-(d) range low end', 0.0195,
          min(deltas_ad), places=4)
    check('M103 ledger quotes the arm (a)-(d) range high end', 0.0446,
          max(deltas_ad), places=4)

    # C103.3: the narrowing margin, which is the limitation the ledger is
    # required to carry. Recompute both the margins and the seed spreads.
    ratios103 = {}
    for atoms, quoted_ratio in [(64, 3.45), (128, 1.55), (256, 4.26),
                                (512, 2.58), (1024, 1.03)]:
        margin = mean103('c_discriminative', atoms) - mean103(
            'a_random_patches', atoms)
        null_values = [acc[('a_random_patches', atoms, s)] for s in seeds103]
        spread = max(null_values) - min(null_values)
        check(f'M103 margin-to-spread ratio at {atoms} atoms', quoted_ratio,
              margin / spread, places=2)
        check(f'M103 ledger quotes the {atoms}-atom ratio row', True,
              re.search(rf'\| {atoms} \| \+{margin:.4f} \| {spread:.4f} \| '
                        rf'\*?\*?{quoted_ratio:.2f}\*?\*? \|',
                        ascii_ledger) is not None)
        ratios103[atoms] = margin / spread
    check('M103 the margin really does narrow from 64 to 1024 atoms', True,
          ratios103[1024] < ratios103[64])
    check('M103 the top readable rung is barely one seed spread', True,
          1.0 <= ratios103[1024] < 1.1)
    check('M103 ledger carries the narrowing-margin limitation', True,
          'C103.3' in ledger and 'narrows' in ledger)
    check('M103 ledger binds C103.3 to C103.1', True,
          'C103.3 travels with C103.1' in ledger)

    # C103.4: the convergence disclosure must match the artifact exactly.
    unconverged = sorted((b, s, arm) for (arm, b, s) in conv
                         if not conv[(arm, b, s)])
    check('M103 total non-converged fits', 10, len(unconverged), places=0)
    readable_bad = [x for x in unconverged if x[0] in readable]
    check('M103 non-converged fits at readable rungs', 5, len(readable_bad),
          places=0)
    check('M103 non-converged fits at the void rung', 5,
          len(unconverged) - len(readable_bad), places=0)
    check('M103 every readable non-convergence is at the 256 rung', True,
          all(b == 256 for b, _, _ in readable_bad))
    check('M103 arm (c) converged at every readable rung', True,
          all(arm != 'c_discriminative' for _, _, arm in readable_bad))
    check('M103 ledger discloses the convergence defect', True,
          'C103.4' in ledger and 'did not converge' in ledger)
    # The interpolation the headline uses must itself be uncontaminated.
    for arm, atoms in [('c_discriminative', 256), ('c_discriminative', 512),
                       ('a_random_patches', 1024)]:
        check(f'M103 interpolation input {arm}@{atoms} converged at all seeds',
              True, all(conv[(arm, atoms, s)] for s in seeds103))

    # C103.5: the 2.9.6 reversal is overturned, and seed 11 reproduces.
    for arm, quoted in [('a_random_patches', 0.6818), ('b_kmeans', 0.6856),
                        ('c_discriminative', 0.6951),
                        ('d_random_projections', 0.6760)]:
        check(f'M103 seed 11 @1024 reproduces 2.9.6 for {arm}', quoted,
              acc[(arm, 1024, 11)], places=4)
    a_beats_b = [b for b in readable
                 if mean103('a_random_patches', b) > mean103('b_kmeans', b)]
    check('M103 arm (a) beats arm (b) at four of five readable rungs', 4,
          len(a_beats_b), places=0)
    check('M103 arm (a) beats arm (b) at the 1024 rung', True,
          1024 in a_beats_b)
    check('M103 ledger records the single-seed artifact', True,
          'single-seed artifact' in ledger)
    check('plan 2.9.6 finding 2 is contradicted in place', True,
          'contradicted after execution \u2014 this finding was single-seed'
          in text)

    # C103.6: the compute ledger, recomputed.
    infer = {}
    for arm, atoms in [('a_random_patches', 1024), ('c_discriminative', 512)]:
        row = next(a for a in next(
            b for b in m103['seeds'][0]['budgets'] if b['atoms'] == atoms
        )['arms'] if a['arm'] == arm)
        infer[(arm, atoms)] = row['inference_macs_per_image']
    a_total = infer[('a_random_patches', 1024)]['total']
    c_total = infer[('c_discriminative', 512)]['total']
    check('M103 arm (a) @1024 inference MACs', 89165584, a_total, places=0)
    check('M103 arm (c) @512 inference MACs', 48834320, c_total, places=0)
    check('M103 ledger quotes the arm (a) inference total', True,
          f'**{a_total:,}** |' in ledger)
    check('M103 ledger quotes the arm (c) inference total', True,
          f'**{c_total:,}** |' in ledger)
    check('M103 inference MAC ratio quoted as 1.83x', 1.83, a_total / c_total,
          places=2)
    train_c = next(a for a in next(
        b for b in m103['seeds'][0]['budgets'] if b['atoms'] == 512
    )['arms'] if a['arm'] == 'c_discriminative')['training_ledger']
    check('M103 arm (c) training MACs quoted as 5.334 TMAC', 5.334,
          train_c['training_macs'] / 1e12, places=3)
    check('M103 break-even quoted as 132,244 inferences', 132244,
          round(train_c['training_macs'] / (a_total - c_total)), places=0)
    check('M103 ledger quotes the break-even figure', True,
          'stated plainly: '
          f'{round(train_c["training_macs"] / (a_total - c_total)):,} '
          'inferences' in ledger)
    check('M103 break-even in CIFAR test passes quoted as 13.2', 13.2,
          train_c['training_macs'] / (a_total - c_total) / 10000, places=1)
    for arm in ('a_random_patches', 'd_random_projections'):
        row = next(a for a in next(
            b for b in m103['seeds'][0]['budgets'] if b['atoms'] == 512
        )['arms'] if a['arm'] == arm)
        check(f'M103 {arm} pays no training cost', 0,
              row['training_ledger']['training_macs'], places=0)
    check('M103 ledger reports training and inference separately', True,
          'never netted' in ledger or 'not netted against inference' in ledger)

    # C103.7: the corrected instrument check.
    ins = m103['instrument']
    check('M103 instrument verdict', 'ok', ins['verdict'])
    check('M103 instrument monotone', True, ins['monotone_in_atom_count'])
    check('M103 instrument clears the internal floor', True,
          ins['clears_floor'])
    check('M103 internal floor is 2.9.3 own reading', 0.6223,
          ins['floor_accuracy'], places=4)
    check('M103 floor reading quoted as 0.6839', 0.6839, ins['floor_reading'],
          places=4)
    check('M103 encode is bitwise repeatable', True,
          m103['encode_bitwise_repeatable'])
    check('M103 anchor gates nothing', True, ins['anchor_gates_nothing'])
    check('M103 anchor is the 4000-feature Coates figure', 4000,
          ins['anchor_features'], places=0)
    check('M103 ledger states the anchor is not an operand', True,
          'anchor and not an' in ascii_ledger)
    check('M103 float64 control absolute difference', 1.22e-4,
          m103['float64_control']['max_absolute_difference'], places=6)
    check('M103 float64 control relative difference', 3.24e-7,
          m103['float64_control']['max_relative_difference'], places=9)

    # C103.8: the void rung and the regularisation disclosure.
    void_rung = next(b for b in m103['seeds'][0]['budgets']
                     if b['atoms'] == 2048)
    check('M103 2048 rung rows per fitted dimension', 6.1,
          void_rung['fit_samples_per_fitted_dimension'], places=1)
    check('M103 2048 rung is marked inadequate', False,
          void_rung['sample_adequate'])
    check('M103 2048 rung is below the floor of 10', True,
          void_rung['fit_samples_per_fitted_dimension'] < 10.0)
    top_selected = [int(b) for b, r in m103['regularisation_by_budget'].items()
                    if r['grid_top_selected']]
    check('M103 grid top was selected at 64, 128 and 256', '64,128,256',
          ','.join(str(b) for b in sorted(top_selected)))
    check('M103 ledger discloses the truncated grid at those budgets', True,
          'top of the' in ledger and 'grid (0.3) was selected' in ledger)
    check('M103 ledger records the void rung as not read', True,
          'void, not negative' in ledger)

    # Standing restrictions the ledger entry must carry.
    check('M103 ledger states it is a Q2 milestone', True,
          '**Q2** (efficiency), on the M103 \u2192 M99 chain' in ledger)
    check('M103 ledger forbids CIFAR/DomainNet comparison', True,
          'No CIFAR-to-DomainNet comparison' in ledger)
    check('M103 ledger disclaims novelty', True,
          'not a new idea' in ledger)
    check('M103 evidence pins the registration section', True,
          '7.9' in m103['registered_in'])

    # The prior-art audit recorded against C103.1 after it was sealed.
    check('M103 ledger discloses the prior art that subsumes C103.1', True,
          '**The phenomenon C103.1 measures is already published, in a '
          'stronger form' in ledger)
    check('M103 ledger names the ridge-leverage separation', True,
          'Avron et al., ICML 2017 (arXiv:1804.09893)' in ledger)
    check('M103 ledger records that C103.3 was predicted', True,
          "**C103.3 is *predicted*, not merely volunteered.**" in ledger)
    check('M103 ledger records the label-free reading', True,
          '**The stronger published mechanism is label-free.**' in ledger)
    check('M103 ledger discloses the Sinha & Duchi fetch failure', True,
          '**Figures not verified by fetch**' in ledger)
    check('M103 ledger states no figure is withdrawn by the audit', True,
          '**What does not change.** No figure in this entry is withdrawn'
          in ledger)
    check('M103 ledger binds the prior-art disclosure to C103.1', True,
          '8. **The C103.1 prior-art disclosure travels with C103.1.'
          in ledger)
    check('M103 ledger keeps the amendment beside the claim, not in it', True,
          ledger.index('**Not entitled.** That this transfers')
          < ledger.index('**[Amendment, recorded after a prior-art audit.'))
    check('ledger records M104-M106 as registered but not run', True,
          '(\u00a77.11) and **M106** (\u00a77.12) are registered and '
          '**not yet run**' in ledger)
    check('ledger records the M105 and M106 conditionality', True,
          'M105 is conditional on M104 surviving all three of its kill'
          in ledger)
    check('ledger records the dense-comparator defect', True,
          '**no v15 milestone as registered compares anything to a\ndense '
          'network**' in ledger)
    check('ledger records the M104 execution-time amendments', True,
          '**M104 execution-time amendments, recorded before the sealed run '
          'started.**' in ledger)
    check('ledger states the amendments predate the run', True,
          '**in place, before any M104 figure existed**' in ledger)
    check('ledger repeats the 3,455 against 3,072 atom spend', True,
          '**3,455** atoms against arm (a)\'s\n**3,072**' in ledger)
    check('ledger repeats the factor of six for the generalists', True,
          'factor of **six** for a generalist' in ledger)
    check('ledger names arm (e) and kill switch 4', True,
          '**arm (e)** and **kill switch 4** are added' in ledger)
    check('ledger repeats the quickdraw traffic share', True,
          'quickdraw holds **29.46%** of train rows' in ledger)
    check('ledger records the per-class reading as reported beside', True,
          'with the per-class reading **reported beside it**' in ledger)
    check('ledger records the head change and its restriction', True,
          '\u00a77.10 restriction 7 records the head change from M103' in ledger)
    check('ledger states the amendments make M104 harder, not easier', True,
          '**Each of the four makes M104 harder\nfor arm (b) to pass, not '
          'easier**' in ledger)
    check('ledger states the cap binds against the nulls', True,
          'the \u00a75.3 cap binds against the nulls\nrather than against arm (b)'
          in ledger)
    check('ledger figures agree with the plan figures', True,
          all(f in ledger and f in text for f in ('**3,455**', '**3,072**',
                                                  '**29.46%**')))
    check('ledger records the M104 fifth amendment', True,
          'A **fifth** amendment was forced by running the instrument once on '
          'a smoke' in ledger)
    check('ledger names the defect the fifth amendment fixes', True,
          'still fall below the floor' in ledger
          and 'while every cap was respected' in ledger)
    check('ledger states the defect is disclosed rather than quietly repaired',
          True,
          'defect is recorded rather than quietly repaired because the guard '
          'it broke' in ledger)
    check('ledger cross-references 7.10 amendment 5', True,
          '\u00a77.10 execution-time amendment 5 fixes it' in ledger)
    check('ledger marks the 3,455 as an anchor and not a prediction', True,
          '**Provenance of the 3,455.**' in ledger
          and '**not a prediction of the run**' in ledger)
    check('ledger records M107 and when it was registered', True,
          '**M107 (plan \u00a77.14), registered while M104 was running and '
          'before any M104' in ledger)
    check('ledger records M107 as unconditional on M104', True,
          '**unconditional** \u2014 it does not depend on M104\u0027s outcome'
          in ledger)
    check('ledger records the M107 execution-time amendments', True,
          '**M107 execution-time amendments, recorded before the run and '
          'before any M107' in ledger)
    check('7.10 amendment 6 keeps M104 MAC counting analytic', True,
          '   **analytic** — `training_macs` and `rank_measurement_macs` are '
          'counted, not' in text)
    check('ledger records the M104 seconds field as not an operand', True,
          'evidence **is not an operand and may never be quoted as one**: the '
          'M107 pixel' in ledger)
    check('ledger records the M104 refutation', True,
          '**M104 result (plan \u00a77.10). Outcome letter: R \u2014 refuted. '
          'Four of five kill' in ledger)
    check('ledger closes M105 and M106', True,
          '**M105 and M106 are closed, not pending.** Both were registered '
          'conditional on' in ledger)
    check('ledger marks the stale not-yet-run claim superseded', True,
          'three are Q2. **[superseded in place, \u00a75.10: M104 has since run '
          'to completion.' in ledger)
    check('ledger records the M107 sealed-directory contamination', True,
          '**2,760-row** `evidence.json` into the **sealed** M107 directory, '
          'where the' in ledger)
    check('ledger records why M107 was not run beside M104', True,
          '**7.1 of 16** cores, and filling the rest with M107 was rejected '
          'because both' in ledger)
    check('ledger records the M107 single seed as not making it harder', True,
          'does **not** make the milestone harder and is recorded as a '
          'limitation of every' in ledger)
    check('ledger records the M107 mixture ladder narrowing', True,
          '**mixture ladder therefore runs only at 128 and 256**' in ledger
          and '**Kill switch 3 is\ndecidable at two budgets instead of six**'
          in ledger)
    check('ledger records the three M107 asymmetries', True,
          all(f in ledger for f in ('**LVD-142M**', '**32\u00d732**',
                                    '**oracle** routing', '**(d5)**')))
    check('ledger keeps prohibition 27 in force until M107 runs', True,
          'Prohibition 27 stays in force\nuntil M107 has actually been run'
          in ledger)

# --- section 2.9 scoping probes: figures quoted in the plan ----------------
# These are unsealed observations (prohibition 23), so there is no evidence file
# to verify them against. What the verifier CAN enforce is that every figure the
# plan quotes is internally consistent and carries its required qualification.
sec29 = re.search(r'## 2\.9 What was measured after M102(.*?)\n## 3\. ',
                  text, re.S)
check('section 2.9 exists', True, sec29 is not None)
if sec29:
    body = sec29.group(1)
    # The document uses the typographic minus sign U+2212 in its tables.
    ascii_body = body.replace('\u2212', '-')
    check('2.9 marked inadmissible under 2.4', True,
          'scoping observations under §2.4' in re.sub(r'\s+', ' ', body))
    # 2.9.1 ensemble deltas must equal the difference of the quoted columns.
    for dims, cheap, full, ens in [(32, 0.5700, 0.6322, 0.6257),
                                   (64, 0.6135, 0.6322, 0.6338)]:
        quoted = re.search(rf'\| {dims} \| {cheap:.4f} \| {full:.4f} \| '
                           rf'\*\*{ens:.4f}\*\* \*\(([-+][0-9.]+)\)\*',
                           ascii_body)
        check(f'2.9.1 delta row present at {dims} dims', True, quoted is not None)
        if quoted:
            check(f'2.9.1 ensemble delta at {dims} dims', round(ens - full, 4),
                  float(quoted.group(1)), places=4)
    check('2.9.1 states the ensemble captures almost none of the gap', True,
          'captures almost' in body)
    # 2.9.2 relative compute must equal the MAC ratio the same table quotes.
    # 2.9.2 relative compute must equal the MAC ratio, computed from exact MAC
    # counts rather than from the rounded GMAC column the table displays. Both
    # the arithmetic and the numbers the plan actually prints are checked, so a
    # corrupted table cell cannot pass on correct arithmetic alone.
    small_macs = 12 * (12 * 257 * 384 ** 2 + 2 * 257 ** 2 * 384)
    for name, layers, dims, gmacs, rel, acc in [
            ('dinov2-small', 12, 384, 6.07, 1.00, 0.6322),
            ('dinov2-base', 12, 768, 23.05, 3.80, 0.5177),
            ('dinov2-large', 24, 1024, 80.86, 13.33, 0.6652)]:
        exact = layers * (12 * 257 * dims ** 2 + 2 * 257 ** 2 * dims)
        check(f'2.9.2 GMACs for {name}', gmacs, exact / 1e9, places=2)
        check(f'2.9.2 relative compute for {name}', rel, exact / small_macs,
              places=2)
        row = re.search(rf'\| {name} \| {dims} \| ([0-9.]+) \| ([0-9.]+)'
                        rf'\u00d7 \|', ascii_body)
        check(f'2.9.2 table row present for {name}', True, row is not None)
        if row:
            check(f'2.9.2 table GMACs cell for {name}', float(row.group(1)),
                  exact / 1e9, places=2)
            check(f'2.9.2 table relative cell for {name}', float(row.group(2)),
                  exact / small_macs, places=2)
    check('2.9.2 discloses the INT8 void', True,
          'VOID' in body and '1.194' in body)
    check('2.9.2 states base is void not negative', True,
          'void, not negative' in body)
    # 2.9.3 learning gain must equal random minus k-means at every rung.
    for atoms, rnd, km, gain in [(64, 0.5240, 0.5169, -0.0071),
                                 (128, 0.5614, 0.5525, -0.0089),
                                 (256, 0.5819, 0.5800, -0.0019),
                                 (512, 0.6129, 0.6059, -0.0070),
                                 (1024, 0.6339, 0.6223, -0.0116)]:
        check(f'2.9.3 learning gain at {atoms} atoms', gain, km - rnd,
              places=4)
        check(f'2.9.3 random beats k-means at {atoms} atoms', True, rnd > km)
        check(f'2.9.3 quotes the gain at {atoms} atoms', True,
              f'**{gain:.4f}**' in ascii_body)
    check('2.9.3 discloses single-seed status', True, 'Single seed' in body)
    check('2.9.3 discloses the CIFAR/DomainNet non-comparability', True,
          'not comparable to any v13/v14/v15 DomainNet figure' in
          re.sub(r'\s+', ' ', body))
    check('2.9.3 discloses Thiry prior art for random-beats-learned', True,
          'or even a learning procedure' in body)
    check('2.9.3 forbids claiming random-beats-learned as this program\'s', True,
          'as a finding of this program' in re.sub(r'\s+', ' ', body))
    check('2.9.3 registered consequence marked superseded', True,
          'superseded by §2.9.4' in body)
    # 2.9.4: the discriminative arm's gain must equal its own table's columns,
    # must exceed the random arm's seed spread by the multiple the text claims,
    # and must be quoted with the shrinking-with-budget direction intact.
    gains = {}
    spreads = {}
    for atoms, rnd, km, disc, gain, spread in [
            (128, 0.5521, 0.5543, 0.5813, 0.0292, 0.0078),
            (256, 0.5881, 0.5798, 0.6041, 0.0160, 0.0053)]:
        row = re.search(rf'\| {atoms} \| {rnd:.4f} \| {km:.4f} \| '
                        rf'\*\*{disc:.4f}\*\* \| \*\*\+([0-9.]+)\*\* \| '
                        rf'([0-9.]+) \|', ascii_body)
        check(f'2.9.4 table row present at {atoms} atoms', True, row is not None)
        if row:
            check(f'2.9.4 quoted gain at {atoms} atoms', float(row.group(1)),
                  round(disc - rnd, 4), places=4)
            check(f'2.9.4 quoted seed spread at {atoms} atoms', spread,
                  float(row.group(2)), places=4)
        check(f'2.9.4 discriminative beats random at {atoms} atoms', True,
              disc > rnd)
        check(f'2.9.4 discriminative beats k-means at {atoms} atoms', True,
              disc > km)
        gains[atoms] = disc - rnd
        spreads[atoms] = spread
    check('2.9.4 gain shrinks as the budget grows', True,
          gains[256] < gains[128])
    # The prose band must be the rounded extremes of the table's own ratios,
    # not a number chosen independently of them.
    ratios = sorted(round(gains[a] / spreads[a], 1) for a in gains)
    band = re.search(r'by ([0-9.]+)\u2013([0-9.]+)\u00d7 the random arm',
                     re.sub(r'\s+', ' ', ascii_body))
    check('2.9.4 quotes a gain-to-spread band', True, band is not None)
    if band:
        check('2.9.4 lower end of the quoted band', float(band.group(1)),
              ratios[0], places=1)
        check('2.9.4 upper end of the quoted band', float(band.group(2)),
              ratios[-1], places=1)
    check('2.9.4 every cell clears the lower end of its own band', True,
          all(gains[a] / spreads[a] >= ratios[0] for a in gains))
    check('2.9.4 states the gain shrinks', True,
          'shrinks as the budget grows' in body)
    check('2.9.4 all six cells claim is stated', True,
          'All six seed-budget cells' in body)
    # The ~232-atom equivalent and the 1.8x reading must follow from the table.
    frac = (0.5813 - 0.5521) / (0.5881 - 0.5521)
    check('2.9.4 interpolated equivalent random atom count', 232.0,
          128 + frac * (256 - 128), places=0)
    check('2.9.4 quoted interpolation fraction', 0.81, frac, places=2)
    check('2.9.4 quoted atom-count ratio', 1.8, (128 + frac * 128) / 128,
          places=1)
    # The mechanism table must actually show two-tailed compression for k-means.
    norms = {}
    for name, mean, med, p5, p95 in [
            ('random patches', 1.296, 1.265, 0.369, 2.292),
            ('k-means centroids', 1.209, 1.179, 0.481, 2.064),
            ('the data itself', 1.248, 1.193, 0.372, 2.308)]:
        row = re.search(rf'\| {re.escape(name)} \| {mean:.3f} \| {med:.3f} \| '
                        rf'\*?\*?{p5:.3f}\*?\*? \| \*?\*?{p95:.3f}\*?\*? \|',
                        ascii_body)
        check(f'2.9.4 atom-norm row present for {name}', True, row is not None)
        norms[name] = (p5, p95)
    check('2.9.4 k-means lower tail is compressed upward', True,
          norms['k-means centroids'][0] > norms['the data itself'][0])
    check('2.9.4 k-means upper tail is compressed downward', True,
          norms['k-means centroids'][1] < norms['the data itself'][1])
    check('2.9.4 random tails track the data more closely than k-means does',
          True,
          abs(norms['random patches'][0] - norms['the data itself'][0])
          < abs(norms['k-means centroids'][0] - norms['the data itself'][0]))
    check('2.9.4 discloses the unpaid training cost of selection', True,
          'a *training* cost the random arm does not pay' in body)
    check('2.9.4 registers the candidate-pool artefact limitation', True,
          'may be an artefact of pool size' in re.sub(r'\s+', ' ', body))
    check('2.9.4 states no gate or kill switch changed', True,
          'none of which is loosened by it' in body)
    # 2.9.5: the re-verification must actually contradict the registered bar.
    for setting, figure in [('2K patches, linear head', 0.8232),
                            ('16K patches, linear head', 0.8562),
                            ('2K patches, one hidden layer', 0.8853)]:
        check(f'2.9.5 quotes the re-verified figure for {setting}', True,
              f'{figure:.4f}' in ascii_body)
    check('2.9.5 the registered linear bar is not reproduced at either size',
          True, abs(0.8232 - 0.869) > 0.001 and abs(0.8562 - 0.869) > 0.001)
    check('2.9.5 the hidden-layer figure does reproduce the registered 0.885',
          True, abs(0.8853 - 0.885) < 0.001)
    check('2.9.5 marks the bar unconfirmed', True,
          'marked **unconfirmed**' in body)
    check('2.9.5 states M103 is not blocked', True,
          'does not block M103' in body or 'not block M103' in body)
    # 2.9.6: the instrumentation run. Its table must be internally consistent,
    # the quoted +0.0133 must be the table's own subtraction, the reversal it
    # claims must actually be present, and the norm table must show the shift
    # the prose reads off it.
    inst = {}
    for arm, accuracy, bold in [('(a) random patches', 0.6818, False),
                                ('(b) k-means', 0.6856, False),
                                ('(c) discriminative selection', 0.6951, True),
                                ('(d) random projections', 0.6760, False)]:
        cell = f'**{accuracy:.4f}**' if bold else f'{accuracy:.4f}'
        check(f'2.9.6 table row present for {arm}', True,
              f'| {arm} | {cell} |' in ascii_body)
        inst[arm] = accuracy
    check('2.9.6 quoted gain over random is the table subtraction', 0.0133,
          inst['(c) discriminative selection'] - inst['(a) random patches'],
          places=4)
    check('2.9.6 quotes the gain in prose', True, '**+0.0133**' in ascii_body)
    check('2.9.6 discriminative still leads at 1024 atoms', True,
          inst['(c) discriminative selection'] > inst['(b) k-means']
          > inst['(a) random patches'])
    check('2.9.6 random projections remain the weakest arm', True,
          inst['(d) random projections'] == min(inst.values()))
    check('2.9.6 the 2.9.3 ordering really is reversed here', True,
          inst['(b) k-means'] > inst['(a) random patches'])
    check('2.9.6 quotes the k-means figure used by the corrected floor check',
          True, '0.6856' in ascii_body)
    # The gain must still be shrinking with budget across 2.9.4 and 2.9.6,
    # because that is the direction the prose reports and the reason M103 is
    # still needed to settle it.
    check('2.9.6 gain is smaller than 2.9.4 at 256 atoms', True,
          round(inst['(c) discriminative selection']
                - inst['(a) random patches'], 4) < 0.0160)
    check('2.9.6 gain is still positive at 1024 atoms', True,
          inst['(c) discriminative selection'] - inst['(a) random patches'] > 0)
    norms_1024 = {}
    for name, mean, med, p5, p95 in [
            ('(a) random patches', 1.248, 1.212, 0.342, 2.285),
            ('(b) k-means', 1.240, 1.205, 0.500, 2.119),
            ('(c) discriminative', 1.734, 1.683, 1.032, 2.683),
            ('(d) random projections', 1.240, 1.209, 0.357, 2.209)]:
        row = re.search(rf'\| {re.escape(name)} \| \*?\*?{mean:.3f}\*?\*? \| '
                        rf'{med:.3f} \| \*?\*?{p5:.3f}\*?\*? \| '
                        rf'\*?\*?{p95:.3f}\*?\*? \|', ascii_body)
        check(f'2.9.6 atom-norm row present for {name}', True, row is not None)
        norms_1024[name] = (mean, med, p5, p95)
    check('2.9.6 k-means lower tail is compressed upward at 1024', True,
          norms_1024['(b) k-means'][2] > norms_1024['(a) random patches'][2])
    check('2.9.6 k-means upper tail is compressed downward at 1024', True,
          norms_1024['(b) k-means'][3] < norms_1024['(a) random patches'][3])
    check('2.9.6 the discriminative dictionary is shifted up at every quantile',
          True,
          all(norms_1024['(c) discriminative'][i]
              > norms_1024['(a) random patches'][i] for i in range(4)))
    check('2.9.6 discriminative 5th percentile clears the random median', True,
          norms_1024['(c) discriminative'][2]
          > norms_1024['(a) random patches'][1] * 0.85)
    check('2.9.6 random projections track random patches as designed', True,
          abs(norms_1024['(d) random projections'][0]
              - norms_1024['(a) random patches'][0]) < 0.05)
    check('2.9.6 declares itself inadmissible under 2.4', True,
          'inadmissible under §2.4' in body)
    check('2.9.6 discloses that the top grid value won for every arm', True,
          'won\n for all four arms' in body.replace('**', '')
          or 'for all four arms' in body)

# --- MAC arithmetic quoted in 2.9.2 ---------------------------------------
def vit_macs(layers, dims, patches=257):
    return layers * (12 * patches * dims ** 2 + 2 * patches ** 2 * dims)


check('ViT-S/14 MACs reproduce the registered v13 figure', 6065759232,
      vit_macs(12, 384), places=0)
check('ViT-B/14 GMACs quoted in 2.9.2', 23.05, vit_macs(12, 768) / 1e9,
      places=2)
check('ViT-L/14 GMACs quoted in 2.9.2', 80.86, vit_macs(24, 1024) / 1e9,
      places=2)

# --- M103 sample-floor arithmetic registered in 7.9 restriction 4 ---------
check('M103 2048-atom rung is below the sample floor', True,
      50000 / (4 * 2048) < 10.0)
check('M103 2048-atom rows per feature', 6.1, 50000 / (4 * 2048), places=1)
check('M103 1024-atom rung clears the sample floor', True,
      50000 / (4 * 1024) >= 10.0)

# --- the corrected instrument check registered in 7.9 via 2.9.6 -----------
# The defect is arithmetic, not editorial: the anchor is a 4000-FEATURE figure
# and the top readable rung is 1024 atoms. Verify that the two are not
# comparable, and that the replacement floor is one this program itself set.
check('Coates anchor feature count exceeds M103 top readable feature count',
      True, 4000 > 4 * 1024 / 4)
check('M103 top readable rung is 1024 atoms', 1024,
      max(b for b in [64, 128, 256, 512, 1024, 2048]
          if 50000 / (4 * b) >= 10.0), places=0)
check('the instrumentation reading falls short of the withdrawn anchor', True,
      0.6856 < 0.796)
check('the withdrawn anchor was outside the registered tolerance anyway', True,
      abs(0.6856 - 0.796) > 0.06)
check('the replacement floor is cleared by the instrumentation run', True,
      0.6856 > 0.6223)
check('the replacement floor comes from this program, not a citation', True,
      '0.6223' in text)

# --- 2.9.7 rank probes, recomputed from their artifacts -------------------
# Every figure 2.9.7 quotes is recomputed here from the probe JSON, and each
# recomputation is paired with a presence check on the plan text. Without the
# presence check a negative control on the prose could not fire.
PROBES = ROOT / 'logs/results/v15/rank_probes'
p1 = json.load(open(PROBES / 'probe1_rank_vs_budget.json'))
p2 = json.load(open(PROBES / 'probe2_rank_by_class.json'))
p3 = json.load(open(PROBES / 'probe3_rank_by_domain.json'))
p4 = json.load(open(PROBES / 'probe4_intrinsic_fingerprint.json'))

# The probe artifacts are unsealed, so the index is the only thing standing
# between a quoted figure and a silently edited JSON.
import hashlib  # noqa: E402

_probe_index = json.load(open(PROBES / 'artifact_index.json'))
for _art in _probe_index['artifacts']:
    check(f'rank-probe artifact {_art["path"]} matches its index digest',
          _art['sha256'],
          hashlib.sha256((PROBES / _art['path']).read_bytes()).hexdigest())
    check(f'rank-probe artifact {_art["path"]} matches its index size',
          _art['bytes'], (PROBES / _art['path']).stat().st_size, places=0)
check('rank-probe index covers every probe artifact', 4,
      len(_probe_index['artifacts']), places=0)
check('rank-probe index pins its registration section', '2.9.7',
      _probe_index['registered_in'])
check('rank-probe index records the probes as unsealed', True,
      'unsealed scoping probes' in _probe_index['status'])
for _p, _name in [(p1, 'probe 1'), (p2, 'probe 2'), (p3, 'probe 3'),
                  (p4, 'probe 4')]:
    check(f'{_name} artifact declares itself not a milestone', True,
          'not a milestone' in _p['note'])
check('2.9.7 points at the runners that produced its artifacts', True,
      'experiments/tier4/rank_probes/' in text)

p1_curve = {c['atoms']: c for c in p1['curve']}
for atoms, quoted_rank, quoted_frac in [
    (64, 37.194, 0.14529), (128, 49.055, 0.09581), (256, 72.344, 0.07065),
    (512, 97.907, 0.04781), (1024, 129.829, 0.03170), (2048, 177.822, 0.02171),
]:
    check(f'2.9.7 probe 1 RankMe at {atoms} atoms', quoted_rank,
          p1_curve[atoms]['rankme'], places=3)
    check(f'2.9.7 probe 1 useful fraction at {atoms} atoms', quoted_frac,
          p1_curve[atoms]['fraction_of_ambient'], places=5)
    check(f'2.9.7 probe 1 quotes the {atoms}-atom row', True,
          f'| {atoms} | {4 * atoms} | {quoted_rank:.3f} | {quoted_frac:.5f} |'
          in text)

# The exponent the prose quotes is a least-squares fit, not an endpoint ratio,
# and the two differ in the third decimal. Recompute the fit the prose names.
import math  # noqa: E402

_la = [math.log(c['atoms']) for c in p1['curve']]
_lr = [math.log(c['rankme']) for c in p1['curve']]
_n = len(_la)
_mla = sum(_la) / _n
_mlr = sum(_lr) / _n
_slope = (sum((a - _mla) * (r - _mlr) for a, r in zip(_la, _lr))
          / sum((a - _mla) ** 2 for a in _la))
check('2.9.7 probe 1 least-squares rank exponent', 0.4553, _slope, places=4)
check('2.9.7 quotes the rank exponent', True, '0.4553' in text)
check('2.9.7 probe 1 useful-fraction exponent is the rank exponent minus one',
      -0.545, _slope - 1.0, places=3)
check('2.9.7 quotes the useful-fraction exponent', True,
      'atoms^\u22120.545' in text)
_growth = [g['rankme_ratio'] for g in p1['growth']]
check('2.9.7 probe 1 slowest doubling', 1.319, min(_growth), places=3)
check('2.9.7 probe 1 fastest doubling', 1.475, max(_growth), places=3)
check('2.9.7 quotes the per-doubling band', True,
      '\u00d71.319\u20131.475 per doubling' in text)
check('2.9.7 probe 1 useful-fraction collapse across the sweep', 6.69,
      p1_curve[64]['fraction_of_ambient'] / p1_curve[2048]['fraction_of_ambient'],
      places=2)
check('2.9.7 quotes the collapse factor', True, '**6.69\u00d7**' in text)
check('2.9.7 records that probe 1 refuted the saturation hypothesis', True,
      'refutes the hypothesis that rank saturation' in text)

check('2.9.7 probe 2 mean class rank', 63.856, p2['mean_class_rankme'],
      places=3)
check('2.9.7 probe 2 mean control rank', 70.118, p2['mean_control_rankme'],
      places=3)
check('2.9.7 probe 2 specialisation ratio', 0.9107, p2['specialisation_ratio'],
      places=4)
check('2.9.7 quotes the class specialisation ratio', True,
      '**0.9107**' in text)
check('2.9.7 probe 2 gap exceeds the control spread', True,
      (p2['mean_control_rankme'] - p2['mean_class_rankme']) > 0.63)
check('2.9.7 quotes the class gap', True, 'specialist gap is\n6.26' in text)

p3_by_domain = {d['domain']: d for d in p3['per_domain']}
_p3_control = p3['mean_control_rankme']
check('2.9.7 probe 3 control mean', 47.247, _p3_control, places=3)
check('2.9.7 quotes the domain control mean', True, '**47.247**' in text)
for domain, quoted_rank, quoted_ratio in [
    ('infograph', 55.286, 1.170), ('clipart', 55.164, 1.168),
    ('painting', 52.585, 1.113), ('real', 52.558, 1.112),
    ('sketch', 22.460, 0.475), ('quickdraw', 8.752, 0.185),
]:
    check(f'2.9.7 probe 3 RankMe for {domain}', quoted_rank,
          p3_by_domain[domain]['rankme'], places=3)
    check(f'2.9.7 probe 3 control ratio for {domain}', quoted_ratio,
          p3_by_domain[domain]['rankme'] / _p3_control, places=3)
    emph = '**' if domain == 'quickdraw' else ''
    check(f'2.9.7 quotes the {domain} row', True,
          f'| {emph}{domain}{emph} | {emph}{quoted_rank:.3f}{emph} '
          f'| {emph}{quoted_ratio:.3f}{emph} |' in text)
_p3_ranks = [d['rankme'] for d in p3['per_domain']]
check('2.9.7 probe 3 domain spread', 6.32, max(_p3_ranks) / min(_p3_ranks),
      places=2)
check('2.9.7 quotes the domain spread', True, '**6.32\u00d7**' in text)
check('2.9.7 probe 3 quickdraw shortfall against the control', 5.40,
      _p3_control / min(_p3_ranks), places=2)
check('2.9.7 quotes the quickdraw shortfall', True, '**5.40\u00d7 below**' in text)
check('2.9.7 probe 3 mean ratio', 0.8706, p3['specialisation_ratio'],
      places=4)
check('2.9.7 quotes the mean ratio it then disowns', True,
      'Mean ratio **0.8706**' in text)
check('2.9.7 probe 3 four domains sit above the control', 4,
      sum(1 for r in _p3_ranks if r > _p3_control), places=0)
check('2.9.7 states how many domains sit above the control', True,
      '**four of six domains sit above it**' in text)
check('2.9.7 discloses the 32x32 resolution limitation', True,
      'measured at 32\u00d732, and a downsample plausibly penalises line art'
      in text)

_p4id = p4['domain_identification']
check('2.9.7 probe 4 intrinsic domain accuracy', 0.49278,
      _p4id['intrinsic_4_scalars'], places=5)
check('2.9.7 probe 4 pooled domain accuracy', 0.69056,
      _p4id['pooled_features'], places=5)
check('2.9.7 probe 4 chance is six-way', 6, round(1 / _p4id['chance']),
      places=0)
check('2.9.7 quotes both router accuracies', True,
      '**4 intrinsic scalars \u2192\n0.49278**' in text
      and '**0.69056**' in text)
check('2.9.7 probe 4 recovered fraction of the pooled probe', 0.71,
      _p4id['intrinsic_4_scalars'] / _p4id['pooled_features'], places=2)
check('2.9.7 quotes the recovered fraction', True, '**71%**' in text)
check('2.9.7 probe 4 dimension ratio', 512,
      _p4id['pooled_dimension'] // 4, places=0)
check('2.9.7 quotes the dimension ratio', True, '**512\u00d7** fewer' in text)
for domain, quoted in [
    ('clipart', [43.52, -2.838, 1.665, 0.820]),
    ('infograph', [47.74, -2.705, 1.493, 0.819]),
    ('painting', [45.15, -2.782, 1.296, 0.841]),
    ('quickdraw', [30.76, -1.247, 0.972, 0.693]),
    ('real', [41.93, -3.002, 1.475, 0.846]),
    ('sketch', [31.40, -2.824, 1.038, 0.798]),
]:
    actual = p4['per_domain_intrinsic_means'][domain]
    check(f'2.9.7 probe 4 RankMe mean for {domain}', quoted[0], actual[0],
          places=2)
    check(f'2.9.7 probe 4 alpha for {domain}', quoted[1], actual[1], places=3)
    check(f'2.9.7 probe 4 scale for {domain}', quoted[2], actual[2], places=3)
    check(f'2.9.7 probe 4 top-10 mass for {domain}', quoted[3], actual[3],
          places=3)
_alphas = {d: v[1] for d, v in p4['per_domain_intrinsic_means'].items()}
check('2.9.7 probe 4 quickdraw is the alpha outlier', 'quickdraw',
      max(_alphas, key=lambda d: _alphas[d]))
check('2.9.7 quotes the quickdraw alpha row', True,
      '| quickdraw | 30.76 | **\u22121.247** | 0.972 | **0.693** |' in text)
_drift = p4['drift_under_growth']
check('2.9.7 probe 4 pooled agreement', 0.90659,
      _drift['pooled']['assignment_agreement'], places=5)
check('2.9.7 probe 4 pooled drift', 9.341,
      100 * _drift['pooled']['assignments_moved'], places=3)
check('2.9.7 probe 4 intrinsic agreement', 0.96664,
      _drift['intrinsic']['assignment_agreement'], places=5)
check('2.9.7 probe 4 intrinsic drift', 3.336,
      100 * _drift['intrinsic']['assignments_moved'], places=3)
check('2.9.7 probe 4 held-out rows in the drift test', 1199,
      _drift['pooled']['rows'], places=0)
check('2.9.7 quotes the drift rows', True, '1,199\nheld-out images' in text)
check('2.9.7 quotes the pooled drift row', True,
      '| pooled features, 2048-d | 0.90659 | **9.341%** |' in text)
check('2.9.7 quotes the intrinsic drift row', True,
      '| intrinsic fingerprints, 4-d | 0.96664 | **3.336%** |' in text)
check('2.9.7 probe 4 stability advantage', 2.80,
      _drift['pooled']['assignments_moved']
      / _drift['intrinsic']['assignments_moved'], places=2)
check('2.9.7 quotes the stability advantage', True,
      '**2.80\u00d7 more stable**' in text)
check('2.9.7 probe 4 grew from four seen domains', 4, len(p4['seen_domains']),
      places=0)
check('2.9.7 records the within-versus-across-image distinction', True,
      'Expert sizing is set by the\nacross-image quantity' in text)
check('2.9.7 states the probes measure no accuracy', True,
      'None of them measures accuracy' in text)

document_failures = []
document_checks_run = []

# --- M104 allocation arithmetic quoted in 7.10's execution-time amendments --
# Every figure the amendment block states is recomputed here from the corpus row
# counts, which are the only inputs it depends on. The allocation itself is
# recomputed with the same rule the runner uses, so a change to either the plan
# prose or the rule breaks this.
M104_TRAIN = {'clipart': 33525, 'infograph': 36023, 'painting': 50416,
              'quickdraw': 120750, 'real': 120906, 'sketch': 48212}
M104_RANK = {'clipart': 1.168, 'infograph': 1.170, 'painting': 1.113,
             'quickdraw': 0.185, 'real': 1.112, 'sketch': 0.475}
m104_names = sorted(M104_TRAIN)
m104_n = [M104_TRAIN[k] for k in m104_names]
m104_total = sum(m104_n)
m104_share = [v / m104_total for v in m104_n]
m104_caps = [v // 40 for v in m104_n]


def m104_allocate(weights, budget=512):
    """The runner's rule: sizes proportional to weights at a matched
    row-weighted total, capped by the section 5.3 floor and redistributed."""
    pinned = [False] * len(weights)
    sizes = [0] * len(weights)
    for _ in range(len(weights) + 1):
        denominator = sum(m104_share[i] * weights[i]
                          for i in range(len(weights)) if not pinned[i])
        used = sum(m104_share[i] * sizes[i]
                   for i in range(len(weights)) if pinned[i])
        scale = (budget - used) / denominator
        for i in range(len(weights)):
            if not pinned[i]:
                sizes[i] = max(16, round(scale * weights[i]))
        over = [i for i in range(len(weights))
                if not pinned[i] and sizes[i] > m104_caps[i]]
        if not over:
            break
        for i in over:
            sizes[i] = m104_caps[i]
            pinned[i] = True
    return sizes


m104_uniform = m104_allocate([1.0] * 6)
m104_ranked = m104_allocate([M104_RANK[k] for k in m104_names])
check('7.10 uniform arm spends 3072 atoms at 512', 3072, sum(m104_uniform),
      places=0)
check('7.10 rank-sized arm spends 3455 atoms at the same MACs', 3455,
      sum(m104_ranked), places=0)
check('7.10 the parameter excess is 12.5 percent', 12.5,
      100 * (sum(m104_ranked) / sum(m104_uniform) - 1), places=1)
check('7.10 both allocations match on row-weighted atoms', True,
      abs(sum(s * a for s, a in zip(m104_share, m104_ranked)) - 512) < 1.0)
check('7.10 the domain size ratio is 3.6x', 3.6,
      max(m104_n) / min(m104_n), places=1)
check('7.10 quickdraw holds 29.46 percent of train rows', 29.46,
      100 * M104_TRAIN['quickdraw'] / m104_total, places=2)
check('7.10 the clipart sample-floor cap is 838 atoms', 838,
      m104_caps[m104_names.index('clipart')], places=0)
check('7.10 the infograph sample-floor cap is 900 atoms', 900,
      m104_caps[m104_names.index('infograph')], places=0)
check('7.10 rank-sizing hits no cap at 512 atoms', True,
      all(a < c for a, c in zip(m104_ranked, m104_caps)))
check('7.10 traffic-inverse sizing DOES hit the cap, unlike rank-sizing', True,
      any(a >= c for a, c in
          zip(m104_allocate([1.0 / s for s in m104_share]), m104_caps)))
check('7.10 the two capped domains are the two highest-rank ones', True,
      {m104_names[i] for i, (a, c) in
       enumerate(zip(m104_allocate([1.0 / s for s in m104_share]), m104_caps))
       if a >= c} == set(sorted(M104_RANK, key=lambda k: -M104_RANK[k])[:2]))
check('7.10 a generalist at the plain atom sum costs 6x a mixture expert', 6.0,
      sum(m104_uniform) / (sum(m104_uniform) / 6), places=6)

# --- M104 sealed-run result quoted in 7.10 and the ledger ------------------
# Every accuracy, margin and kill-switch verdict the result block states is
# recomputed here FROM THE SEEDS, not read out of the runner's own gate. The
# runner writes a gate; recomputing it from the per-arm records is the only way
# to catch a gate that agrees with itself and disagrees with the measurement.
M104_EVIDENCE = ROOT / 'logs/results/v15/m104_experts/evidence.json'
if M104_EVIDENCE.exists():
    m104_ev = json.loads(M104_EVIDENCE.read_text(encoding='utf-8'))

    def m104_pooled(arm_name):
        vals = [arm['pooled_test_accuracy']
                for seed in m104_ev['seeds']
                for bucket in seed['budgets']
                for arm in bucket['arms']
                if arm['arm'] == arm_name and arm['sample_adequate']]
        return sum(vals) / len(vals)

    def m104_domain(arm_name, domain):
        vals = [arm['per_domain_test_accuracy'][domain]
                for seed in m104_ev['seeds']
                for bucket in seed['budgets']
                for arm in bucket['arms']
                if arm['arm'] == arm_name
                and domain in arm['per_domain_test_accuracy']]
        return sum(vals) / len(vals)

    m104_a = m104_pooled('a_uniform')
    m104_b = m104_pooled('b_rank_sized')
    m104_c1 = m104_pooled('c1_generalist_mac_matched')
    m104_c2 = m104_pooled('c2_generalist_atom_matched')
    m104_d = m104_pooled('d_random_sized')
    m104_e = m104_pooled('e_traffic_inverse')
    m104_a_vals = [arm['pooled_test_accuracy']
                   for seed in m104_ev['seeds']
                   for bucket in seed['budgets']
                   for arm in bucket['arms']
                   if arm['arm'] == 'a_uniform']
    m104_spread = max(m104_a_vals) - min(m104_a_vals)

    check('7.10 result: three seeds ran', 3, len(m104_ev['seeds']), places=0)
    check('7.10 result: 409832 train rows', 409832, m104_ev['train_rows'],
          places=0)
    check('7.10 result: 100000 test rows', 100000, m104_ev['test_rows'],
          places=0)
    check('7.10 result: routing is the oracle domain label', True,
          m104_ev['routing'] == 'oracle_domain_label')
    check('7.10 result: corpus digest 81099916e5036d1c', True,
          m104_ev['corpus_sha256'].startswith('81099916e5036d1c'))
    check('7.10 result: uniform scores 24.22 percent', 24.22,
          100 * m104_a, places=2)
    check('7.10 result: rank-sized scores 22.47 percent', 22.47,
          100 * m104_b, places=2)
    check('7.10 result: MAC-matched generalist scores 18.12 percent', 18.12,
          100 * m104_c1, places=2)
    check('7.10 result: atom-matched generalist scores 24.51 percent', 24.51,
          100 * m104_c2, places=2)
    check('7.10 result: random-sized scores 23.54 percent', 23.54,
          100 * m104_d, places=2)
    check('7.10 result: traffic-inverse scores 23.53 percent', 23.53,
          100 * m104_e, places=2)
    check('7.10 result: the margin is -1.76 points', -1.76,
          100 * (m104_b - m104_a), places=2)
    check('7.10 result: uniform seed spread is 0.16 points', 0.16,
          100 * m104_spread, places=2)
    check('7.10 result: kill switch 1 fired', True,
          (m104_b - m104_a) <= m104_spread)
    check('7.10 result: kill switch 2 fired', True,
          (m104_b - m104_d) <= 0.005)
    check('7.10 result: kill switch 4 fired', True,
          (m104_b - m104_e) <= 0.005)
    m104_best_mix = max(m104_a, m104_b, m104_d, m104_e)
    check('7.10 result: kill switch 3 fired on the atom-matched generalist',
          True, m104_c2 >= m104_best_mix - 0.005)
    check('7.10 result: kill switch 3 did NOT fire on the MAC-matched one',
          False, m104_c1 >= m104_best_mix - 0.005)
    check('7.10 result: the atom-matched generalist beats every mixture arm',
          True, m104_c2 > max(m104_a, m104_b, m104_d, m104_e))
    check('7.10 result: rank-sizing loses to BOTH its nulls', True,
          m104_b < m104_d and m104_b < m104_e)
    m104_low = ['quickdraw', 'sketch']
    m104_all_domains = ['clipart', 'infograph', 'painting', 'quickdraw',
                        'real', 'sketch']
    m104_margins = {d: m104_domain('b_rank_sized', d)
                    - m104_domain('a_uniform', d) for d in m104_all_domains}
    m104_low_mean = sum(m104_margins[d] for d in m104_low) / len(m104_low)
    m104_high = [d for d in m104_all_domains if d not in m104_low]
    m104_high_mean = sum(m104_margins[d] for d in m104_high) / len(m104_high)
    check('7.10 result: the low-rank margin is -4.84 points', -4.84,
          100 * m104_low_mean, places=2)
    check('7.10 result: the high-rank margin is +1.10 points', 1.10,
          100 * m104_high_mean, places=2)
    check('7.10 result: the mechanism inverted, low below high', True,
          m104_low_mean < m104_high_mean)
    check('7.10 result: the registered mechanism is unsupported', True,
          m104_low_mean <= 0)
    check('7.10 result: quickdraw loses 7.93 points', -7.93,
          100 * m104_margins['quickdraw'], places=2)
    check('7.10 result: sketch loses 1.76 points', -1.76,
          100 * m104_margins['sketch'], places=2)
    check('7.10 result: clipart gains 1.45 points', 1.45,
          100 * m104_margins['clipart'], places=2)
    check('7.10 result: infograph gains 0.43 points', 0.43,
          100 * m104_margins['infograph'], places=2)
    check('7.10 result: painting gains 0.76 points', 0.76,
          100 * m104_margins['painting'], places=2)
    check('7.10 result: real gains 1.77 points', 1.77,
          100 * m104_margins['real'], places=2)
    check('7.10 result: quickdraw scores 37.07 percent under uniform', 37.07,
          100 * m104_domain('a_uniform', 'quickdraw'), places=2)
    check('7.10 result: quickdraw is the most accurate domain under uniform',
          True, all(m104_domain('a_uniform', 'quickdraw')
                    >= m104_domain('a_uniform', d)
                    for d in m104_all_domains))
    check('7.10 result: quickdraw holds 29.5 percent of the train rows', 29.5,
          100 * m104_ev['train_rows_by_domain']['quickdraw']
          / m104_ev['train_rows'], places=1)
    m104_qd_atoms = [dict(zip(arm['allocation']['experts'],
                              arm['allocation']['atoms']))['quickdraw']
                     for seed in m104_ev['seeds']
                     for bucket in seed['budgets']
                     for arm in bucket['arms']
                     if arm['arm'] == 'b_rank_sized']
    check('7.10 result: rank-sizing hands quickdraw about 104 atoms', 104,
          sum(m104_qd_atoms) / len(m104_qd_atoms), places=0)
    check('7.10 result: quickdraw gets under 3.4 percent of the pool', True,
          (sum(m104_qd_atoms) / len(m104_qd_atoms)) / 3072 < 0.034)
    check('7.10 result: no arm was voided by the sample floor', 0,
          len(m104_ev['gate']['budgets']['512']['void_arms']), places=0)
else:
    print('  NOTE  M104 evidence absent; its result checks did not run.')

# --- M107 arithmetic quoted in 7.14 and its execution-time amendments -------
# The transformer figures are recomputed from the ONNX graphs themselves, with
# the same function the runner uses, because they drive an operand and R7
# forbids an external figure being one. If the cache is unavailable the
# geometry falls back to the values the graphs were measured to hold, and the
# fallback is announced rather than silently substituted.
M107_CLASSES = 345
M107_TRAIN_PER_CLASS = 400
M107_TEST_PER_CLASS = 100
# Clipart's share of the shared subsample, quoted in 7.14 execution-time
# amendment 5. It is a property of the draw, so the SEALED RUN recomputes it
# and writes it into the evidence; when that evidence exists this constant is
# checked against it the way M104's row counts are. Until then the arithmetic
# built on it is checked here and the constant is pinned so it cannot drift.
M107_CLIPART_ROWS = 11224
m107_train = M107_CLASSES * M107_TRAIN_PER_CLASS
m107_test = M107_CLASSES * M107_TEST_PER_CLASS


def m107_geometry(name):
    """Depth, width and MLP hidden size read off the ONNX graph.

    Deliberately NOT imported from the runner. A verifier that calls the
    function it is checking verifies that the function is self-consistent, not
    that the number in the plan is right. This reads the same file the runner
    will read and derives the same three numbers independently. When M107's
    evidence exists, the runner's own MAC ledger is recomputed against this
    formula the same way M103's and M104's are, which is where the two
    implementations get compared.
    """
    import onnx

    root = pathlib.Path(
        __import__('os').environ.get('GEODE_CACHE_DIR', r'D:\geode-ml\data\cache')
    ) / 'huggingface' / 'hub'
    found = sorted(root.glob(
        f'models--onnx-community--dinov2-{name}-ONNX/snapshots/*/onnx/model.onnx'
    ))
    graph = onnx.load(str(found[0]), load_external_data=False).graph
    width = int(graph.output[0].type.tensor_type.shape.dim[2].dim_value)
    square = sum(1 for i in graph.initializer if tuple(i.dims) == (width, width))
    hidden = {tuple(i.dims)[1] for i in graph.initializer
              if len(i.dims) == 2 and i.dims[0] == width and i.dims[1] != width}
    return {'width': width, 'depth': square // 4, 'mlp_hidden': hidden.pop()}


try:
    m107_small = m107_geometry('small')
    m107_large = m107_geometry('large')
    m107_measured = True
except Exception as exc:                                   # pragma: no cover
    print(f'  (DINOv2 graphs unavailable: {exc})')
    m107_small = {'width': 384, 'depth': 12, 'mlp_hidden': 1536}
    m107_large = {'width': 1024, 'depth': 24, 'mlp_hidden': 4096}
    m107_measured = False


def m107_dense_macs(geometry, resolution):
    """Analytic ViT multiply-accumulates for one image, in millions."""
    patches = (resolution // 14) ** 2
    tokens = patches + 1
    width, depth = geometry['width'], geometry['depth']
    total = patches * 14 * 14 * 3 * width + 2 * width * M107_CLASSES
    total += depth * (4 * tokens * width * width
                      + 2 * tokens * tokens * width
                      + 2 * tokens * width * geometry['mlp_hidden'])
    return total / 1e6


def m107_sparse_macs(atoms):
    """The sparse ledger of section 2.9.3, in millions: 27x27 patches of a
    108-dimensional whitened patch, 2x2 pooling, 345 outputs."""
    return (729 * 108 * 108 + 729 * atoms * 108
            + 4 * atoms * M107_CLASSES) / 1e6


M107_BUDGETS = [128, 256, 512, 1024, 2048, 3072]
M107_REGISTERED_SWEEP = [140, 98, 70, 42]
M107_AMENDED_SWEEP = [140, 98, 70, 56, 42, 28]


def m107_comparable(sweep):
    """Sparse budgets that have a dense reference at or below them.

    This is the gate's own rule -- a sparse point at M MACs is compared against
    the best dense point at or below M -- applied to the two sweeps, so the
    amendment's justification is recomputed rather than asserted. The full
    dense ladder includes 224 for every model size, so the upper end of the
    overlap is always the sparse ceiling.
    """
    dense = sorted(m107_dense_macs(m107_small, r) for r in sweep)
    dense.append(m107_dense_macs(m107_small, 224))
    out = {}
    for budget in M107_BUDGETS:
        macs = m107_sparse_macs(budget)
        below = [d for d in dense if d <= macs]
        if below:
            out[budget] = max(below)
    return out


check('7.14 the shared subsample is 138,000 train rows', 138000, m107_train,
      places=0)
check('7.14 the shared subsample is 34,500 test rows', 34500, m107_test,
      places=0)
check('7.14 the sample floor permits a 3,450-atom sparse generalist', 3450,
      m107_train // (4 * 10), places=0)
check('7.14 the sparse ladder tops out below that floor', True,
      3072 < m107_train // (4 * 10))
check('7.14 DINOv2-large gives 67.4 rows per fitted dimension', 67.4,
      m107_train / (2 * m107_large['width']), places=1)
check('7.14 the sparse generalist at 3,072 atoms gives 11.2', 11.2,
      m107_train / (4 * 3072), places=1)
check('7.14 amendment 5: clipart clears the floor at 256 atoms', 10.96,
      M107_CLIPART_ROWS / (4 * 256), places=2)
check('7.14 amendment 5: clipart fails the floor at 512 atoms', 5.48,
      M107_CLIPART_ROWS / (4 * 512), places=2)
check('7.14 amendment 5: the generalist at 3,072 still clears it', 11.23,
      m107_train / (4 * 3072), places=2)
check('7.14 amendment 5: 256 is the largest legal mixture budget in the ladder',
      256, max(b for b in M107_BUDGETS
               if M107_CLIPART_ROWS / (4 * b) >= 10), places=0)
check('7.14 amendment 5: four of the six budgets would have been voided', 4,
      sum(1 for b in M107_BUDGETS if M107_CLIPART_ROWS / (4 * b) < 10),
      places=0)
check('7.14 amendment 6: the smoke config declares itself inadmissible', True,
      '_smoke_note' in json.load(open(
          ROOT / 'experiments/configs/v15/m107_smoke.json', encoding='utf-8')))
check('7.14 amendment 6: the sealed config does not', False,
      '_smoke_note' in json.load(open(
          ROOT / 'experiments/configs/v15/m107_dense.json', encoding='utf-8')))
check('7.14 amendment 6: the runner refuses the sealed path', True,
      'REFUSING TO RUN' in (
          ROOT / 'experiments/tier4/eval_v15_m107_dense.py'
      ).read_text(encoding='utf-8'))
check('7.14 amendment 5: the floor caps a mixture expert at 280 atoms', 280,
      max(a for a in range(1, 4000)
          if M107_CLIPART_ROWS / (4 * a) >= 10), places=0)
check('7.14 amendment 5: the config runs the mixture at exactly those budgets',
      True,
      json.load(open(ROOT / 'experiments/configs/v15/m107_dense.json',
                     encoding='utf-8'))['sparse']['mixture_budgets']
      == [b for b in M107_BUDGETS if M107_CLIPART_ROWS / (4 * b) >= 10])
check('7.14 amendment 1: the sparse ladder tops out at 255 million MACs', 255,
      round(m107_sparse_macs(3072)), places=0)
check('7.14 amendment 1: the registered sweep bottoms out at 216 million', 216,
      round(m107_dense_macs(m107_small, 42)), places=0)
check('7.14 amendment 1: resolution 28 lands at 108 million MACs', 108,
      round(m107_dense_macs(m107_small, 28)), places=0)
check('7.14 amendment 1: resolution 56 lands at 368 million MACs', 368,
      round(m107_dense_macs(m107_small, 56)), places=0)
check('7.14 amendment 1: the registered sweep left ONE comparable budget', 1,
      len(m107_comparable(M107_REGISTERED_SWEEP)), places=0)
check('7.14 amendment 1: the amended sweep leaves TWO', 2,
      len(m107_comparable(M107_AMENDED_SWEEP)), places=0)
check('7.14 amendment 1: no already-comparable budget changed its reference',
      True,
      all(m107_comparable(M107_AMENDED_SWEEP)[b]
          == m107_comparable(M107_REGISTERED_SWEEP)[b]
          for b in m107_comparable(M107_REGISTERED_SWEEP)))
check('7.14 amendment 1: 56 sits above the sparse ceiling', True,
      m107_dense_macs(m107_small, 56) > m107_sparse_macs(3072))
check('7.14 the DINOv2 geometry is read off the graph, not quoted', True,
      m107_measured)
check('7.14 DINOv2-small and -large depths are 12 and 24', True,
      (m107_small['depth'], m107_large['depth']) == (12, 24))
check('7.14 both DINOv2 MLP ratios are 4', True,
      m107_small['mlp_hidden'] == 4 * m107_small['width']
      and m107_large['mlp_hidden'] == 4 * m107_large['width'])

# --- M107 sealed-run result quoted in 7.14 and the ledger ------------------
# Same discipline as the M104 result block: every accuracy, every margin and
# every kill-switch verdict the result block states is recomputed here FROM THE
# PER-ARM RECORDS at the one chosen penalty, never read out of the runner's own
# gate. The runner's gate is then compared against this reconstruction, which is
# the only check that can catch a gate agreeing with itself while disagreeing
# with the measurement.
M107_EVIDENCE = ROOT / 'logs/results/v15/m107_dense/evidence.json'
if M107_EVIDENCE.exists():
    m107_ev = json.loads(M107_EVIDENCE.read_text(encoding='utf-8'))
    m107_arms = m107_ev['arms']
    m107_penalty = str(m107_ev['head']['chosen_penalty'])

    def m107_acc(name):
        return m107_arms[name]['accuracy_by_penalty'][m107_penalty]

    def m107_macs(name):
        return m107_arms[name]['macs']['total']

    m107_dense_curve = sorted(
        (m107_macs(n), m107_acc(n), n) for n in m107_arms
        if m107_arms[n]['family'] == 'dense' and not m107_arms[n].get('void'))
    m107_gen_curve = sorted(
        (m107_macs(n), m107_acc(n), n) for n in m107_arms
        if n.startswith('s_generalist') and not m107_arms[n].get('void'))
    m107_mix_curve = sorted(
        (m107_macs(n), m107_acc(n), n) for n in m107_arms
        if n.startswith('s_mixture') and not m107_arms[n].get('void'))

    def m107_opponent(macs):
        """7.14 gate item 4: the best dense point at or below this budget."""
        below = [d for d in m107_dense_curve if d[0] <= macs]
        return max(below, key=lambda d: d[1]) if below else None

    m107_decided = [(g, m107_opponent(g[0])) for g in m107_gen_curve
                    if m107_opponent(g[0]) is not None]
    m107_void = [g for g in m107_gen_curve if m107_opponent(g[0]) is None]
    m107_wins = [(g, o) for g, o in m107_decided if g[1] > o[1]]

    check('7.14 result: the evidence is stamped admissible', True,
          m107_ev['admissible_as_evidence'] is True)
    check('7.14 result: the evidence names the sealed config', 'm107_dense.json',
          m107_ev['config_file'])
    check('7.14 result: the corpus is the sealed 138,000 train rows', 138000,
          m107_ev['corpus']['train_rows'], places=0)
    check('7.14 result: the corpus is the sealed 34,500 test rows', 34500,
          m107_ev['corpus']['test_rows'], places=0)
    check('7.14 result: the corpus digest is the registered one', '63f590097008f749',
          m107_ev['corpus']['subsample_sha256'][:16])
    check('7.14 result: the subsample matches the registered arithmetic', True,
          m107_ev['corpus']['train_rows'] == m107_train
          and m107_ev['corpus']['test_rows'] == m107_test)
    check('7.14 result: eighteen arms ran', 18, len(m107_arms), places=0)
    check('7.14 result: no arm was voided', 0,
          sum(1 for a in m107_arms.values() if a.get('void')), places=0)
    check('7.14 result: the penalty was chosen on the sparse side',
          's_generalist_128', m107_ev['head']['chosen_on'])
    check('7.14 result: the chosen penalty is 1.0', 1.0,
          float(m107_penalty), places=6)
    check('7.14 result: one penalty is applied to every arm', True,
          all(m107_penalty in a['accuracy_by_penalty'] for a in m107_arms.values()))

    # Every accuracy the plan and ledger quote, recomputed from correct counts
    # rather than read from the accuracy field the runner wrote.
    for name, quoted in [('d4a_small_28', 15.99), ('d4b_small_42', 19.72),
                         ('d4c_small_56', 24.50), ('d4d_small_70', 31.18),
                         ('d4e_small_98', 44.76), ('d4f_small_140', 49.74),
                         ('d5_small_224_from_32', 38.86), ('d1_small_224', 53.75),
                         ('d2_base_224', 61.30), ('d3_large_224', 65.06),
                         ('s_generalist_128', 11.17), ('s_generalist_256', 14.00),
                         ('s_generalist_512', 16.42), ('s_generalist_1024', 18.64),
                         ('s_generalist_2048', 20.61), ('s_generalist_3072', 21.52),
                         ('s_mixture_128', 16.59), ('s_mixture_256', 18.71)]:
        check(f'7.14 result: {name} scores {quoted:.2f} percent', quoted,
              100 * m107_arms[name]['correct_by_penalty'][m107_penalty]
              / m107_arms[name]['test_rows'], places=2)

    for name, quoted in [('d4a_small_28', 107566848), ('d4b_small_42', 215555328),
                         ('d4c_small_56', 367513344), ('d4d_small_70', 564215040),
                         ('d4e_small_98', 1096051968), ('d4f_small_140', 2261456640),
                         ('d1_small_224', 6123826944), ('d2_base_224', 23161757184),
                         ('d3_large_224', 81012688896)]:
        check(f'7.14 result: {name} costs {quoted:,} MACs', quoted,
              m107_macs(name), places=0)
    check('7.14 result: the information-matched arm costs what the pixel one does',
          m107_macs('d1_small_224'), m107_macs('d5_small_224_from_32'), places=0)
    for atoms, quoted in [(128, 18757392), (256, 29011728), (512, 49520400),
                          (1024, 90537744), (2048, 172572432), (3072, 254607120)]:
        check(f'7.14 result: the sparse generalist at {atoms} costs {quoted:,} MACs',
              quoted, m107_macs(f's_generalist_{atoms}'), places=0)

    # The runner's MAC ledger against this file's own analytic formula. R7: an
    # external figure is an anchor, never an operand, so the geometry is read
    # off the graph above and the arithmetic redone here.
    check('7.14 result: the runner MAC ledger matches the analytic formula', True,
          all(abs(m107_dense_macs(m107_small, r) * 1e6 - m107_macs(n)) < 1.0
              for n, r in [('d4a_small_28', 28), ('d4b_small_42', 42),
                           ('d4c_small_56', 56), ('d4d_small_70', 70),
                           ('d4e_small_98', 98), ('d4f_small_140', 140),
                           ('d1_small_224', 224)]))
    check('7.14 result: the runner sparse MAC ledger matches too', True,
          all(abs(m107_sparse_macs(a) * 1e6 - m107_macs(f's_generalist_{a}')) < 1.0
              for a in M107_BUDGETS))
    check('7.14 result: an oracle mixture costs what one expert costs', True,
          all(m107_macs(f's_mixture_{a}') == m107_macs(f's_generalist_{a}')
              for a in (128, 256)))

    # The gate itself, rebuilt from the arms.
    check('7.14 result: four of six sparse budgets are void, not won', 4,
          len(m107_void), places=0)
    check('7.14 result: two sparse budgets are decidable', 2,
          len(m107_decided), places=0)
    check('7.14 result: the void budgets are the four cheapest', True,
          [g[2] for g in m107_void] == ['s_generalist_128', 's_generalist_256',
                                        's_generalist_512', 's_generalist_1024'])
    check('7.14 result: kill switch 1 did NOT fire', False,
          bool(m107_decided) and not m107_wins)
    check('7.14 result: kill switch 2 FIRED', True, bool(m107_wins))
    check('7.14 result: kill switch 2 fired at both decidable budgets', 2,
          len(m107_wins), places=0)
    check('7.14 result: the 2048 crossing is against d4a_small_28',
          'd4a_small_28',
          next(o[2] for g, o in m107_wins if g[2] == 's_generalist_2048'))
    check('7.14 result: the 3072 crossing is against d4b_small_42',
          'd4b_small_42',
          next(o[2] for g, o in m107_wins if g[2] == 's_generalist_3072'))
    check('7.14 result: the 2048 crossing margin is +4.62 pp', 4.62,
          100 * (m107_acc('s_generalist_2048') - m107_acc('d4a_small_28')),
          places=2)
    check('7.14 result: the 3072 crossing margin is +1.80 pp', 1.80,
          100 * (m107_acc('s_generalist_3072') - m107_acc('d4b_small_42')),
          places=2)
    check('7.14 result: the crossings sit between 20.61 and 21.52 percent', True,
          abs(100 * min(g[1] for g, _ in m107_wins) - 20.61) < 0.005
          and abs(100 * max(g[1] for g, _ in m107_wins) - 21.52) < 0.005)

    # Bound 3: the window closes, and by how much.
    m107_ceiling = m107_gen_curve[-1]
    m107_passer = next(d for d in m107_dense_curve if d[1] > m107_ceiling[1])
    check('7.14 result: the sparse ceiling is the 3,072-atom generalist',
          's_generalist_3072', m107_ceiling[2])
    check('7.14 result: d4c_small_56 is the first dense arm above that ceiling',
          'd4c_small_56', m107_passer[2])
    check('7.14 result: it passes the ceiling for 1.44x the MACs', 1.44,
          m107_passer[0] / m107_ceiling[0], places=2)
    check('7.14 result: no sparse arm reaches the passing dense accuracy', True,
          all(g[1] < m107_passer[1] for g in m107_gen_curve + m107_mix_curve))

    # Kill switch 3, and the M104 reconciliation the gate demands.
    m107_ks3 = [(mix[0], next(g for g in m107_gen_curve if g[0] == mix[0]), mix)
                for mix in m107_mix_curve]
    check('7.14 result: kill switch 3 is decidable at two budgets, not six', 2,
          len(m107_ks3), places=0)
    check('7.14 result: kill switch 3 did NOT fire', False,
          bool(m107_ks3) and all(gen[1] >= mix[1] - 0.005
                                 for _, gen, mix in m107_ks3))
    check('7.14 result: the oracle mixture leads by 5.42 pp at 128 atoms', 5.42,
          100 * (m107_acc('s_mixture_128') - m107_acc('s_generalist_128')),
          places=2)
    check('7.14 result: the oracle mixture leads by 4.71 pp at 256 atoms', 4.71,
          100 * (m107_acc('s_mixture_256') - m107_acc('s_generalist_256')),
          places=2)
    check('7.14 result: the mixture ladder stops where amendment 5 says it does',
          True,
          sorted(int(n.rsplit('_', 1)[1]) for _, _, n in m107_mix_curve)
          == [b for b in M107_BUDGETS if M107_CLIPART_ROWS / (4 * b) >= 10])

    # Restriction 7: the information-matched control, and its measured cost.
    check('7.14 result: the resolution asymmetry is worth 14.89 pp', 14.89,
          100 * (m107_acc('d1_small_224') - m107_acc('d5_small_224_from_32')),
          places=2)
    check('7.14 result: the information-matched arm still beats the sparse ceiling',
          True, m107_acc('d5_small_224_from_32') > m107_ceiling[1])
    check('7.14 result: it spends 24.1x the sparse ceiling to do so', 24.1,
          m107_macs('d5_small_224_from_32') / m107_ceiling[0], places=1)
    check('7.14 result: no information-matched dense arm sits inside the window',
          True, m107_macs('d5_small_224_from_32') > m107_ceiling[0])

    # The sample-floor finding: the sparse ladder stopped on the corpus, not the
    # method. Recomputed from the row counts, not read from the runner's field.
    for atoms, quoted in [(128, 269.53), (256, 134.77), (512, 67.38),
                          (1024, 33.69), (2048, 16.85), (3072, 11.23)]:
        check(f'7.14 result: the generalist at {atoms} atoms has'
              f' {quoted} rows per fitted dimension', quoted,
              m107_ev['corpus']['train_rows'] / (4 * atoms), places=2)
    check('7.14 result: the runner agrees on rows per fitted dimension', True,
          all(abs(m107_arms[f's_generalist_{a}']['rows_per_fitted_dimension']
                  - m107_ev['corpus']['train_rows'] / (4 * a)) < 0.01
              for a in M107_BUDGETS))
    check('7.14 result: the ladder stopped one rung short of the floor', True,
          10 <= m107_ev['corpus']['train_rows'] / (4 * 3072) < 12)
    check('7.14 result: it was still improving when it stopped', 0.91,
          100 * (m107_acc('s_generalist_3072') - m107_acc('s_generalist_2048')),
          places=2)
    check('7.14 result: every step of the sparse ladder was an improvement', True,
          all(b[1] > a[1] for a, b in zip(m107_gen_curve, m107_gen_curve[1:])))
    check('7.14 result: the dense arms sit between 67.38 and 179.69 rows/dim',
          True,
          abs(min(m107_arms[n]['rows_per_fitted_dimension']
                  for _, _, n in m107_dense_curve) - 67.38) < 0.01
          and abs(max(m107_arms[n]['rows_per_fitted_dimension']
                      for _, _, n in m107_dense_curve) - 179.69) < 0.01)
    check('7.14 result: no dense arm came near the floor', True,
          min(m107_arms[n]['rows_per_fitted_dimension']
              for _, _, n in m107_dense_curve) > 6 * 10)

    # Bound 5: the gate lets the sparse arm outspend its opponent, because the
    # dense ladder steps by roughly 2x. The interpolated dense curve is
    # ARITHMETIC, not a measured arm, and is recomputed here for exactly that
    # reason -- a figure this document quotes must be checkable even when it
    # describes something that was never run.
    m107_ratio = {
        's_generalist_2048': m107_macs('s_generalist_2048') / m107_macs('d4a_small_28'),
        's_generalist_3072': m107_macs('s_generalist_3072') / m107_macs('d4b_small_42'),
    }
    check('7.14 result: the 2048 crossing outspends its opponent by 1.60x', 1.60,
          m107_ratio['s_generalist_2048'], places=2)
    check('7.14 result: the 3072 crossing outspends its opponent by 1.18x', 1.18,
          m107_ratio['s_generalist_3072'], places=2)
    check('7.14 result: no dense arm exists inside the lower crossing gap', 0,
          sum(1 for d in m107_dense_curve
              if m107_macs('d4a_small_28') < d[0] < m107_macs('d4b_small_42')),
          places=0)

    def m107_interp(macs, log_axis):
        """Dense accuracy at an unmeasured budget. Deliberately kept out of the
        gate: interpolation is not a measurement, and the plan says so."""
        ladder = [d for d in m107_dense_curve if d[2] != 'd5_small_224_from_32']
        lo = max((d for d in ladder if d[0] <= macs), key=lambda d: d[0])
        higher = [d for d in ladder if d[0] >= macs]
        if not higher:
            return lo[1]
        hi = min(higher, key=lambda d: d[0])
        if hi[0] == lo[0]:
            return lo[1]
        if log_axis:
            span = math.log(hi[0]) - math.log(lo[0])
            position = (math.log(macs) - math.log(lo[0])) / span
        else:
            position = (macs - lo[0]) / (hi[0] - lo[0])
        return lo[1] + position * (hi[1] - lo[1])

    for name, log_axis, quoted in [('s_generalist_2048', False, 2.4),
                                   ('s_generalist_2048', True, 2.1),
                                   ('s_generalist_3072', False, 0.6),
                                   ('s_generalist_3072', True, 0.3)]:
        axis = 'log-MACs' if log_axis else 'linear'
        check(f'7.14 result: the {name} margin is about {quoted} pp under'
              f' {axis} interpolation', quoted,
              100 * (m107_acc(name) - m107_interp(m107_macs(name), log_axis)),
              places=1)
    check('7.14 result: the crossing survives both interpolations at both budgets',
          True,
          all(m107_acc(n) > m107_interp(m107_macs(n), log_axis)
              for n in ('s_generalist_2048', 's_generalist_3072')
              for log_axis in (False, True)))
    check('7.14 result: interpolation shrinks every margin rather than growing it',
          True,
          all(m107_acc(n) - m107_interp(m107_macs(n), log_axis)
              < m107_acc(n) - opponent[1]
              for n, opponent in [('s_generalist_2048',
                                   m107_opponent(m107_macs('s_generalist_2048'))),
                                  ('s_generalist_3072',
                                   m107_opponent(m107_macs('s_generalist_3072')))]
              for log_axis in (False, True)))

    # Finally: this reconstruction against the runner's own gate.
    m107_runner_gate = m107_ev['gate']
    check('7.14 result: the runner agrees kill switch 1 did not fire', False,
          m107_runner_gate['kill_switch_1_dense_dominates_everywhere']['fired'])
    check('7.14 result: the runner agrees kill switch 2 fired', True,
          m107_runner_gate['kill_switch_2_sparse_wins_somewhere']['fired'])
    check('7.14 result: the runner agrees kill switch 3 did not fire', False,
          m107_runner_gate['kill_switch_3_generalist_beats_mixture']['fired'])
    check('7.14 result: the runner agrees on the two crossing budgets', True,
          sorted(round(p[0]) for p in
                 m107_runner_gate['kill_switch_2_sparse_wins_somewhere']['points'])
          == sorted(round(g[0]) for g, _ in m107_wins))
    check('7.14 result: the runner agrees the comparison is admissible', True,
          m107_runner_gate['comparison_admissible'] is True)
    check('7.14 result: the runner voided no arm', 0,
          len(m107_runner_gate['voided_arms']), places=0)
    check('7.14 result: all three switches are decidable', True,
          all(m107_runner_gate[k]['decidable'] for k in m107_runner_gate
              if k.startswith('kill_switch_')))
else:                                                      # pragma: no cover
    print('  (M107 evidence not present -- its result checks are skipped)')

for label, quoted in [
    ('7.14 quotes the 138,000 train rows', '**400 train rows per class**'),
    ('7.14 quotes the 3,450-atom floor headroom',
     '**3,450** atoms, which is above the top'),
    ('7.14 quotes the 67.4 rows per fitted dimension', '**67.4**'),
    ('7.14 quotes the sparse 11.2 rows per fitted dimension', '**11.2**'),
    ('7.14 amendment 1 quotes the 255-million sparse ceiling',
     'the sparse ladder tops out at **255 million** MACs while the registered'),
    ('7.14 amendment 1 quotes the 216-million registered floor',
     "sweep's cheapest point sits at **216 million**, so exactly **one** sparse"),
    ('7.14 amendment 1 quotes the 108 and 368 million additions',
     'budget would have had a dense reference at or below it. A comparison with one'),
    ('7.10 amendment 6 forbids quoting the seconds field',
     '   quoted as one.** **[registered during execution, on discovering the'),
    ('7.10 amendment 6 names the seconds field as not an operand',
     "6. **The `seconds` field in M104's evidence is NOT an operand and may never be"),
    ('7.10 amendment 6 quotes the measured core occupancy',
     '   the opposite was nearly done.** **[registered during execution.]** M104 was'),
    ('7.10 amendment 7 refuses to edit a sealed numerics block',
     '   block for scheduling convenience is not a trade this program makes**, and the'),
    ('7.10 amendment 7 names the thread counts that forced the decision',
     '   at **16**, so running them together would place over thirty spinning threads'),
    ('7.10 amendment 6 states what would have forced a rerun',
     '   wall-clock, the correct response would have been to stop and rerun it alone'),
    ('7.14 amendment 3 quotes the 64 verified rows',
     'runner decodes the first **64** selected rows of each split straight'),
    ('7.10 result: kill switch 1 is the headline, not a footnote',
     'spread. Under §11.1 this is the headline of M104 and is not reportable as a'),
    ('7.10 result: rank-sizing is refuted',
     'switches fired. Rank-sizing is refuted.** **[written after the sealed run;'),
    ('7.10 result: the mechanism inverted rather than merely failing',
     '**The registered mechanism is not merely unsupported — it is inverted.** The'),
    ('7.10 result: rank measures inputs, capacity follows labels',
     'is a property of the input distribution; the capacity a domain needs is a'),
    ('7.10 result: M105 and M106 are closed, not pending',
     'not. **M105 and M106 do not proceed.** Pursuing a router for a partition that'),
    ('7.10 result: sparse representation survives the refutation',
     '**What survives.** Nothing in this refutes sparse dictionaries as a'),
    ('7.10 result: the quickdraw starvation is quantified',
     'rule handed it **~104 atoms**, about'),
    ('7.14 amendment 6 records the sealed-directory contamination',
     '   `logs/results/v15/m107_dense/` — the sealed directory. Nothing had read it'),
    ('7.14 amendment 6 makes the runner refuse to start',
     '   **refuses to start** when a config declaring itself inadmissible is pointed'),
    ('7.14 amendment 6 stamps the evidence with its admissibility',
     '   `admissible_as_evidence` and the `config_file` that produced it, so a reader'),
    ('7.14 amendment 5 quotes the clipart row count',
     'showed `clipart` holds **11,224** of the 138,000 train rows'),
    ('7.14 amendment 5 quotes the 10.96 and 5.48 floor readings',
     'ten rows per fitted dimension at 256 atoms (**10.96**) and fails at 512'),
    ('7.14 amendment 5 narrows kill switch 3 to two budgets',
     'at two budgets instead of six**, which narrows M107 and is registered here'),
    ('7.14 amendment 5 rejects capping and redistributing',
     '   its own floor and redistributing, as §7.10 does, was considered and'),
    ('7.14 result: kill switch 2 is the headline, not a footnote',
     '**+1.80 pp**. Under §11.1 this'),
    ('7.14 result: the registered prediction failed',
     'prediction is refuted.** **[written after the sealed run;'),
    ('7.14 result: the prediction was registered against the thesis',
     'precisely so that it could not be quietly softened afterwards. **It failed.**'),
    ('7.14 result: four budgets are void rather than won',
     '1. **Two of six budgets are decidable; the other four are void, not won.** The'),
    ('7.14 result: the crossing accuracies are not deployable',
     '2. **The crossings are at accuracies nobody would deploy.** **20.61%** and'),
    ('7.14 result: the window closes at 1.44x the sparse ceiling',
     '3. **Dense passes the sparse ceiling for 1.44× the cost.** The sparse ladder tops'),
    ('7.14 result: the dense ladder is resolution-starved at its bottom',
     '   its bottom end.** `d4a_small_28` and `d4b_small_42` feed DINOv2 28×28 and'),
    ('7.14 result: kill switch 3 reconciles with M104 rather than contradicting it',
     '**Kill switch 3 did not fire, and it reconciles with M104 rather than'),
    ('7.14 result: partitioning is per-MAC and not per-parameter',
     'Partitioning buys nothing per *parameter* — M104 — and a great deal per'),
    ('7.14 result: the oracle subsidy is named as the reason',
     'the oracle. Both figures are oracle figures and neither survives without a router'),
    ('7.14 result: the resolution asymmetry is finally measured',
     'therefore worth **14.89 pp** to the dense side on this corpus — a measured'),
    ('7.14 result: no information-matched arm sits inside the window',
     'no information-matched dense arm inside the crossing window, so kill switch 2\'s'),
    ('7.14 result: the sparse ladder was truncated by the corpus',
     'does not know where the sparse curve goes next. That is a registered question for'),
    ('7.14 result: the two families buy capacity in different currencies',
     '**The two families do not pay for capacity in the same currency**: the'),
    ('7.14 result: prohibition 27 is discharged only for this comparison',
     '**What M107 licenses, and what it does not.** Prohibition 27 is discharged for'),
    ('7.14 result: bound 5 admits the sparse arm outspends its opponent',
     '5. **The comparison rule lets the sparse arm outspend its opponent, and the'),
    ('7.14 result: bound 5 was added after the run rather than omitted',
     '   because it bounds a figure this document already quotes.]** Gate item 4'),
    ('7.14 result: bound 5 states the crossing survives interpolation',
     'interpolation at the upper one**, and any successor milestone should place a'),
    ('7.14 result: there are five bounds, not four',
     '**The five bounds that travel with that headline, none of them optional.** The'),
]:
    check(label, True, quoted in text)
for label, quoted in [
    ('ledger 7.14 result: the outcome letter is P',
     '**M107 result (plan §7.14). Outcome letter: P — the registered prediction is'),
    ('ledger 7.14 result: the refutation favours the thesis',
     'refuted, and the refutation favours this program\'s thesis.** Evidence:'),
    ('ledger 7.14 result: the gate was recomputed independently',
     'own gate by `experiments/tier4/report_v15_m107_gate.py`, which agrees with it on'),
    ('ledger 7.14 result: the crossing margins are quoted with their asymmetry',
     '(**+1.80 pp**), both under the LVD-142M asymmetry. **This claim is admissible'),
    ('ledger 7.14 result: the five bounds are called non-optional',
     'only with its five registered bounds**, all recorded in §7.14: **four of the six**'),
    ('ledger 7.14 result: bound 5 is recorded as nearly undoing the headline',
     "**M107's fifth bound was found after the run and is the one that most nearly"),
    ('ledger 7.14 result: bound 5 calls interpolation arithmetic, not evidence',
     'budget, which is not a measured arm and is recorded as arithmetic rather than'),
    ('ledger 7.14 result: bound 5 names the successor experiment',
     'correct successor experiment places a **measured** dense arm inside the window'),
    ('ledger 7.14 result: prohibition 27 is discharged narrowly',
     '§7.14 restriction 6 and remains in force everywhere else.'),
    ('ledger 7.14 result: kill switch 3 reconciles M104',
     'contradicting it.** At matched inference MACs and under **oracle** routing the'),
    ('ledger 7.14 result: the per-parameter and per-MAC readings differ',
     'budgets differ: **partitioning buys nothing per parameter and a great deal per'),
    ('ledger 7.14 result: the resolution asymmetry is quantified',
     'sparse arms see scores **38.86%**, so the asymmetry is worth **14.89 pp** to the'),
    ('ledger 7.14 result: the crossing cannot be re-decided at information parity',
     'crossing window**, so kill switch 2 cannot be re-decided at information parity;'),
    ('ledger 7.14 result: the ladder was truncated by the corpus',
     'registered question, not a claim.** Rows per fitted dimension falls from'),
    ('ledger 7.14 result: the curve above 3,072 atoms is unmeasured',
     '3,072 atoms is unmeasured**, and no sentence in this program may assume it'),
    ('ledger 7.14: prohibition 27 is marked resolved in place',
     'comparison M107 measures. **[resolved in place, §5.10: M107 has since run to'),
]:
    check(label, True, quoted in ledger)
for label, quoted in [
    ('7.10 quotes the 3072-atom uniform spend', '**3,072**'),
    ('7.10 quotes the 3455-atom rank-sized spend', '**3,455**'),
    ('7.10 quotes the 12.5 percent parameter excess', '12.5% more parameters'),
    ('7.10 quotes the 3.6x domain size ratio', 'differ in size by **3.6×**'),
    ('7.10 quotes quickdraw at 29.46 percent of rows', '**29.46%**'),
    ('7.10 quotes the clipart cap of 838', 'clipart at **838** atoms'),
    ('7.10 quotes the infograph cap of 900', 'infograph at **900**'),
    ('7.10 quotes the real and clipart row counts',
     '`real` holds 120,906 train rows, `clipart` 33,525'),
]:
    check(label, True, quoted in text)

# Reporting runs AFTER every figure check has been registered. Placing it
# earlier once let a block of checks be counted in the final tally while their
# MISMATCH lines were never printed, which is a silent failure mode in the
# instrument that is supposed to detect silent failure modes.
failed = [c for c in checks if not c[0]]
for ok, label, quoted, actual in checks:
    if not ok:
        print(f'MISMATCH  {label}: plan says {quoted}, evidence says {actual}')
print(f'\n{len(checks) - len(failed)}/{len(checks)} quoted figures verified against evidence.')


print('\nDocument checks:')
for label, pattern in [
    ('no-novelty-claim rule present', r'Claim novelty'),
    ('R5 null contract present', r'An operand without its null is not evidence'),
    ('sample floor present', r'10 fit samples per fitted tangent dimension'),
    ('L6 free-control bar present', r'L6 — dominance over the free control'),
    ('L1prime budgeted accuracy present', r"L1′ — budgeted accuracy"),
    ('A5 settled, not still open', r'A5 — dense components are admissible'),
    ('A5 old blocking language removed', r'^(?!.*flagged as a design decision)'),
    ('dictionary regeneration registered', r'not_m80_dictionary'),
    ('hierarchy do-not-pursue registered', r'Pursue hierarchical routing'),
    ('three questions registered', r'### 3\.4 How the three questions interact'),
    ('efficiency disclosed not gated', r'Efficiency is disclosed, not gated'),
    ('Q1 sole success criterion', r'Q1 remains the plan.s sole success criterion'),
    ('efficiency contract registered', r'### 5\.11 Efficiency accounting'),
    ('sequence contract registered', r'### 5\.12 Sequence-task contract'),
    ('M100 registered', r'M100 — the cost of the system'),
    ('M101 registered', r'M101 — additive construction on a sequence task'),
    ('M98 unblocked', r'A5 settled, no longer blocked'),
    ('trunk dominance prior registered', r'\*\*P6 —'),
    ('sequence prior registered', r'\*\*P7 —'),
    ('efficiency literature present', r'### 8\.7 Sparsity and efficiency'),
    ('sequence literature present', r'### 8\.8 Sequence modelling'),
    ('no-FLOP-only rule present', r'No FLOP-only efficiency claim is admissible'),
    ('language extrapolation forbidden', r'Extrapolate a forecasting result to language'),
    ('reporting order registered', r'### 11\.3 Reporting order'),
    ('confirmation replay registered', r'Confirmation replay for a positive result'),
    ('2.8 difficulty skew registered', r'### 2\.8 Where the compute actually is'),
    ('2.8 marked as scoping observation', r'Status: scoping observation, not an operand'),
    ('2.8 trunk limitation stated', r'demonstrates\s+\*\*no systems saving whatsoever\*\*'),
    ('3.2.1 redirect registered', r'#### 3\.2\.1 Where Q2 must be answered'),
    ('3.4.7 redirect containment registered',
     r'\*\*3\.4\.7 The v15 redirect does not widen the success surface'),
    ('H110 registered', r'\*\*H110 —'),
    ('H111 registered', r'\*\*H111 —'),
    ('H110 bar pre-registered above baseline', r'registered target is \*\*> 60%\*\*'),
    ('M102 registered', r'M102 — abstention as the objective'),
    ('M102 tier split registered', r'Tier B \(systems, conditional'),
    ('M102 budget-matched dense gate null present',
     r'budget-matched dense gate'),
    ('M99 regated on evidence', r'M99 opens when \*\*M100 confirms H106\*\*'),
    ('M99 old gate contradicted in place, not deleted',
     r'The original gate was \*"H102\s*\n?refuted and H104 confirmed"\*'),
    ('H111 scope limit forbidden-sentence rule',
     r'Extend H111 to representation training'),
    ('Tier A compute-language prohibition present',
     r'Report a Tier A abstention result in the language of compute saving'),
    ('2.8 figures marked inadmissible', r'Quote §2\.8.s figures as sealed evidence'),
    ('multiplicity count updated to twelve', r'registers \*\*twelve\*\* hypotheses'),
    ('8.9 abstention literature present', r'### 8\.9 Abstention, deferral and cascades'),
    ('8.9 joint-training prior art disclosed', r'\*\*D2 — learning to defer already trains the gate jointly'),
    ('8.9 H110 not-novel statement present',
     r'\*\*H110 is not a novel idea and is not registered as one\.\*\*'),
    ('8.9 H111 literature prior is negative',
     r'\*\*The literature prior is negative, and H111 is registered as expected-refuted\.\*\*'),
    ('8.9 D5 gap disclosed as search limit, not absence',
     r'this plan does not assert that the framing is\s*\n?\s*absent from the literature'),
    ('M102 temperature-scaling baseline arm registered',
     r"\(b′\) Arm \(b\) with a temperature-scaled gate"),
    ('M102 second kill switch registered', r'\*\*Second kill switch, from §8\.9 D6'),
    ('novelty prohibition for cascades present',
     r'Present joint gate training, selective prediction or cascading as new'),
    ('8.6 H111 regime gap disclosed', r'\*\*The H111 regime gap\.\*\*'),
    ('2.8.5 supersession recorded in place', r'\*\*2\.8\.5 Superseded by M102'),
    ('2.8.5 oracle defect disclosed', r'a real gate could beat it by preferring'),
    ('2.8.5 44.4% marked not reproducible',
     r'is not\s*\n?reproducible: the corrected reading at the same operating point is \*\*30\.6%\*\*'),
    ('status header records M102 execution', r'One milestone has since been\s*\n?executed: M102 Tier A'),
    ('status header records no outcome letter',
     r'\*\*Q1 remains unanswered and v15 therefore has no outcome letter\.\*\*'),
    # --- the post-M102 amendment ---
    ('status header records the amendment',
     r'\*\*Amendment, recorded after M102\.\*\*'),
    ('2.9 registered', r'## 2\.9 What was measured after M102'),
    ('2.9.1 closes lever 1', r'### 2\.9\.1 Lever 1 is closed a second way'),
    ('2.9.2 backbone curve registered',
     r'### 2\.9\.2 The accuracy-versus-trunk-compute curve'),
    ('2.9.3 patch probe registered',
     r'### 2\.9\.3 A backprop-free representation'),
    ('2.9.1 withdraws the largest-headroom description',
     r'that description is \*\*withdrawn\*\*'),
    ('3.2.1 lever 1 marked closed', r'H110 is refuted, this lever is closed'),
    ('3.2.1 concentration risk registered',
     r'\*\*Registered concentration risk, stated against interest\.'),
    ('4.3 most-promising judgement contradicted',
     r"this paragraph's judgement did not survive"),
    ('M102 ordering deviation disclosed',
     r'\*\*Registered deviation from this table\.'),
    ('M103 registered', r'### 7\.9 M103 — is a grown dictionary better'),
    ('M103 kill switch registered',
     r'\*\*Kill switch\.\*\* If arm \(c\) does not reach arm \(a\).s 1024-atom'),
    ('M103 second kill switch registered',
     r'\*\*Second kill switch\.\*\* If arm \(d\)'),
    ('M103 sample floor pre-registered', r'\*\*expected void\*\*'),
    ('M99 second regate registered', r'\*\*Second regate\.'),
    ('M99 second regate is a disjunction',
     r'M99 opens when _either_ M100 confirms H106\s*\n?_or_ M103'),
    ('M99 regate justified as not loosening',
     r'\*\*Why this is not gate-loosening\.\*\*'),
    ('M99 null tightened to best-of', r'best of \{random patches,\s*\n?k-means\}'),
    ('M99 fifth restriction registered', r'\*\*Fifth restriction, added with the second regate'),
    ('prohibition 23 on 2.9 figures present',
     r'Quote any §2\.9 figure as evidence'),
    ('prohibition 24 on corpus mixing present',
     r'Compare any CIFAR-10 figure to any DomainNet figure'),
    ('prohibition 21 escape condition closed',
     r'The escape is therefore closed, not pending'),
    # --- the second amendment ---
    ('status header records the second amendment',
     r'\*\*Second amendment, recorded after the §2\.9 audit\.\*\*'),
    ('2.9.4 registered', r'### 2\.9\.4 Atom choice does matter'),
    ('2.9.5 registered', r'### 2\.9\.5 The registered Thiry bar did not re-verify'),
    ('2.9.4 states both readings of 2.9.3',
     r'\*\*\(A\) Atom choice does not matter.*?\*\*\(B\) k-means chooses badly'),
    ('2.9.4 names the OMP selection procedure',
     r'group orthogonal matching pursuit'),
    ('2.9.4 selection uses the training split only',
     r'subsample of the \*\*training split only\*\*'),
    ('8.5 Thiry bar marked unconfirmed',
     r'\*\*0\.869\*\* \*\(unconfirmed — §2\.9\.5\)\*'),
    ('M103 prior reversal recorded in place',
     r'\*\*\[recorded after execution — the prior on this milestone has\s*\n?reversed\.\]\*\*'),
    ('M103 states nothing else in the section changes',
     r'\*\*Nothing\s*\n?else in this section changes\*\*'),
    ('M103 candidate-pool matching restriction registered',
     r'\*\*Candidate-pool matching\.'),
    ('M103 requires arms a and c to share a pool',
     r'must\s*\n?\s*draw from the same pool at the same size'),
    ('M103 training-compute disclosure registered',
     r'\*\*Training-compute disclosure\.'),
    ('10.2 gate-loosening prediction contradicted in place',
     r'the prediction in the paragraph above did not\s*\n?survive'),
    ('10.2 keeps the harder-than-H106 claim',
     r'still stands\*\*'),
    ('10.2 registers that a refutation is not expected either',
     r'cannot be described as expected either'),
    ('prohibition 23 extended to 2.9.4', r'§2\.9\.4.s\s*\n?\s*discriminative-selection gain may not be described as a finding'),
    ('prohibition 23 names Thiry as the source of 2.9.3',
     r'rather than this program.s\*\*'),
    ('3.2.1 records the first affirmative evidence on the chain',
     r'§2\.9\.4 supplies\s*\n?the first affirmative scoping evidence'),
    ('3.2.1 states the concentration is not reduced',
     r'It does not\s*\n?reduce the concentration'),
    # --- third amendment: 2.9.6 and the corrected instrument check ---
    ('status header records the third amendment',
     r'\*\*Third amendment, recorded after the M103 instrumentation run\.\*\*'),
    ('2.9.6 registered',
     r'### 2\.9\.6 The M103 instrumentation run'),
    ('2.9.6 declares itself an instrumentation run, not a milestone',
     r'It is an instrumentation run,\s*\n?not a milestone'),
    ('2.9.6 discloses single seed and single budget',
     r'single seed, single budget, unsealed, inadmissible'),
    ('2.9.6 records that 2.9.4 limitation (ii) resolves',
     r'registered limitation \(ii\) was that it might vanish by\s*\n?1024'),
    ('2.9.6 records the 2.9.3 reversal',
     r'\*\*Second \u2014 \u00a72\.9\.3\'s ordering reverses at full scale\.\*\*'),
    ('2.9.6 explains the reversal by the relaxed instrument',
     r'20,000 rows, stride 2 and a head that did not converge; relaxing all'),
    ('2.9.6 records the instrument-check defect',
     r'registered instrument check in \u00a77\.9 design item 4 cannot be\s*\n?'
     r'satisfied by any run'),
    ('2.9.6 names the 4000-feature mismatch',
     r'Coates anchor is \*\*0\.796 at 4000 features\*\*'),
    ('2.9.6 cites R7 on anchors versus operands',
     r'R7 states that external figures are anchors and\s*\n?never operands'),
    ('2.9.6 states the correction tightens rather than relaxes',
     r'correction \*\*tightens\*\* conformance to R7 rather than relaxing a bar'),
    ('2.9.6 discloses the truncated regularisation grid',
     r'the head\u2019s regularisation grid was\s*\n?truncated|'
     r"the head's regularisation grid was\s*\n?truncated"),
    ('2.9.6 discloses the grid was extended after seeing data',
     r'made \*\*after seeing data\*\* and is recorded as such'),
    ('7.9 item 4 corrected in place rather than deleted',
     r'\*\*\[corrected after execution \u2014 this check as written cannot be '
     r'satisfied by\s*\n?any run, and it violates R7\. Retained per \u00a75\.10\.\]\*\*'),
    ('7.9 corrected check registers monotonicity',
     r'\*\*\(i\) Monotonicity\.\*\* Arm \(b\)\'s accuracy must rise with atom count'),
    ('7.9 corrected check registers the internal floor',
     r'\*\*\(ii\) A floor set by this program\'s own weaker instrument\.\*\*'),
    ('7.9 corrected check registers encode determinism',
     r'\*\*\(iii\) Encode determinism\.\*\*'),
    ('7.9 corrected check demotes Coates to a non-gating anchor',
     r'as an anchor\*\*, with\s*\n?\s*the 4000-versus-1024 mismatch stated'),
    ('7.9 correction states no operand or bar is changed',
     r'change any\s*\n?\s*operand, either kill switch, the sample floor, or the '
     r'acceptance criterion'),
    ('prohibition 23 extended to 2.9.6',
     r'\u00a72\.9\.6\'s figures are \*\*single-seed\*\*'),
    ('prohibition 23 names the truncated grid as a reason',
     r'\*\*truncated regularisation grid whose top value won for\s*\n?\s*every arm\*\*'),
    # --- fifth amendment: the prior-art audit and the M104-M106 pivot ---
    ('status header records the fifth amendment',
     r'\*\*Fifth amendment, recorded after a prior-art audit of M103\.\*\*'),
    ('fifth amendment states the audit went against M103',
     r'\*\*the audit went against\s*\n?it\*\*'),
    ('fifth amendment withdraws no M103 figure',
     r'\*\*No M103 operand changes and no M103 figure is withdrawn\*\*'),
    ('2.9.7 registered',
     r'### 2\.9\.7 Effective rank as a sizing instrument'),
    ('2.9.7 declares itself unsealed and inadmissible',
     r'\*\*unsealed, single-run, inadmissible as operands\s*\n?under \u00a72\.4\*\*'),
    ('2.9.7 names RankMe as an instrument it did not build',
     r'It was\s*\n?\*not\* built for this program'),
    ('2.9.7 records probe 1 refuting the plan author\'s own hypothesis',
     r'the hypothesis was formed before the probe\s*\n?and the probe killed it'),
    ('2.9.7 records the row-matched control design',
     r'a specialist sees fewer rows, and fewer\s*\n?rows lower rank by\s*\n?themselves'),
    ('2.9.7 disowns the mean as the wrong statistic',
     r'\*\*which conceals the finding entirely\*\*'),
    ('2.9.7 checks probe 3 against the pixel-complexity artifact',
     r'the pixel-complexity\s*\n?ordering tracks the rank ordering exactly'),
    ('2.9.7 forbids quoting the spread as resolution-independent',
     r'no milestone may quote 6\.32\u00d7 as a\s*\n?resolution-independent figure'),
    ('2.9.7 states the intrinsic features cannot move under growth',
     r'its features\s*\n?are a function of the input alone, so growth cannot move '
     r'them at all'),
    ('2.9.7 separates within-image from across-image diversity',
     r'\*\*within-image\*\* diversity is high for quickdraw'),
    ('M104 registered',
     r'### 7\.10 M104 \u2014 does sizing an expert to its sub-population'),
    ('M104 registers the uniform-MoE null',
     r'\*\*\(a\) Uniform MoE\*\* \u2014 six experts, equal atom count\. \*\*The null\*\*'),
    ('M104 registers the structure-matched random-partition null',
     r'\*\*This is\s*\n?\s*the structure-matched null R5 requires\*\*'),
    ('M104 matches on inference MACs rather than parameters',
     r'\*\*Matched on inference MACs, not parameters\*\*'),
    ('M104 declares its routing an oracle upper bound',
     r'\*\*M104 therefore measures an upper\s*\n?bound and may not be reported as a '
     r'system result\*\*'),
    ('M104 registers its prediction before measurement',
     r'\*\*Registered prediction, recorded before measurement\.\*\*'),
    ('M104 registers that a uniform margin refutes its mechanism',
     r'\*\*If the margin is uniform across all six domains, the stated mechanism\s*\n?'
     r'is wrong even if the aggregate numbers favour arm \(b\)\*\*'),
    ('M104 kill switch 1 registered',
     r'\*\*Kill switch 1\.\*\* If arm \(b\) does not beat arm \(a\)'),
    ('M104 kill switch 2 registered',
     r'\*\*Kill switch 2\.\*\* If arm \(d\)'),
    ('M104 kill switch 3 registered',
     r'\*\*Kill switch 3\.\*\* If arm \(c\), the single generalist'),
    ('M104 binds the sample floor per expert',
     r'binds \*\*per\s*\n?expert, on that expert\'s own rows\*\*'),
    ('M104 keeps void distinct from negative',
     r'makes its arm \*\*void, not negative\*\*'),
    ('M105 registered',
     r'### 7\.11 M105 \u2014 does the intrinsic router survive contact'),
    ('M105 is conditional on M104 surviving',
     r'\*\*Conditional on M104 surviving all three kill switches\.\*\*'),
    ('M105 registers the random-routing null',
     r'\*\*\(d\) Random routing\*\* \u2014 \*\*the null\*\*'),
    ('M105 binds the routing tax to its headline',
     r'\*\*The routing tax is reported with the headline\.\*\*'),
    ('M105 counts the router in the compute ledger',
     r'total inference MACs \*\*including the\s*\n?router\*\*'),
    ('M105 kill switch registered',
     r'\*\*Kill switch\.\*\* If \(b\) does not beat \(d\) by more than the seed spread'),
    ('M106 registered',
     r'### 7\.12 M106 \u2014 does the construction actually compose additively'),
    ('M106 is conditional on M105 surviving',
     r'\*\*Conditional on M105 surviving its kill switch\.\*\*'),
    ('M106 forbids refitting earlier experts',
     r'\*\*without refitting the first four experts and without re-measuring\s*\n?'
     r'their allocations\*\*'),
    ('M106 names the O(K squared) failure mode',
     r'total construction cost is \*\*O\(K\u00b2\)\*\*'),
    ('M106 registers two growth orders',
     r'\*\*Order sensitivity\.\*\* Growth is run in \*\*two different domain orders\*\*'),
    ('M106 kill switch registered',
     r'\*\*the construction is not additive\*\* and the\s*\nO\(K\) claim fails'),
    ('M106 forbids continual-learning language',
     r'\*\*no M106 figure may be described as demonstrating\s*\n?continual learning'),
    ('7.13 records the missing dense comparator',
     r'\*\*Neither compared anything to a dense network\*\*'),
    ('7.13 restricts efficiency claims until a dense comparator exists',
     r'\*\*No M104, M105 or M106 document may state or imply\s*\n?an efficiency result '
     r'against dense networks\*\*'),
    ('6.1 P2 hardness citation corrected in place',
     r'\*\*\[corrected in place after the M103 prior-art audit\. Retained per '
     r'\u00a75\.10\.\]\*\* The\s*\nBlum & Rivest citation above is about \*\*training\*\*'),
    ('6.1 P2 names the PAC-learnability result that actually applies',
     r'\*\*Fang et al\., NeurIPS 2022\*\* \(\u00a78\.10\.5\), which is a '
     r'\*\*PAC-learnability\*\* result'),
    ('8.10 registered',
     r'### 8\.10 Effective rank, random features and mixtures of experts'),
    ('8.10 disclaims novelty as every other lineage section does',
     r'\*\*no novelty\s*\nclaim of any kind, and no assertion of absence of prior art'
     r'\.\*\*'),
    ('8.10.1 records the Avron ridge-leverage separation',
     r'\*\*ridge leverage\s*\n  scores\*\* requires `O\(s_\u03bb \u00b7 log s_\u03bb\)`'),
    ('8.10.1 records the matching lower bound',
     r'\*\*with a matching\s*\n  lower bound\*\*'),
    ('8.10.1 records that C103.3 was predicted by the theory',
     r'\*\*C103\.3\'s narrowing margin is \*predicted\*, not merely volunteered\.\*\*'),
    ('8.10.1 records that the published mechanism is label-free',
     r'\*\*The strongest published results are label-free\.\*\*'),
    ('8.10.1 discloses the Sinha & Duchi fetch failure',
     r'\*\*Exact figures not verified by fetch\*\*'),
    ('8.10.2 attributes RankMe to its authors',
     r'\*\*This is the instrument\s*\n  \u00a72\.9\.7 uses, unmodified\.\*\*'),
    ('8.10.3 records that constructive white-box networks already exist',
     r'\*\*This program may not build a constructive\s*\nwhite-box network'),
    ('8.10.3 records that CRATE is public and at scale',
     r'\*\*A sparse,\s*\n?\s*inspectable architecture at scale already exists and '
     r'is public\.\*\*'),
    ('8.10.4 records that published experts are uniformly sized',
     r'the experts\s*\n  remain \*\*uniformly sized\*\*'),
    ('8.10.4 states the search failure without claiming novelty',
     r'\*\*search failure\s*\ndisclosure under \u00a78\.6 and not a novelty claim\*\*'),
    ('8.10.4 states the correct reading of a failed search',
     r'\*"this program did not find it"\*, never \*"it does not exist\."\*'),
    ('8.10.5 corrects the Blum & Rivest misuse',
     r'\*\*That result is about \*training\* a\s*\n3-node network'),
    ('8.10.5 separates closed-set routing from open-set rejection',
     r'\*\*closed-set\*\* domain probe at\s*\n\*\*0\.8946\*\*'),
    ('8.10.5 keeps the corpus restriction on the 0.8946 figure',
     r'\*\*the 0\.8946 figure is a v14\s*\nDomainNet figure that may not be compared to '
     r'any CIFAR-10 figure\*\*'),
    ('prohibition 25 on presenting borrowed instruments as this program\'s',
     r'25\. \*\*Present effective-rank measurement, constructive white-box networks'),
    ('prohibition 26 binds the C103.1 prior-art disclosure',
     r'26\. \*\*State C103\.1 without its prior art'),
    ('prohibition 27 forbids a dense-network efficiency claim from M104-M106',
     r'27\. \*\*Claim an efficiency result against dense networks from M104, M105 or '
     r'M106'),
    # --- M104 execution-time amendments, registered before the sealed run ---
    ('7.10 registers its execution-time amendments before measurement',
     r'\*\*Execution-time amendments, registered before any M104 figure was '
     r'computed\.\*\*'),
    ('7.10 states the amendments were made before the run started',
     r'\*\*before the sealed run was started\*\*'),
    ('7.10 states every amendment makes the milestone harder',
     r'^Each is written so that a reader can see whether it makes the milestone'),
    ('7.10 does not claim amendment 5 makes the milestone harder',
     r'^The first four make it \*\*harder\*\*; the fifth makes the sample floor'),
    ('7.10 derives the MAC match as the row-weighted atom sum',
     r"^\s+`Σ_e f_e·A_e`, where `f_e` is domain \*e\*.s share of rows\."),
    ('7.10 names the plain atom sum as the reading it is NOT',
     r"^\s+plain atom sum, because DomainNet.s domains differ in size by \*\*3\.6×\*\*$"),
    ('7.10 reports the parameter excess rather than matching it away',
     r'\*\*reported rather than matched away\*\*'),
    ('7.10 marks design item 2 as governing over design item 1',
     r'^\s+GOVERNS\.\]\*\*$'),
    ('7.10 amends design item 1(c) in place rather than replacing it',
     r'independent of sizing\. \*\*\[amended in place$'),
    ('7.10 states the generalist conflict is a factor of six',
     r'For a \*generalist\* they differ by a \*\*factor of$'),
    ('7.10 runs both generalists rather than choosing one',
     r'\*\*Both are run\*\*: arm \(c1\) at the'),
    ('7.10 states arm (d) does not control the traffic confound',
     r'effective rank\. \*\*Arm \(d\) does not control$'),
    ('7.10 registers kill switch 4',
     r'\*\*Kill switch 4\.\*\* \*\*\[added before execution, with arm \(e\)\.\]\*\*'),
    ('7.10 states kill switch 4 in the arbitrage form',
     r'\*\*traffic-weighted MAC arbitrage\*\*'),
    ('7.10 states the per-class reading is reported for every expert',
     r'^\s+reported for every expert of every arm\*\*\. On a 345-class corpus'),
    ('7.10 states the per-class reading binds every arm equally',
     r'it is met by no arm \*equally\*'),
    ('7.10 records the sample-floor cap as binding against the nulls',
     r'\*\*The floor therefore makes\s*\nthe nulls more like the treatment, not less\*\*'),
    ('7.10 restriction 7 records the head change from M103',
     r'7\. \*\*The head is a multi-output ridge, and that is a change from M103\.\*\*'),
    ('7.10 restriction 7 states the constant is chosen on the null arm',
     r'\*\*once, on the null arm \(a\), at the first seed\*\*'),
    ('7.10 amends kill switch 3 in place for the two generalists',
     r'\*\*\[amended in place before execution: evaluated separately\s*\nagainst arm '
     r'\(c1\)'),
    ('7.10 records five execution-time amendments',
     r'exposed a fifth\. All five are recorded here, in the plan'),
    ('7.10 amendment 5 refits the reported model on every row',
     r'5\. \*\*The reported model is refitted on every row the expert owns, and the'),
    ('7.10 amendment 5 attributes the defect to the smoke run',
     r'execution, after the smoke run exposed the defect it fixes\.\]\*\*'),
    ('7.10 amendment 5 records the two voided experts',
     r"\(b\).s six experts were voided anyway\*\*\. A guard that does not guard"),
    ('7.10 amendment 5 states it costs no extra encode pass',
     r'\*\*no additional encode pass and no additional memory\*\*'),
    ('7.10 amendment 5 states the floor is now exactly enforced',
     r'is now enforced \*\*exactly\*\* by the cap rather than'),
    ('7.10 marks the 3,455 as an illustration and not a prediction',
     r'^\s+\*\*That figure illustrates the size of the parameter excess the MAC match$'),
    ('7.10 states the sealed run re-measures rank rather than importing it',
     r"^\s+never an operand\. The sealed run does \*\*not\*\* consume them: it re-measures$"),
    ('7.14 records that M107 was registered before any M104 accuracy existed',
     r'### 7\.14 M107 .* \*\*\[new — registered while M104 was running, before '
     r'any M104 accuracy existed\]\*\*'),
    ('7.14 registers a prediction against the program own thesis',
     r'dominates the sparse ladder in accuracy at every MAC budget where the two$'),
    ('7.14 marks its prediction as against interest',
     r"\*\*This prediction is against the program's$"),
    ('7.14 kill switch 1 refutes Q2 if the dense curve dominates',
     r"\*\*§3\.2 Q2's efficiency claim is refuted at this$"),
    ('7.14 forbids reporting kill switch 1 as a footnote',
     r'^It may not be reported as a footnote, a limitation, or future work\.$'),
    ('7.14 records the LVD-142M training asymmetry',
     r'\*\*LVD-142M\*\*, 142 million curated images'),
    ('7.14 records the resolution asymmetry as favouring dense',
     r"^2b\. \*\*The resolution asymmetry is real, runs in dense's favour, and is measured$"),
    ('7.14 defines arm d5 as the information-matched control',
     r'\*\*\(d1\) minus \(d5\) is what the extra pixels are worth;$'),
    ('7.14 restriction 7 binds arms d1 and d5 together',
     r'^7\. \*\*Arm \(d1\) and arm \(d5\) are reported together or not at all\*\*'),
    ('7.14 restriction 5 forbids a wall-clock comparison between families',
     r'^5\. \*\*Analytic MACs only\*\*, per design item 4; no wall-clock comparison'),
    ('7.14 chooses the head constant on the sparse side',
     r'chosen \*\*once, on the sparse generalist at$'),
    ('7.13 cross-references 7.14 as the measurement that closes it',
     r'\*\*\[The comparator is now designed, in §7\.14, and registered while M104 was still$'),
    ('7.13 keeps prohibition 27 in force until 7.14 has run',
     r'stays in force until §7\.14 has actually been run\.\]\*\*'),
    ('7.14 amendment 1 adds two resolutions before measurement',
     r'^1\. \*\*The resolution sweep gains 28 and 56\.\*\* Design item 2 registers$'),
    ('7.14 amendment 1 states the overlap is two budgets wide',
     r"^   make the sparse side's job easier at any budget that was already$"),
    ('7.14 amendment 2 makes the sample floor void an arm',
     r'^2\. \*\*The §5\.3 floor voids an arm rather than being reported beside it\.\*\*'),
    ('7.14 amendment 1 refuses a degenerate dense opponent',
     r'^   would have to sit below one patch of image, and handing the sparse side a$'),
    ('7.14 amendment 2 aborts if the selection arm is itself void',
     r'^   constant on is itself void, because every other arm inherits that constant\.$'),
    ('7.14 amendment 3 proves the pixel identity rather than asserting it',
     r'^3\. \*\*The instrument proves, rather than asserts, that the two families see the$'),
    ('7.14 amendment 3 requires bitwise equality',
     r'parquet and requires them \*\*bitwise\*\* equal to the cached tensors; a mismatch$'),
    ('7.14 amendment 4 records the single seed as not making it harder',
     r'^   this does not make the milestone harder\*\*, and it is recorded as a limitation$'),
    ('7.14 amendment 5 stops the mixture ladder at 256 atoms',
     r'^5\. \*\*The mixture ladder runs only at 128 and 256 atoms, and §5\.3 is why\.\*\*$'),
    ('7.14 amendment 5 requires kill switch 3 to disclose the narrowing',
     r'^   rather than discovered in the output — and every sentence reporting kill$'),
]:
    ok = re.search(pattern, text, re.M | re.S) if 'removed' not in label \
        else ('flagged as a design decision' not in text)
    document_checks_run.append(label)
    if not ok:
        document_failures.append(label)
    print(f'  {"OK " if ok else "MISSING"} {label}')

# Cross-reference checking. The earlier version accepted any §X.Y whose X was a
# top-level header, which let §7.9 pass while no such section existed. A §X.Y
# reference now requires an X.Y header at some level, or an X.Y bold pseudo-
# header, which is how this document marks its sub-registrations.
# References into other documents (`OTHER_DOC.md` §9.7) are removed first: their
# targets are not in this file and cannot be resolved here.
xref_text = re.sub(r'`[A-Za-z0-9_.]+\.md`\s*§\d+\.\d+', '', text)
dangling = sorted(set(re.findall(r'§(\d+\.\d+)', xref_text)))
headers = set(re.findall(r'^#{2,6} (\d+\.\d+)', text, re.M))
headers |= set(re.findall(r'^\*\*(\d+\.\d+)', text, re.M))
headers |= {h.rsplit('.', 1)[0] for h in
            re.findall(r'^#{2,6} (\d+\.\d+\.\d+)', text, re.M)}
bad = [d for d in dangling if d not in headers]
print(f'  {"OK " if not bad else "BROKEN"} cross-references '
      f'({len(dangling)} distinct; broken: {bad if bad else "none"})')

# Negative control on the cross-reference checker itself: the check above is
# only meaningful if it fires on a reference that does not resolve. Prove it
# does, on a section number this document will never contain.
probe = xref_text + '\n\nSee §99.9 for nothing.\n'
probe_bad = [d for d in sorted(set(re.findall(r'§(\d+\.\d+)', probe)))
             if d not in headers]
print(f'  {"OK " if probe_bad == ["99.9"] else "BROKEN"} cross-reference '
      f'checker fires on a dangling reference (control: {probe_bad})')

# Dangling-reference checking cannot catch a reference that points at a real but
# wrong section, which is what section renumbering actually produces. Pin every
# milestone to the section number that heads it.
milestone_section = {}
for num, mid in re.findall(r'^### (\d+\.\d+) (M\d+)', text, re.M):
    milestone_section[mid] = num
mis = []
for mid, num in milestone_section.items():
    # Only direct adjacency counts: "§7.5 M100" or "M100 (§7.5)". A looser
    # window produces false positives such as "(§6.2 A5), so M98's ..." where
    # the section reference belongs to something else on the same line.
    for ref in re.findall(rf'§(\d+\.\d+) {mid}\b', text):
        if ref != num:
            mis.append(f'{mid} referenced as §{ref} but heads §{num}')
    for ref in re.findall(rf'\b{mid} \(§(\d+\.\d+)\)', text):
        if ref != num:
            mis.append(f'{mid} referenced as §{ref} but heads §{num}')
print(f'  {"OK " if not mis else "BROKEN"} milestone section pins '
      f'({len(milestone_section)} milestones; wrong: {mis if mis else "none"})')
print(f'\nPlan length: {len(text):,} chars, {text.count(chr(10)) + 1:,} lines')

structural_failures = []

# Structural check 4: the negative controls must be WELL FORMED. A control whose
# corruption target is absent silently corrupts nothing, and one that appears
# twice corrupts the wrong copy -- either way the control passes while proving
# nothing, which is the exact failure this whole file exists to prevent. Both
# faults are settled by counting, in milliseconds, instead of by the six-minute
# control suite. Note the routing rule: run_negative_control sends a control to
# the LEDGER iff the word 'ledger' appears in its label, so a plan control whose
# label happens to contain 'compute ledger' is looked up in the wrong document.
# That has happened. Counting here is what caught it.
control_faults = []
control_missing = []
for _entry in NEGATIVE_CONTROLS:
    _label, _target = _entry[0], _entry[1]
    _doc = ledger if ('ledger' in _label and LEDGER_PATH.exists()) else text
    if '\n' in _target:
        control_faults.append(f'{_label}: target spans a line break')
        continue
    _seen = _doc.count(_target)
    if _seen > 1:
        control_faults.append(f'{_label}: target appears {_seen} times')
    elif _seen == 0:
        control_missing.append(_label)
# Exactly one target is absent while a control is in flight, because that is the
# corruption. More than one means a target has genuinely rotted.
if len(control_missing) > 1:
    control_faults.append(f'targets not found: {control_missing}')
print(f'  {"OK " if not control_faults else "BROKEN"} negative controls '
      f'well formed ({len(NEGATIVE_CONTROLS)} controls; '
      f'{control_faults if control_faults else "no faults"})')
if control_faults:
    structural_failures.append(f'malformed negative controls: {control_faults}')

# Structural check 5: no two checks may share a label, and every negative
# control must name something that can actually appear in a failure line. Two
# checks with the same label are indistinguishable in the output and in the
# control suite: a control naming that label cannot say which one it broke, and
# a passing copy hides a failing one. That has happened -- three presence checks
# quoting LEDGER text were pasted into the PLAN presence list, where they
# searched the wrong document and failed, while the correct copies passed under
# the same names. Counting labels catches it in milliseconds; the control suite
# could not, because it saw the label fire and asked no further questions.
# run_negative_control matches its expectation as a SUBSTRING of a failure line,
# so a target may name a whole label, part of one, or a structural category.
STRUCTURAL_CATEGORIES = (
    'malformed negative controls', 'duplicate check labels',
    'negative controls naming no registered check', 'broken cross-references',
    'cross-reference checker did not fire on its control',
    'milestone section pins')
_label_counts = Counter(label for _, label, _, _ in checks)
_label_counts.update(document_checks_run)
_duplicate_labels = sorted(l for l, n in _label_counts.items() if n > 1)
if _duplicate_labels:
    structural_failures.append(f'duplicate check labels: {_duplicate_labels}')

# The detector above cannot be exercised by a negative control, because a
# control corrupts a DOCUMENT and this check reads the verifier's own label
# list. So it is exercised directly, the same way the cross-reference checker
# is: a probe list carrying one deliberate duplicate and one deliberate orphan
# must produce exactly those two findings. A checker that cannot be shown to
# fire is an assertion, not a check.
_probe_labels = Counter(['a real label', 'a real label', 'a unique label'])
_probe_duplicates = sorted(l for l, n in _probe_labels.items() if n > 1)
_probe_orphans = sorted(
    t for t in {'a real label', 'a label nobody registered'}
    if not any(t in label for label in _probe_labels)
    and not any(t in category for category in STRUCTURAL_CATEGORIES))
if _probe_duplicates != ['a real label']:
    structural_failures.append(
        f'duplicate-label detector did not fire on its control: '
        f'{_probe_duplicates}')
if _probe_orphans != ['a label nobody registered']:
    structural_failures.append(
        f'orphan-control detector did not fire on its control: '
        f'{_probe_orphans}')
_control_targets_missing = sorted(
    target for target in
    {entry[3] for entry in NEGATIVE_CONTROLS if len(entry) > 3}
    if not any(target in label for label in _label_counts)
    and not any(target in category for category in STRUCTURAL_CATEGORIES))
if _control_targets_missing:
    structural_failures.append(
        f'negative controls naming no registered check: '
        f'{_control_targets_missing}')
print(f'  {"OK " if not _duplicate_labels and not _control_targets_missing else "BROKEN"}'
      f' check labels unique ({len(checks) + len(document_checks_run)} checks, '
      f'{len(_label_counts)} distinct names)')

if bad:
    structural_failures.append(f'broken cross-references: {bad}')
if probe_bad != ['99.9']:
    structural_failures.append(
        f'cross-reference checker did not fire on its control: {probe_bad}')
if mis:
    structural_failures.append(f'milestone section pins: {mis}')

if failed or document_failures or structural_failures:
    print(f'\nFAILED: {len(failed)} figure, {len(document_failures)} document, '
          f'{len(structural_failures)} structural.')
    for label in document_failures:
        print(f'  MISSING     {label}')
    for problem in structural_failures:
        print(f'  STRUCTURAL  {problem}')
    sys.exit(1)

print(f'\nPASSED: {len(checks)} figure checks, '
      f'{len(document_checks_run)} document checks, '
      f'{5} structural checks.')