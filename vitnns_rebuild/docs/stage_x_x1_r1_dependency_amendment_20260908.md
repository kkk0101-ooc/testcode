# Stage X X1-R1 dependency-materialization amendment

Date: 2026-09-08. Amendment ID:
`STAGE_X_X1_R1_DEPENDENCY_MATERIALIZATION_20260908`.

The sealed X1 v1 attempt is `BLOCKED_MISSING_PINNED_COMPARATOR_ENGINE`. This
prospective recovery authorizes only retrieval of the minimum source and
license files needed for the two already-registered comparator engines:

1. Chefer Transformer Attribution, official repository
   `https://github.com/hila-chefer/Transformer-Explainability`, commit
   `c3e578f76b954e8528afeaaee26de3f07e3fe559`, MIT license.
2. AttnLRP/LXT, official repository
   `https://github.com/rachtibat/LRP-eXplains-Transformers`, commit
   `f04344177551bfcc3665e88afe96501f43fe560b`, Clear-BSD license with the
   repository's stated scientific-use patent limitation.

Retrieval uses partial Git clones in temporary directories followed by an
allowlisted source-only copy. The retained roots are
`third_party/stage_x/transformer_explainability/` and
`third_party/stage_x/attnlrp_lxt/`. Allowed files are Python source, package
metadata, README, LICENSE/COPYING, and a generated provenance manifest. Images,
notebooks, PDFs, archives, NumPy arrays, model files, checkpoints, datasets,
results, and Git metadata are forbidden. Combined retained bytes and temporary
clone bytes each have a hard 64 MiB cap. Exact file SHA-256 and a canonical
inventory digest must be sealed before any engine import.

This amendment does not alter the NNS-v2 mathematics, identity control, five
competitive comparators, models, datasets, samples, statistics, margins, or
budgets. It permits no image access, checkpoint access, model load, Torch/timm
import, CUDA/GPU access, attribution, inference, or X2. A dependency
materialization PASS authorizes only a fresh
`X1_R1_NNS_V2_IMPLEMENTATION_ONLY`.
