#!/usr/bin/env python3
"""Emit a canonical N1 synthetic scientific payload for process determinism."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from nns_cst.core import (  # noqa: E402
    canonical_digest, estimate_subspace, fit_frozen_readout,
    projector_distance, projector_similarity, readout_score,
    semantic_complement_attribution, transport_residual,
)


def scientific_payload():
    estimate = estimate_subspace(
        [[3.0, 2.0, 0.0], [2.0, 3.0, 0.0], [4.0, 2.5, 0.0]],
        rank=1,
        control_differences=[[1.0, 0.0, 0.0]],
    )
    readout = fit_frozen_readout(
        [[0.0, 2.0, 0.0], [1.0, 3.0, 0.0]],
        [[0.0, -2.0, 0.0], [-1.0, -3.0, 0.0]],
        estimate["basis"],
    )
    rotated = [[2.0 ** -0.5, 2.0 ** -0.5, 0.0],
               [-2.0 ** -0.5, 2.0 ** -0.5, 0.0]]
    reference = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    transport = transport_residual(
        [[1.0, 0.0], [0.0, 1.0]], [[0.5, -0.25], [0.25, 0.5]], [2.0, -1.0])
    attribution = semantic_complement_attribution(
        [2.0, -3.0, 4.0], [0.5, 2.0, -1.0], reference)
    return {
        "subspace_projector": estimate["projector"],
        "eigenvalues": estimate["eigenvalues"],
        "readout": readout,
        "positive_score": readout_score(readout, [0.0, 4.0, 0.0]),
        "basis_distance": projector_distance(reference, rotated),
        "basis_similarity": projector_similarity(reference, rotated),
        "transport": transport,
        "attribution": attribution,
    }


def main():
    payload = scientific_payload()
    print(json.dumps({"digest": canonical_digest(payload), "payload": payload},
                     ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
