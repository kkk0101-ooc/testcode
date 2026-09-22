"""Closed, provenance-bound frozen readout serialization."""
from __future__ import annotations

import re
from typing import Any, Dict, Sequence

from nns_cst.core import fit_frozen_readout as _fit_reference


def fit_frozen_readout(
    positive: Sequence[Sequence[float]],
    negative: Sequence[Sequence[float]],
    basis: Sequence[Sequence[float]],
    fit_split: str,
    contract_sha256: str,
) -> Dict[str, Any]:
    if not fit_split:
        raise ValueError("fit_split is required")
    if re.match(r"^[0-9a-f]{64}$", contract_sha256) is None:
        raise ValueError("contract_sha256 must be 64 lowercase hex characters")
    base = _fit_reference(positive, negative, basis)
    return {
        "schema_version": 1,
        "document_type": "nns_cst_frozen_readout",
        "method": base["method"],
        "weights": base["weights"],
        "threshold": base["threshold"],
        "rank": base["rank"],
        "dimension": base["dimension"],
        "fit_split": fit_split,
        "contract_sha256": contract_sha256,
        "validation_refit_allowed": False,
        "confirmation_refit_allowed": False,
    }
