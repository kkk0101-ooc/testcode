#!/usr/bin/env python3
"""Independent-process deterministic payload including the closed R3 readout."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from n1_determinism_probe import scientific_payload  # noqa: E402
from nns_cst.core import canonical_digest, readout_score  # noqa: E402
from nns_cst_r3.readout import fit_frozen_readout  # noqa: E402


def payload():
    result = scientific_payload()
    result.pop("readout", None)
    contract_sha = hashlib.sha256((HERE / "N1数学合同.md").read_bytes()).hexdigest()
    frozen = fit_frozen_readout(
        [[0.0, 2.0, 0.0], [1.0, 3.0, 0.0]],
        [[0.0, -2.0, 0.0], [-1.0, -3.0, 0.0]],
        [[0.0, 1.0, 0.0]], "engineering_fixture", contract_sha)
    result["frozen_readout"] = frozen
    result["positive_score"] = readout_score(frozen, [0.0, 4.0, 0.0])
    result["negative_score"] = readout_score(frozen, [0.0, -4.0, 0.0])
    return result


def main():
    value = payload()
    print(json.dumps({"digest": canonical_digest(value), "payload": value},
                     ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
