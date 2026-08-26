"""D6 reuses D5's process-wide synthetic-evidence-only guard.

The D6 numerical/export implementation is allowed to consume synthetic L0/L1
fixture bytes after D5 validation.  It retains the D5 prohibition on images,
formal/outcome roots, model construction, autograd, CUDA, and subprocesses.
"""

from __future__ import annotations

import os
from pathlib import Path


if os.environ.get("VITNNS_D6_STRICT") == "1":
    guard = Path(__file__).resolve().parents[1] / "d5_guard" / "sitecustomize.py"
    source = guard.read_text(encoding="utf-8")
    source = source.replace("VITNNS_D5_STRICT", "VITNNS_D6_STRICT")
    source = source.replace("VITNNS_D5_GUARD_ACTIVE", "VITNNS_D6_GUARD_ACTIVE")
    source = source.replace("D5 strict guard", "D6 strict guard")
    code = compile(source, str(guard), "exec")
    exec(code, globals(), globals())
