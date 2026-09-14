# S1-R1-A3 report-only recovery amendment

Status: `FROZEN_AFTER_INTERMEDIATE_BEFORE_REPORT_REDUCTION`. The first CPU-only `reports intermediate` invocation failed before writing any report because the runner lacked `import numpy as np`. All 24 immutable unit terminals, arrays, A1/A2 bindings, and the stage pending receipt remain unchanged.

A3 adds only the missing NumPy import and a live implementation seal for the report producer/validator. A1 and A2 seals remain historical Git bindings. A3 may read the 0–23 sealed arrays and terminals for CPU-only reduction; it must not decode a new image or mask, load the model, call CUDA, create a new unit, or change any scientific definition, threshold, denominator rule, or authorization.
