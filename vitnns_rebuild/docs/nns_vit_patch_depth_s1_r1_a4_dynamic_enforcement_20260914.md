# S1-R1-A4 dynamic execution enforcement

Status: `FROZEN_AFTER_INTERMEDIATE_BEFORE_FINAL_SEAL`. A2 defined the fixed-GPU global claim queue and passed synthetic tests, but the legacy static `run-level full_pilot` command remained callable and full reconciliation did not yet include dynamic-launch terminals. No full or recovery launch occurred.

A4 disables static full-pilot execution while A2 is active, makes full reconciliation/reporting/validation include immutable dynamic unit terminals, and binds the latest live execution/report/validator code. It adds no unit, data access, model load, CUDA call, metric, threshold, or scientific decision. The 24 intermediate terminals remain immutable.
