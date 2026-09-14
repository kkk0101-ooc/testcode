# S3-R4-A1 prospective recovery-script syntax amendment

The first S3-R4 recovery implementation commit contained one extra closing brace in the final recovery-launch dictionary entry. Compilation rejected the script before execution, so no S3-R4 evidence or payload root was created and no source data or arrays were accessed by the recovery.

A1 removes only that extra brace and adds no behavioral or scientific change. The corrected script must compile and the existing S3 unit tests must pass before the first S3-R4 CPU-only launch.
