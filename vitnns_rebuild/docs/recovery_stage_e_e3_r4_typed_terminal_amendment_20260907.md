# Stage E E3-R4 prospective typed-terminal amendment

Status: authorized prospectively on 2026-09-07 after E3-R2 terminal FAIL and
before execution of the missing complement. E3-R3 had a pilot-only authority
and a running pilot was interrupted before any pilot terminal payload was
published. Its pending receipt and authority remain immutable.

E3 v1 FAIL, E3-R1 BLOCKED, E3-R2 FAIL, every prior terminal attempt, and the
E3-R3 pending/authority records are read-only inputs. The E3-R2 order-117,
coordinate `(11,2)`, NNS attempt remains `FAILED`; it is not retried, replaced,
interpolated, zero-filled, or included in numeric summaries.

E3-R4 runs only the 1,835 frozen E3 units without a legal terminal result. It
does not change samples, coordinates, methods, targets, baselines, seeds,
fidelity evaluation, descriptive statistics, or the numerical contract. New
NNS attempts retain the frozen schedule `512,1024,2048,4096,8192,16384,32768`
and conservation rule `abs_error <= 1e-4 + 0.01*abs(delta)`. Each new expected
unit must terminate atomically as `VALID`, `FAILED`, or `NA`; scientific
failure or NA does not stop the remaining complement.

Completion requires exactly 3,000 immutable new-compute terminal rows plus the
read-only primary anchor, complete 500-image coverage, explicit valid/failed/NA
counts in every coordinate/stratum cell, and numeric summaries computed only
over rows labeled `VALID`. Cost remains fully measured and disclosed, including
the interrupted E3-R3 pilot, but is not an E3-R4 completion gate.

E3-R4 may issue workflow/package `PASS` with
`scientific_validity=MIXED`; it may never issue scientific `PASS` while any
expected unit is `FAILED` or `NA`. E4 is authorized only after atomic E3-R4
terminal publication and must retain and disclose every typed failure/NA and
the valid-only summary population. E4 may not reinterpret package PASS as
all-unit scientific validity.
