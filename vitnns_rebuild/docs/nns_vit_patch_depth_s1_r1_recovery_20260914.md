# NNS ViT Patch/Depth S1-R1 recovery amendment

The immutable S1 root `evidence/nns_vit_patch_depth/s1_pilot_20260914` ended `FAILED_IMPLEMENTATION`: its first single-GPU unit completed scientific computation and wrote 31 arrays, but terminal canonical JSON serialization rejected a NumPy `int64`. No unit was declared `VALID`; no confirmatory or formal probe bytes or outcomes were accessed.

S1-R1 creates the fresh root `evidence/nns_vit_patch_depth/s1_r1_pilot_20260914`. It changes only terminal serialization by recursively converting NumPy scalars and arrays to JSON-native values. Target, component, pilot schedule and records digest, P1/P2 seeds and controls, geometry, quadrature schedule, tolerances, D1/D2 definitions, escalation gates, and terminal policy are unchanged.

The failed root remains immutable. Its 31 partial arrays are not valid S1 units and are excluded from all endpoint and resource denominators. S1-R1 may independently re-hash those arrays and use their byte identities only as the single-GPU side of the deterministic comparison against the newly computed K=3 smoke unit at schedule index 0. All six smoke units are recomputed in S1-R1; no failed scientific outcome is rebound as valid.
