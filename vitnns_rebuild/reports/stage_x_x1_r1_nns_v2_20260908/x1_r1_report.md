# Stage X X1-R1 completion report

X1_R1_NNS_V2_IMPLEMENTATION_ONLY completed as PASS under the authoritative
X0-R1 identity-control amendment. No X2 development or formal candidate was
accessed.

The dependency recovery was source-recomputed as 46 retained files and 708,444
bytes at the frozen Chefer and AttnLRP commits. The recovery-v2 lock remained at
SHA-256 81beabd6aba9ede777e973e2ee5f0c60ce2aa88ce55b29d025e95d67ab175984;
Zennit 0.5.1 remained separately bound to its 54,629-byte wheel with SHA-256
a49be81f34ccb851633fea08a07bba2d04e73b00c25ffe54ac0ec822d0a3b096.

The real engineering audit used only prior-access order 44 and the sealed
recovery-v2 ViT-B/16. The predicted/true class was 713, runner-up was 742, and
the clean raw-logit margin was 3.4057936668395996. The first passing frozen
Gauss-Legendre node was 512; 128 and 256 were retained as failed convergence
attempts. NNS-v2 and the independent full-model IG identity control had the
same signed [3,224,224] digest and maximum elementwise error 0.0.

All 135 registered complete cuts passed across 38,107 logical components. The
maximum pixel reconstruction error was 1.564621925354004e-06. Residual paths
were computed through real add-node VJPs; no subtraction surrogate or
cross-layer sum was used. GPU accumulation was bounded to one deterministic
component batch while persistent reconstruction lived on CPU.

The competitive family was exactly Grad-CAM, Transformer Attribution / Chefer,
AttnLRP, RISE, and Occlusion. All adapters passed target, shape, finiteness, and
source-identity checks. All 15 AttnLRP gamma candidates had exact forward-logit
parity and none was selected. The algorithm is recorded as "AttnLRP outside
attention + CP-LRP inside attention." RISE used all 8,000 frozen masks.

The focused X1-R1 suite passed 51 tests. Formal execution used 141.53 wall
seconds, 1,718.67 CPU seconds, 0.03932 H20-GPU-hours, and 4,389,339,136 peak
reserved bytes. All X1 caps passed. The fused/unfused native-logit comparison
was recorded honestly at maximum absolute difference 1.6927719116210938e-05
with the same top-1 class; all attribution hooks used the unfused native path.

The completion receipt was published atomically, validates from source, and
authorizes only X2_NNS_V2_DEVELOPMENT_AND_FUTILITY_ONLY.
