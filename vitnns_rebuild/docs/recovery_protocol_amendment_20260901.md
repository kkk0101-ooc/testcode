# Recovery protocol amendment — 2026-09-01

- Amendment ID: `RECOVERY-C7-RERUN-20260901`
- Status: `APPROVED_FOR_RECOVERY_PREPARATION`
- Formal Stage C/Stage D execution authorization: `NOT_ISSUED`
- Branch and parent baseline: `stage-d-d3-pass` at
  `33f814723e03dfca419fbeb90de65e280590ba2f`
- Amendment identity: the Git commit containing this file
- New holdout or Stage D reserve result access while preparing this amendment:
  none

This document is a versioned overlay on the historical Stage C and Stage D
contracts. It changes recovery identity, evidence custody, rerun order, and the
interpretation of the reconstructed chain. It does not erase or rewrite any
historical completion record, result, failure, or decision.

## 1. Reason and missing evidence

A machine restart and loss of local, unversioned data removed the historical
runtime evidence roots. In particular, the sealed C10 root
`/data00/home/yeziqi.831/testcode/vitnns_c10_formal_92ad813_20260824`
(historical whole-root digest
`382992d94f071ce46164b92ab9eed6a60743a30fdab0bf73253bab95156986a4`)
was not recovered. Historical summaries and tracked documents do not replace
that source evidence.

The exact historical frozen state-dict binary was also not recovered:

```text
2e718037429f18ec77aa629471beb2e99993d7383375188bab777aba04d5e4d0
```

That digest remains the identity of the missing historical artifact. It must
never be assigned to, aliased to, or claimed for a recovered candidate.

## 2. Recovered and versioned assets

The following recovered assets are stored under `assets/`; the large payloads
are tracked through Git LFS. Their remote publication is verified separately
from this amendment's execution authority:

- `assets/models/model.safetensors`, SHA-256
  `32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b`;
- `assets/models/pytorch_model.bin`, SHA-256
  `06e516abac6943dbba7b5dddfc39af6d63de8ebe6fa730240acfd32b94e74d19`;
- `assets/models/vit_base_patch16_224_weights.semantic_candidate.pth`,
  SHA-256 and LFS OID
  `489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d`;
- the four LFS parts of the official ImageNet validation archive, the official
  devkit, and the 500 Stage C JPEGs under `assets/data/ILSVRC2012_val`;
- the tracked checksums, labels, and manifests described by
  `assets/RECOVERY_STATUS.md` and
  `assets/manifests/RECOVERY_HASHES.md`.

The three model files contain the same 152 tensors bit-for-bit. Their canonical
sorted-key/dtype/shape/raw-tensor digest is
`150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd`.
This proves the recorded parameter equivalence, not historical file identity.

For recovery-v2, the newly frozen canonical state-dict artifact is:

```text
path: assets/models/vit_base_patch16_224_weights.semantic_candidate.pth
file_sha256: 489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d
canonical_tensor_digest: 150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd
identity: RECOVERY_V2_NEW_ARTIFACT_NOT_HISTORICAL_2E718
```

The other two recovered model files remain independently hashed semantic
sources; they are not the canonical recovery-v2 runtime artifact. Before any
formal C7 run, a clean implementation commit must replace the historical model
path/hash bindings in every applicable config, run spec, validator, and C9/D7
code path with the recovery-v2 identity above, then pass Stage A verification.
This amendment alone does not make the current historical defaults runnable.

Only the Stage C 500-image subset has been extracted. The complete official
validation archive parts necessarily contain the source archive bytes, but no
Stage D reserve member was selected, extracted, decoded, or individually
hashed during recovery. The locally reassembled
`assets/imagenet/ILSVRC2012_img_val.tar` remains ignored and must not be
committed.

## 3. Contract continuity and interpretation

The scientific methods, sample roles, thresholds, statistics, and decision
rules are not changed by this amendment. Their tracked files remain bytewise
unchanged in this documentation-only step:

| Contract | Pre-amendment SHA-256 | SHA-256 after this amendment |
|---|---|---|
| `docs/阶段C_protocol.md` | `ff958a9455ae15307827a9e40fc6d7d03c9baa10bf1fb8b83650a10bd85ff7b8` | unchanged |
| `docs/stage_c_acceptance.md` | `f4629105d411b32d4f7700caa22d7d6b6c1bf4b8e01f37e4e2ac6ac4851dfafd` | unchanged |
| `config/stage_c.json` | `0ee57c32b0fcb3116f91fddce5d918c8eb1ae05606c3616b6a2ec9afa9f0e669` | unchanged |

The effective recovery contract is those immutable base files plus this
amendment at its containing Git commit. The later machine-binding commit must
record the new config/run-spec hashes before C7 is authorized.
`vitnns_rebuild` remains the only active implementation; `49_vit ` is historical
reference material and must not be used in the recovery execution chain.

The original 15 Stage C holdout outcomes are already historically known. A
rerun on the same registry is therefore a recovery replication and audit-chain
reconstruction; it does not restore those images to unseen confirmatory status.
New provenance must record this prior historical exposure. A new confirmatory
claim would require a separately frozen, previously unseen registry and split
before any corresponding outcome access. In a rebuilt C8,
`holdout_accessed=false` describes that C8 process only; it does not erase the
historical exposure. C10 must still apply the frozen aggregation and decision
rules for reproducibility, but the resulting decision must be labeled as a
recovery replication rather than a new confirmatory finding.

## 4. Mandatory recovery sequence

With no exact backup of both missing historical artifacts, the only authorized
order is:

1. bind the recovery-v2 model identity in code and machine-readable contracts,
   commit them, and pass Stage A verification on that clean commit; then issue
   a separate committed C7 recovery launch record that binds this amendment
   and its commit, the model file/tensor digests, Stage A evidence, the clean C7
   producer/dependency/run-spec identities, and fresh `evidence/` paths;
2. rerun C7 diagnostic and benchmark from fresh roots;
3. rerun the complete C8 calibration and machine selection from a fresh root;
4. rerun the complete frozen C9 pilot from a fresh root;
5. build and validate a new immutable C10 root only from the new sealed C9
   evidence;
6. reissue and revalidate the Stage D D0-D6 bindings against this amendment,
   the recovery-v2 model identity, the new sealed C10 root, code, dependencies,
   and the corrected recovery access attestation;
7. only then evaluate the D7 sequence `smoke -> budget -> freeze producer/ref
   -> authorize`.

The tracked `selection/stage_c_selection.json` is historical metadata only. It
does not contain C8's missing 2,180-unit source evidence and must not be used to
skip C8. No C7, C8, C9, or C10 result may be reconstructed from historical
summary values. Historical Stage C decisions remain historical and are not
silently upgraded by the recovery run.

Stage D D0-D6 implementation and synthetic-test success remain valid historical
engineering evidence, but their old C10/model bindings are not a current launch
authorization. The renewed binding chain must record, in order: D0 scope bound
to this amendment, the new model, and new sealed C10 while preserving all old
C10 statuses; D1's unchanged 500-row registry/order plus a corrected access
attestation; D2's unchanged scientific contract bound to the new identities;
D3's strict validator/config/authorization bindings; D4's clean producer
source/tree/dependency/model bindings; and D5/D6's source-recomputing validator
and analysis bindings. Historical completion records remain audit evidence and
must not be accepted as current launch inputs.

D7 is blocked until step 6 passes. A D7 authorization and its validator must
bind this amendment ID and commit, the recovery-v2 model file and canonical
tensor digests, the new sealed C10 root/producer/completion/inventory digests,
each renewed D0-D6 record and digest, the corrected access attestation, and the
frozen producer, dependencies, plan, budget, shard policy, and fresh formal
evidence root. D7 must not access reserve images, initialize D8, or issue partial
authorization; Stage D reserve access remains governed by the subsequent
authorized D8.0 boundary.

## 5. Evidence custody

Every new formal output root and detached formal report for C7 through C10 and
any later Stage D run must use a fresh repository-relative `evidence/...` path
(workspace-relative `vitnns_rebuild/evidence/...`), for example
`evidence/stage_c/recovery_20260901/...`. The entire tree is covered by Git LFS.
Formal evidence must not be placed in `/tmp`, a home-directory cache,
`outputs/`, or `artifacts/`.

Each attempt remains immutable. Failed, incomplete, or superseded attempts must
be retained and disclosed; roots must not be overwritten or spliced. After a
formal root reaches its sealed terminal state, validate it, review the LFS
inventory, commit it, push the commit and LFS objects, and verify the remote
publication before deleting or advancing past the local source. The retained
publication record must include the evidence commit SHA and LFS OID inventory;
verification must confirm both the remote ref and retrievability of the LFS
payloads by an independent clone/pull or equivalent readback.

No formal C7/C8/C9/C10 or Stage D D7/D8 execution is authorized by this
amendment commit. In particular, the sequence above is not itself launch
authorization: formal C7 remains blocked until the separate launch record in
step 1 exists and validates.
