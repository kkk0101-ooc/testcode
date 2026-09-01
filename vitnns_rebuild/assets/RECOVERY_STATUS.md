# ViT-NNS recovery status — 2026-09-01

Authoritative recovery overlay:
[`../docs/recovery_protocol_amendment_20260901.md`](../docs/recovery_protocol_amendment_20260901.md).

## Authoritative code state

- Repository: `/data01/home/yeziqi.831/nxtrec_wksp/testcode`
- Active implementation: `vitnns_rebuild` (`49_vit ` is legacy only)
- Recovery asset baseline: `stage-d-d3-pass` /
  `7697f4a86adfb8490f6b07ab5725e0998d016e65`; amendment parent:
  `33f814723e03dfca419fbeb90de65e280590ba2f`. The containing amendment
  commit is authoritative for the current recovery state.
- Actual implementation progress: Stage D D0-D6 PASS; D7 is next in the historical handoff, but cannot currently start because the sealed C10 root and exact frozen model artifact are absent.
- Worktree/index: the recovery asset baseline was clean; each later execution
  boundary must independently require and record a clean committed tree.

Reconstructed local refs whose exact commits still existed:

- `refs/archive/stage-d-d0-completion` -> `4c76d468e1368858c52ff8ef73e640463e15ac49`
- `refs/archive/stage-d-d1-completion` -> `d594eadae5b5aa715d12346f6dc93507f1e093a9`
- `refs/archive/stage-d-d2-completion` -> `c57438640e882604031775c944d8ff5ce7932713`
- `refs/archive/c10-aggregation-producer` -> `92ad81342ff22e2710df495a9961a3ab8c015a3a`

Missing Git objects/refs that must not be fabricated:

- C7 producer `314ae5e7f30b5d41f51d3b0d359992d7f0a95fdd`
- C9 producer `d8190b130d00d3b19fe7a586103a3a9cdb55e43d`

## Recovered runtime environment

Activate with:

```bash
cd /data01/home/yeziqi.831/nxtrec_wksp/testcode
source .conda/etc/profile.d/conda.sh
conda activate vitnns
cd vitnns_rebuild
```

Verified:

- Python 3.10.21
- torch 2.2.2+cu121
- torchvision 0.17.2+cu121
- timm 0.9.16
- all packages from `vitnns_rebuild/requirements.lock.txt`; `pip check` passes
- 8 x NVIDIA H20, 97,871 MiB each, driver 535.261.03
- editable `vitnns-rebuild` installation and all four CLI entry points
- Conda environment location: `testcode/.conda/envs/vitnns`

D6 focused test evidence from the recovered environment:

- ordinary D6/CLI tests: 76 passed
- strict D6 guard self-test, launched with the required process-level environment: 1 passed
- no formal model/data workload was started

## Recovered official data

Root: `/data01/home/yeziqi.831/nxtrec_wksp/testcode/vitnns_rebuild/assets`

- ImageNet validation tar split parts: `imagenet/ILSVRC2012_img_val.parts/part-00` through `part-03`
  - bytes: 6,744,924,160
  - official MD5: `29b22e2961454d5413ddabcf34fc5622` (verified)
  - reassemble with `assets/assemble_imagenet_val.sh` when a full tar is needed
- official devkit: `imagenet/ILSVRC2012_devkit_t12.tar.gz`
  - SHA-256: `b59243268c0d266621fd587d2018f69e906fb22875aca0e295b48cafaa927953` (matches frozen project value)
- official validation synset labels:
  - SHA-256: `313fc3f23c864dde9183c9a368809065765dd1b657a6904fded44e3e46d34604` (matches frozen project value)
- restored Stage C dataset: `data/ILSVRC2012_val`
  - exactly 500 JPEGs
  - 500/500 image bytes, SHA-256, byte sizes and dimensions match the tracked frozen manifest
  - regenerated manifest CSV is byte-identical to the tracked file and has SHA-256 `042cfd7845b49f8949de13c78bc40c68192179d6fc004bcdbecf55e502c2eb89`
- Stage C 20-image registry subset:
  - the 20 rows in `manifests/stage_c_sample_registry.csv` are present inside `data/ILSVRC2012_val`
  - 20/20 byte/hash/size/dimension/RGB checks passed during recovery
  - no duplicate `data/stage_c_20` copy is retained

The Stage C 500-image manifest and Stage D 500-image registry have zero overlap by sample ID and image SHA-256. Only the Stage C 500 images were extracted. Stage D reserve JPEG members were not selected, extracted, decoded, or individually hashed. The complete official tar parts necessarily contain their source bytes; a future versioned recovery/access attestation must record this archive acquisition rather than repeating the stronger historical claim that no container holding reserve bytes was obtained.

## Model recovery

Frozen formal artifact still missing:

- required state dict SHA-256: `2e718037429f18ec77aa629471beb2e99993d7383375188bab777aba04d5e4d0`
- optional legacy full pickle SHA-256: `a76a7c85db473b539c1756066376d851f3fdccf1cf494b63be81d6adf3236f05`

Recovered non-formal candidates under `models/`:

- HF revision `063c6c38a5d8510b2e57df480445e94b231dad2c` `model.safetensors`:
  - SHA-256 `32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b`
- HF `pytorch_model.bin`:
  - SHA-256 `06e516abac6943dbba7b5dddfc39af6d63de8ebe6fa730240acfd32b94e74d19`
- torch 2.2.2 reserialized semantic candidate:
  - SHA-256 `489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d`

All three contain the same 152-key set; all 152 tensors compare bitwise equal with `torch.equal`, with 86,567,656 total parameters. Canonical sorted-key/dtype/shape/raw-tensor digest for the candidate is `150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd`. This proves parameter equivalence under that definition, not frozen `.pth` file identity. None was copied to the formal default path.

## Historical experimental evidence

The Markdown and frozen D0 scope record preserve historical summary results, including C9 320/320 units and C10's primary fidelity/stability decisions. They do not replace the missing evidence roots. Missing roots include C7, C8, C9 and C10 runtime/report trees. In particular, the 301,224-row C10 table and its 524,796,892-byte sealed root cannot be regenerated from summary values.

Current Stage D static validation intentionally fails closed with `sealed C10 source verification failed`. Therefore D7 must not be started.

## Resume / rerun decision

1. If an external backup restores both the exact frozen model artifact and the exact sealed C10 root, re-run read-only validators. If all hashes pass, continue at D7; C8/C9 roots are not needed merely to continue the already-frozen Stage D contract.
2. If C9 root plus its exact producer object/ref is restored, C10 can be rebuilt/validated first; then retry Stage D validation.
3. With no external evidence backup (current state), the active amendment
   `RECOVERY-C7-RERUN-20260901` makes C7 the earliest recovery-replication
   point, not C8, C9 or D7. Next, bind its new model artifact identity and
   canonical tensor digest in the machine contracts, pass Stage A verification,
   and issue the amendment's separate C7 launch record.
4. Then run C7 -> C8 -> C9 -> C10. The tracked C8 selection documents what was selected but cannot replace C8's 2,180-unit source evidence. Because the original 15 holdout outcomes are historically known, this rerun reconstructs the audit chain but does not restore unseen confirmatory status.
5. After the new C10 is sealed, reissue/revalidate the Stage D D0-D6 binding chain against the new identities, then execute D7. Do not launch the 500-image Stage D workload before D7 issues a new execution authorization.

Historical C10 summary must remain labeled historical and not silently upgraded to fresh evidence.
