# Stage X X1-R1 Zennit dependency addendum

Frozen before installation on 2026-09-08. The vendored AttnLRP official ViT
example imports Zennit `LayerMapComposite` and `Gamma`; the current environment
does not contain Zennit. This addendum authorizes exactly the PyPI wheel
`zennit==0.5.1`, URL
`https://files.pythonhosted.org/packages/1e/85/878cf390bdc993c6977bc7ae6ce31642e3383984729ec60fcff38230a4a1/zennit-0.5.1-py3-none-any.whl`,
SHA-256 `a49be81f34ccb851633fea08a07bba2d04e73b00c25ffe54ac0ec822d0a3b096`,
size 54,629 bytes, LGPL-3.0-or-later. Installation must use `pip --no-deps`;
no other package, model, checkpoint, image, or dataset may be downloaded.

This does not freeze a result-dependent AttnLRP gamma. X1 validates the official
candidate grid `conv_gamma in {0.1,0.25,100}` and
`linear_gamma in {0,0.01,0.05,0.1,1}` for forward parity and API compatibility.
X2 selects one pair by a deterministic, pre-outcome development criterion and
seals it before X3.
