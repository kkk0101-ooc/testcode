#!/usr/bin/env python3
"""Compute source-bound response learning images for three real ViT modules."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import validate_nns_vit_unified_paper_figures_r4 as common

REPO = common.REPO
ROOT = REPO / 'reports/nns_vit_unified_paper_figures_r9_20260917'
OLD = REPO / 'reports/nns_vit_learning_image_figures_15_21_20260917'
CHANNELS = [0, 255, 511, 767]
MODULES = ['patch_embedding', 'attention', 'residual']


def main():
    started = time.monotonic()
    if (ROOT / 'terminal_receipt.json').exists():
        raise RuntimeError('Sealed R9 root')
    data = ROOT / 'source_arrays'
    data.mkdir(parents=True, exist_ok=True)
    metadata = common.read(OLD / 'numeric_display_data.json')
    assert common.verify(metadata)
    source = metadata['source']
    inventory = common.read(OLD / 'source_arrays_manifest.json')
    assert common.verify(inventory)
    item = next(r for r in inventory['files'] if r['path'] == 'original_rgb.npy')
    photo_path = OLD / 'source_arrays/original_rgb.npy'
    assert common.sha(photo_path) == item['sha256']
    photo = np.load(photo_path, allow_pickle=False)
    np.save(data / 'original_rgb.npy', photo, allow_pickle=False)
    sys.path.insert(0, str(REPO / 'src'))
    from vitnns.modeling import load_canonical_model
    model_path = REPO / 'assets/models/vit_base_patch16_224_weights.semantic_candidate.pth'
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = load_canonical_model(
        model_path, device='cpu', unfused_attention=True,
        expected_state_dict_sha256=source['checkpoint_sha256'],
        expected_model_name='vit_base_patch16_224.augreg2_in21k_ft_in1k',
        expected_model_identity='F3C1_DIRECT_PIN_OF_RECOVERY_V2_BYTES',
        expected_canonical_tensor_digest='150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd',
    ).eval()
    model.requires_grad_(False)
    x32 = (torch.from_numpy(photo.copy()).permute(2, 0, 1).float()[None] / 255 - .5) / .5
    captured = {}
    def capture(name):
        def hook(module, inputs, output):
            captured[name] = output.detach().double().numpy().copy()
        return hook
    def capture_residual(module, inputs):
        captured['residual'] = inputs[0].detach().double().numpy().copy()
    handles = [model.patch_embed.register_forward_hook(capture('patch_embedding')),
               model.blocks[0].drop_path1.register_forward_hook(capture('attention')),
               model.blocks[0].norm2.register_forward_pre_hook(capture_residual)]
    with torch.no_grad():
        logits = model(x32)[0].numpy()
    for h in handles:
        h.remove()
    logit_error = abs(float(logits[source['class_id']]) - source['recorded_clean_logit'])
    assert int(logits.argmax()) == source['class_id'] and logit_error < 1e-4
    model.double()
    x = x32.double()
    def forward(z):
        patch = model.patch_embed(z)
        h = model.norm_pre(model.patch_drop(model._pos_embed(patch)))
        b = model.blocks[0]
        a = b.drop_path1(b.ls1(b.attn(b.norm1(h))))
        return {'patch_embedding': patch, 'attention': a[:, 1:], 'residual': (h+a)[:, 1:]}
    with torch.no_grad():
        clean = forward(x)
        baseline = forward(torch.zeros_like(x))
    parity = {}
    for name in MODULES:
        legacy = captured[name] if name == 'patch_embedding' else captured[name][:, 1:]
        raw = clean[name].numpy()
        error = float(np.max(np.abs(raw-legacy)))
        assert np.allclose(raw, legacy, atol=1e-4, rtol=1e-4), name
        parity[name] = error
        np.save(data / f'{name}_clean_tokens.npy', raw[0], allow_pickle=False)
        np.save(data / f'{name}_baseline_tokens.npy', baseline[name].numpy()[0], allow_pickle=False)
    np.save(data / 'logits.npy', logits, allow_pickle=False)
    delta = np.stack([(clean[n]-baseline[n])[0, :, CHANNELS].sum(0).numpy() for n in MODULES])
    clean_values = np.stack([clean[n][0, :, CHANNELS].sum(0).numpy() for n in MODULES])
    base_values = np.stack([baseline[n][0, :, CHANNELS].sum(0).numpy() for n in MODULES])
    print('Model and forward parity passed; computing real module response maps', flush=True)
    previous = None
    attempts = []
    final = None
    for order in [64, 128, 256, 512]:
        nodes, weights = np.polynomial.legendre.leggauss(order)
        alphas, weights = (nodes+1)/2, weights/2
        maps = torch.zeros((3, 4, 3, 224, 224), dtype=torch.float64)
        for begin in range(0, order, 8):
            alpha = torch.from_numpy(alphas[begin:begin+8])[:, None, None, None]
            weight = torch.from_numpy(weights[begin:begin+8])[:, None, None, None]
            z = (alpha*x).requires_grad_(True)
            values = forward(z)
            for m, name in enumerate(MODULES):
                for c, channel in enumerate(CHANNELS):
                    target = values[name][:, :, channel].sum()
                    grad = torch.autograd.grad(target, z, retain_graph=not (m == 2 and c == 3))[0]
                    maps[m, c] += (weight*grad*x).sum(0).detach()
            if begin % 32 == 0:
                print(f'Quadrature {order}: {min(begin+8, order)}/{order}', flush=True)
        current = maps.numpy()
        closure = np.abs(current.sum((2, 3, 4))-delta)
        residual_error = float(np.max(np.abs(current[2]-current[0]-current[1])))
        change = None if previous is None else np.sum(np.abs(current-previous), axis=(2,3,4))/np.maximum(np.sum(np.abs(current),axis=(2,3,4)), 1e-12)
        report = {'order': order, 'completeness_abs': closure.tolist(),
                  'map_L1_relative_change': None if change is None else change.tolist(),
                  'residual_additivity_max_abs': residual_error}
        attempts.append(report)
        np.save(data / f'module_response_rgb_ig_order_{order}.npy', current, allow_pickle=False)
        print(json.dumps(report), flush=True)
        if change is not None and np.all(change < .005) and np.all(closure < 1e-4 + .001*np.abs(delta)):
            final = current
            break
        previous = current
    if final is None:
        common.write(ROOT / 'module_compute_failed.json', common.signed({'attempts': attempts}))
        raise RuntimeError('Response map convergence failed')
    # Analytic patch-projection check, independent of path quadrature.
    weights = model.patch_embed.proj.weight.detach().numpy()[CHANNELS]
    tiled = np.tile(weights, (1, 1, 14, 14))
    analytic = tiled*x.numpy()[0][None]
    analytic_error = float(np.max(np.abs(final[0]-analytic)))
    assert analytic_error < 1e-9
    assert attempts[-1]['residual_additivity_max_abs'] < 1e-9
    np.save(data / 'module_response_rgb_ig.npy', final, allow_pickle=False)
    np.save(data / 'module_response_maps.npy', final.sum(2), allow_pickle=False)
    binding_paths = [photo_path, OLD/'numeric_display_data.json', OLD/'source_arrays_manifest.json',
                     model_path, Path(__file__), REPO/'src/vitnns/modeling.py',
                     REPO/'docs/unified_paper_figures_r9_design_20260917.md']
    sources = [{'path': os.path.relpath(p, REPO), 'sha256': common.sha(p), 'bytes': p.stat().st_size} for p in binding_paths]
    common.write(ROOT / 'module_compute_receipt.json', common.signed({
        'status': 'PASS_SINGLE_IMAGE_RESPONSE_MAPS', 'source_object_id': source['source_object_id'],
        'selection': 'first_frozen_Y2_cohort_sample', 'channels': CHANNELS, 'modules': MODULES,
        'target': 'sum of one fixed channel over the 196 patch-token outputs of the named module',
        'block': 0, 'reference': 'zero in normalized input; RGB gray 0.5',
        'path': 'straight line from reference to input; all Q/K/V terms active',
        'response_type': 'own response; no downstream class-logit weighting',
        'raw_shape': list(final.shape), 'map_shape': list(final.sum(2).shape),
        'clean_values': clean_values.tolist(), 'baseline_values': base_values.tolist(),
        'target_deltas': delta.tolist(), 'attempts': attempts, 'selected_order': order,
        'patch_analytic_max_abs': analytic_error, 'float32_float64_forward_max_abs': parity,
        'full_model_logit_abs_error': logit_error, 'class_id_for_forward_parity_only': source['class_id'],
        'checkpoint_sha256': source['checkpoint_sha256'], 'source_files': sources,
        'runtime': {'device':'cpu', 'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'],
                    'torch_version':torch.__version__, 'threads':4, 'dtype':'float64',
                    'sys_executable':str(Path(sys.executable).resolve()), 'wall_seconds':time.monotonic()-started},
        'scope': 'single-input method illustration; no new cohort or intervention result',
    }))
    print('PASS: module response learning images saved', flush=True)


if __name__ == '__main__':
    main()
