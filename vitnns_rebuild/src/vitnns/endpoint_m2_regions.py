"""Outcome-independent region selection for supplementary Endpoint M2."""
import hashlib
import numpy as np

BLOCKS = (2, 5, 10)
HEADS = {'vit_b16': 12, 'deit_s16': 6}
REFERENCES = ('normalized_zero', 'blur_rgb')
METHODS = ('endpoint_absolute', 'endpoint_signed', 'regional_taylor',
           'gradient_only', 'displacement_only', 'random', 'center', 'attention')
COUNTS = {'patch': (1, 10, 20, 49), 'legacy_grid': (1, 2, 4)}
GRIDS = {'patch': 14, 'legacy_grid': 4}
RANDOM_REPEATS = 20

def slices(geometry):
    n = GRIDS[geometry]; width = 224 // n
    return [(slice(y*width, (y+1)*width), slice(x*width, (x+1)*width))
            for y in range(n) for x in range(n)]

def aggregate_map(value, geometry, absolute=True):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (1, 3, 224, 224) or not np.isfinite(a).all():
        raise ValueError('MAP_CONTRACT')
    if absolute: a = np.abs(a)
    return [float(a[:, :, ys, xs].sum()) for ys, xs in slices(geometry)]

def rank(scores):
    a = np.asarray(scores, dtype=float)
    if a.ndim != 1 or not np.isfinite(a).all(): raise ValueError('RANK_CONTRACT')
    return np.lexsort((np.arange(len(a)), -a)).tolist()

def region_orders(scores, geometry, key):
    n = GRIDS[geometry]
    output = []
    for method in METHODS:
        if method == 'random':
            for repeat in range(RANDOM_REPEATS):
                seed = int.from_bytes(hashlib.sha256(
                    f'ENDPOINT_M2_20260919|{key}|{geometry}|{repeat}'.encode()).digest()[:8], 'big')
                order = np.random.Generator(np.random.PCG64(seed)).permutation(n*n).tolist()
                output.append(dict(method=method, repeat=repeat, seed=seed, order=order))
        elif method == 'center':
            values = [-((i//n-(n-1)/2)**2+(i%n-(n-1)/2)**2) for i in range(n*n)]
            output.append(dict(method=method, repeat=None, seed=None, order=rank(values)))
        else:
            output.append(dict(method=method, repeat=None, seed=None, order=rank(scores[method])))
    return output

def auc(effects, counts, geometry):
    x = np.array([0, *counts], dtype=float) / GRIDS[geometry]**2
    y = np.array([0, *effects], dtype=float)
    if len(x) != len(y) or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('AUC_CONTRACT')
    return float(np.trapz(y, x) / x[-1])
