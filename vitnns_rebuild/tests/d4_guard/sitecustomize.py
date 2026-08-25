"""Process-wide synthetic/mock-only safety guard for Stage D D4 tests.

The D4 runtime is allowed to persist synthetic numeric NPY/NPZ fixtures, but it
must never inspect image bytes, model weights, the candidate access log, or a
Stage D outcome/formal root.  The guard is deliberately installed by
``sitecustomize`` so it is active before pytest collection imports product
modules.
"""

from __future__ import annotations

import builtins
import importlib.abc
import io
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any


if os.environ.get("VITNNS_D4_STRICT") == "1":
    _FORBIDDEN_IMPORTS = (
        "PIL",
        "cv2",
        "timm",
        "torchvision",
        "vitnns.attribution",
        "vitnns.dataset",
        "vitnns.extractor",
        "vitnns.modeling",
        "vitnns.runner",
        "vitnns.stage_c_attention",
        "vitnns.stage_c_class_logit",
        "vitnns.stage_c_value_path",
    )
    _FORBIDDEN_SUFFIXES = (
        ".jpeg", ".jpg", ".png", ".gif", ".webp", ".bmp",
        ".pt", ".pth", ".safetensors", ".ckpt", ".bin",
    )
    _FORBIDDEN_BASENAMES = frozenset({
        "stage_d_v2_reserve.json",
        "stage_d_v2_reserve_access_log.json",
    })
    _FORBIDDEN_OUTCOME_COMPONENTS = frozenset({
        "stage_d_results",
        "stage_d_outcomes",
        "stage_d_formal",
        "predictions",
        "logits",
        "delta_fid",
    })
    _FORBIDDEN_DATA_COMPONENTS = frozenset({
        "ilsvrc2012_val", "stage_c_plus_v2",
    })

    def _matches_import(name: str) -> bool:
        return any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in _FORBIDDEN_IMPORTS
        )

    already_loaded = sorted(name for name in sys.modules if _matches_import(name))
    if already_loaded:
        raise RuntimeError(
            f"D4 strict guard started after forbidden imports: {already_loaded}"
        )

    class _BlockedModuleFinder(importlib.abc.MetaPathFinder):
        def find_spec(
            self, fullname: str, path: object = None, target: object = None,
        ) -> None:
            if _matches_import(fullname):
                raise ImportError(f"D4 strict guard blocked import: {fullname}")
            return None

    sys.meta_path.insert(0, _BlockedModuleFinder())
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {None, ""}:
        raise RuntimeError(
            "D4 strict guard requires CUDA_VISIBLE_DEVICES to be empty"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    def _render_path(value: object) -> str | None:
        if isinstance(value, int):
            return None
        try:
            return os.fsdecode(os.fspath(value))
        except TypeError:
            return None

    def _reject_forbidden_path(value: object) -> None:
        rendered = _render_path(value)
        if rendered is None:
            return
        lowered = rendered.lower()
        path = Path(os.path.abspath(rendered))
        if lowered.endswith(_FORBIDDEN_SUFFIXES):
            raise PermissionError(
                f"D4 strict guard blocked image/model path: {rendered}"
            )
        if path.name.lower() in _FORBIDDEN_BASENAMES:
            raise PermissionError(
                f"D4 strict guard blocked candidate reserve material: {rendered}"
            )
        parts = {part.lower() for part in path.parts}
        if parts & _FORBIDDEN_DATA_COMPONENTS:
            raise PermissionError(
                f"D4 strict guard blocked reserve data path: {rendered}"
            )
        if parts & _FORBIDDEN_OUTCOME_COMPONENTS:
            raise PermissionError(
                f"D4 strict guard blocked Stage D outcome path: {rendered}"
            )
        if lowered.startswith("/dev/nvidia"):
            raise PermissionError(
                f"D4 strict guard blocked GPU device access: {rendered}"
            )

    _real_builtin_open = builtins.open
    _real_io_open = io.open
    _real_os_open = os.open
    _real_os_stat = os.stat
    _real_os_lstat = os.lstat
    _real_os_listdir = os.listdir
    _real_os_scandir = os.scandir
    _real_path_open = Path.open
    _real_path_stat = Path.stat
    _real_popen = subprocess.Popen

    def _guarded_builtin_open(file: object, *args: Any, **kwargs: Any):
        _reject_forbidden_path(file)
        return _real_builtin_open(file, *args, **kwargs)

    def _guarded_io_open(file: object, *args: Any, **kwargs: Any):
        _reject_forbidden_path(file)
        return _real_io_open(file, *args, **kwargs)

    def _guarded_os_open(path: object, *args: Any, **kwargs: Any):
        _reject_forbidden_path(path)
        return _real_os_open(path, *args, **kwargs)

    def _guarded_os_stat(path: object, *args: Any, **kwargs: Any):
        _reject_forbidden_path(path)
        return _real_os_stat(path, *args, **kwargs)

    def _guarded_os_lstat(path: object, *args: Any, **kwargs: Any):
        _reject_forbidden_path(path)
        return _real_os_lstat(path, *args, **kwargs)

    def _guarded_os_listdir(path: object = "."):
        _reject_forbidden_path(path)
        return _real_os_listdir(path)

    def _guarded_os_scandir(path: object = "."):
        _reject_forbidden_path(path)
        return _real_os_scandir(path)

    def _guarded_path_open(path: Path, *args: Any, **kwargs: Any):
        _reject_forbidden_path(path)
        return _real_path_open(path, *args, **kwargs)

    def _guarded_path_stat(path: Path, *args: Any, **kwargs: Any):
        _reject_forbidden_path(path)
        return _real_path_stat(path, *args, **kwargs)

    def _guarded_popen(args: object, *popen_args: Any, **popen_kwargs: Any):
        del popen_args, popen_kwargs
        raise PermissionError(f"D4 strict guard blocked subprocess: {args!r}")

    def _blocked_process_api(*args: Any, **kwargs: Any):
        del args, kwargs
        raise PermissionError("D4 strict guard blocked external process creation")

    builtins.open = _guarded_builtin_open
    io.open = _guarded_io_open
    os.open = _guarded_os_open
    os.stat = _guarded_os_stat
    os.lstat = _guarded_os_lstat
    os.listdir = _guarded_os_listdir
    os.scandir = _guarded_os_scandir
    Path.open = _guarded_path_open
    Path.stat = _guarded_path_stat
    subprocess.Popen = _guarded_popen
    os.system = _blocked_process_api
    os.popen = _blocked_process_api
    for _name in (
        "fork", "forkpty", "posix_spawn", "posix_spawnp",
        "spawnl", "spawnle", "spawnlp", "spawnlpe",
        "spawnv", "spawnve", "spawnvp", "spawnvpe",
    ):
        if hasattr(os, _name):
            setattr(os, _name, _blocked_process_api)

    # Torch is imported by the already-tested resource primitives, so the D4
    # guard permits tensor-only imports but blocks model construction, module
    # execution, checkpoint loading, autograd and every CUDA entry used by the
    # production backend.
    import torch as _torch

    def _blocked_torch_api(*args: Any, **kwargs: Any):
        del args, kwargs
        raise PermissionError(
            "D4 strict guard blocked model, autograd, or CUDA execution"
        )

    _torch.nn.Module.__init__ = _blocked_torch_api
    _torch.nn.Module._call_impl = _blocked_torch_api
    _torch.load = _blocked_torch_api
    _torch.jit.load = _blocked_torch_api
    _torch.hub.load = _blocked_torch_api
    _torch.autograd.grad = _blocked_torch_api
    _torch.Tensor.backward = _blocked_torch_api
    _torch.cuda.init = _blocked_torch_api
    _torch.cuda._lazy_init = _blocked_torch_api
    _torch.cuda.set_device = _blocked_torch_api
    _torch.cuda.synchronize = _blocked_torch_api
    _torch.cuda.is_available = lambda: False
    os.environ["VITNNS_D4_GUARD_ACTIVE"] = "1"
