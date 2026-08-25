"""Process-wide metadata/static-only safety guard for Stage D D2 tests."""

from __future__ import annotations

import builtins
import importlib.abc
import io
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


if os.environ.get("VITNNS_D2_STRICT") == "1":
    _FORBIDDEN_IMPORTS = (
        "PIL", "cv2", "timm", "torch", "torchvision",
        "vitnns.attribution", "vitnns.dataset", "vitnns.extractor",
        "vitnns.modeling", "vitnns.runner", "vitnns.stage_c_runner",
    )
    _FORBIDDEN_SUFFIXES = (
        ".jpeg", ".jpg", ".png", ".gif", ".webp", ".bmp",
        ".pt", ".pth", ".safetensors", ".npy", ".npz",
    )
    _FORBIDDEN_OUTCOME_COMPONENTS = (
        "stage_d_results", "stage_d_outcomes", "stage_d_formal",
    )

    def _matches_import(name: str) -> bool:
        return any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in _FORBIDDEN_IMPORTS
        )

    already_loaded = sorted(name for name in sys.modules if _matches_import(name))
    if already_loaded:
        raise RuntimeError(
            f"D2 strict guard started after forbidden imports: {already_loaded}"
        )

    class _BlockedModuleFinder(importlib.abc.MetaPathFinder):
        def find_spec(
            self, fullname: str, path: object = None, target: object = None,
        ) -> None:
            if _matches_import(fullname):
                raise ImportError(f"D2 strict guard blocked import: {fullname}")
            return None

    sys.meta_path.insert(0, _BlockedModuleFinder())

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
        if lowered.endswith(_FORBIDDEN_SUFFIXES):
            raise PermissionError(
                f"D2 strict guard blocked image/model/binary path: {rendered}"
            )
        parts = {part.lower() for part in Path(os.path.abspath(rendered)).parts}
        if parts & set(_FORBIDDEN_OUTCOME_COMPONENTS):
            raise PermissionError(
                f"D2 strict guard blocked Stage D outcome path: {rendered}"
            )
        if lowered.startswith("/dev/nvidia"):
            raise PermissionError(
                f"D2 strict guard blocked GPU device access: {rendered}"
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
        command = args[0] if isinstance(args, (list, tuple)) and args else args
        rendered = _render_path(command)
        if rendered is None or Path(rendered).name != "git":
            raise PermissionError(f"D2 strict guard blocked subprocess: {args!r}")
        return _real_popen(args, *popen_args, **popen_kwargs)

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
    os.environ["VITNNS_D2_GUARD_ACTIVE"] = "1"
