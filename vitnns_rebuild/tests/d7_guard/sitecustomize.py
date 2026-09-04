"""Process-wide non-reserve guard for Stage D D7 tests."""

from __future__ import annotations

import builtins
import io
import os
from pathlib import Path
import subprocess


if os.environ.get("VITNNS_D7_STRICT") == "1":
    real_builtin_open = builtins.open
    real_io_open = io.open
    real_os_open = os.open
    real_os_stat = os.stat
    real_os_lstat = os.lstat
    real_os_listdir = os.listdir
    real_os_scandir = os.scandir
    real_path_open = Path.open
    real_path_stat = Path.stat
    real_popen = subprocess.Popen

    def render(value: object) -> str | None:
        if type(value) is int:
            return None
        try:
            return os.path.abspath(os.fsdecode(os.fspath(value)))
        except TypeError:
            return None

    def forbidden(path: str) -> bool:
        lowered = path.casefold()
        parts = tuple(part.casefold() for part in Path(path).parts)
        if "ilsvrc2012_val" in parts and lowered.endswith((".jpeg", ".jpg")):
            return True
        return any(
            part.startswith(("stage_d_results", "stage_d_outcomes", "vitnns_d8_formal", "vitnns_d9_formal", "vitnns_d10_formal"))
            for part in parts
        )

    def reject(value: object) -> None:
        path = render(value)
        if path is not None and forbidden(path):
            raise PermissionError(f"D7 strict guard blocked reserve/formal path: {path}")

    def guarded_open(file: object, *args, **kwargs):
        reject(file)
        return real_builtin_open(file, *args, **kwargs)

    def guarded_io_open(file: object, *args, **kwargs):
        reject(file)
        return real_io_open(file, *args, **kwargs)

    def guarded_os_open(path: object, *args, **kwargs):
        reject(path)
        return real_os_open(path, *args, **kwargs)

    def guarded_stat(path: object, *args, **kwargs):
        reject(path)
        return real_os_stat(path, *args, **kwargs)

    def guarded_lstat(path: object, *args, **kwargs):
        reject(path)
        return real_os_lstat(path, *args, **kwargs)

    def guarded_listdir(path: object = "."):
        reject(path)
        return real_os_listdir(path)

    def guarded_scandir(path: object = "."):
        reject(path)
        return real_os_scandir(path)

    def guarded_path_open(path: Path, *args, **kwargs):
        reject(path)
        return real_path_open(path, *args, **kwargs)

    def guarded_path_stat(path: Path, *args, **kwargs):
        reject(path)
        return real_path_stat(path, *args, **kwargs)

    def guarded_popen(args: object, *popen_args, **kwargs):
        command = args[0] if isinstance(args, (list, tuple)) and args else args
        name = Path(os.fsdecode(command)).name if command is not None else ""
        if name not in {"git", "git-lfs"}:
            raise PermissionError(f"D7 strict guard blocked subprocess: {args!r}")
        return real_popen(args, *popen_args, **kwargs)

    builtins.open = guarded_open
    io.open = guarded_io_open
    os.open = guarded_os_open
    os.stat = guarded_stat
    os.lstat = guarded_lstat
    os.listdir = guarded_listdir
    os.scandir = guarded_scandir
    Path.open = guarded_path_open
    Path.stat = guarded_path_stat
    subprocess.Popen = guarded_popen
    os.environ["VITNNS_D7_GUARD_ACTIVE"] = "1"
