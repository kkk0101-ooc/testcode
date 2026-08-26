"""Process-wide synthetic-evidence-only safety guard for Stage D D5 tests.

D5 may inspect ordinary source/control files and synthetic NPY/NPZ evidence.
It must not inspect image bytes, model weights, candidate-reserve material, or
real Stage D outcome/formal roots.  Loading this module through
``sitecustomize`` installs the guard before pytest collection.
"""

from __future__ import annotations

import builtins
import importlib.abc
import io
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any


def _install_d5_guard() -> None:
    forbidden_imports = (
        "PIL",
        "cv2",
        "timm",
        "torchvision",
        "vitnns.attribution",
        "vitnns.dataset",
        "vitnns.display",
        "vitnns.extractor",
        "vitnns.legacy",
        "vitnns.modeling",
        "vitnns.runner",
        "vitnns.stage_b_runner",
        "vitnns.stage_c_attention",
        "vitnns.stage_c_class_logit",
        "vitnns.stage_c_fidelity",
        "vitnns.stage_c_perturbations",
        "vitnns.stage_c_stability",
        "vitnns.stage_c_value_path",
    )
    # NPY/NPZ and ordinary BIN/DAT files are intentionally allowed.  The
    # latter are used by synthetic layout/tamper fixtures and are not, by
    # themselves, evidence that a path contains model weights.
    forbidden_suffixes = (
        ".jpeg", ".jpg", ".png", ".gif", ".webp", ".bmp",
        ".tif", ".tiff", ".ppm", ".pgm", ".pbm", ".pnm",
        ".tga", ".ico", ".svg", ".avif", ".heic", ".heif",
        ".pt", ".pth", ".safetensors", ".ckpt", ".onnx",
        ".weights",
    )
    forbidden_basenames = frozenset({
        "stage_d_v2_reserve.json",
        "stage_d_v2_reserve_access_log.json",
        "vit_base_patch16_224_weights.bin",
        "vit_base_patch16_224_weights.dat",
        "vit_base_patch16_224_full.bin",
        "vit_base_patch16_224_full.dat",
    })
    forbidden_outcome_components = frozenset({
        "stage_d_results",
        "stage_d_outcomes",
        "stage_d_formal",
        "stage_d_smoke",
        "predictions",
        "logits",
        "delta_fid",
    })
    formal_component = re.compile(
        r"^(?:(?:vitnns_)?stage_d_(?:results?|outcomes?|formal|smoke)"
        r"|vitnns_d(?:8|9|10)_formal)"
        r"(?:_[0-9a-f]{7,40})?(?:_[0-9]{8})?$"
    )

    def matches_import(name: str) -> bool:
        return any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in forbidden_imports
        )

    already_loaded = sorted(name for name in sys.modules if matches_import(name))
    if already_loaded:
        raise RuntimeError(
            f"D5 strict guard started after forbidden imports: {already_loaded}"
        )

    class BlockedModuleFinder(importlib.abc.MetaPathFinder):
        def find_spec(
            self, fullname: str, path: object = None, target: object = None,
        ) -> None:
            if matches_import(fullname):
                raise ImportError(f"D5 strict guard blocked import: {fullname}")
            return None

    sys.meta_path.insert(0, BlockedModuleFinder())
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {None, ""}:
        # ``site`` catches ordinary Exceptions from sitecustomize and resumes
        # startup, which would silently discard this process-wide boundary.
        raise SystemExit(
            "D5 strict guard requires CUDA_VISIBLE_DEVICES to be empty"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Keep originals only in this installer's closure.  In particular, do not
    # publish bypass handles such as ``sitecustomize._real_os_open``.
    real_builtin_open = builtins.open
    real_io_open = io.open
    real_os_open = os.open
    real_os_fdopen = os.fdopen
    real_os_stat = os.stat
    real_os_lstat = os.lstat
    real_os_fstat = os.fstat
    real_os_listdir = os.listdir
    real_os_scandir = os.scandir
    real_os_readlink = os.readlink
    real_os_access = os.access
    real_os_read = os.read
    real_os_pread = getattr(os, "pread", None)
    real_os_readv = getattr(os, "readv", None)
    real_os_preadv = getattr(os, "preadv", None)
    real_path_open = Path.open
    real_path_stat = Path.stat
    path_accessor = getattr(Path, "_accessor", None)
    real_popen = subprocess.Popen

    def render_path(value: object) -> str | None:
        try:
            return os.fsdecode(os.fspath(value))
        except TypeError:
            return None

    def fd_path(descriptor: object) -> str | None:
        if type(descriptor) is not int or descriptor < 0:
            return None
        try:
            target = os.fsdecode(
                real_os_readlink(f"/proc/self/fd/{descriptor}")
            )
        except OSError:
            return None
        if target.endswith(" (deleted)"):
            target = target[:-10]
        if os.path.isabs(target):
            return target
        if target.startswith(("pipe:[", "socket:[", "anon_inode:")):
            return ""
        raise PermissionError(
            f"D5 strict guard rejected unresolved file descriptor: {descriptor}"
        )

    def lexical_path(value: object, *, dir_fd: object = None) -> str | None:
        if type(value) is int:
            target = fd_path(value)
            if target is None:
                raise PermissionError(
                    f"D5 strict guard rejected unresolved file descriptor: {value}"
                )
            return target or None
        rendered = render_path(value)
        if rendered is None:
            return None
        if os.path.isabs(rendered):
            return os.path.normpath(rendered)
        if type(dir_fd) is int:
            base = fd_path(dir_fd)
            if not base:
                raise PermissionError(
                    f"D5 strict guard rejected unresolved dir_fd: {dir_fd}"
                )
        else:
            base = os.getcwd()
        return os.path.normpath(os.path.join(base, rendered))

    def resolved_path(path: str) -> str:
        """Resolve existing symlinks without calling patched path APIs."""

        pending = list(Path(os.path.normpath(path)).parts[1:])
        resolved = os.path.sep
        followed = 0
        while pending:
            component = pending.pop(0)
            candidate = os.path.join(resolved, component)
            reason = forbidden_reason(candidate)
            if reason is not None:
                raise PermissionError(
                    f"D5 strict guard blocked {reason}: {candidate}"
                )
            try:
                metadata = real_os_lstat(candidate)
            except (FileNotFoundError, NotADirectoryError):
                return os.path.normpath(
                    os.path.join(candidate, *pending) if pending else candidate
                )
            except OSError as error:
                raise PermissionError(
                    f"D5 strict guard could not safely resolve path: {path}"
                ) from error
            if not stat.S_ISLNK(metadata.st_mode):
                resolved = candidate
                continue
            followed += 1
            if followed > 40:
                raise PermissionError(
                    f"D5 strict guard rejected a symlink loop: {path}"
                )
            target = os.fsdecode(real_os_readlink(candidate))
            target = (
                target if os.path.isabs(target)
                else os.path.join(os.path.dirname(candidate), target)
            )
            combined = os.path.normpath(
                os.path.join(target, *pending) if pending else target
            )
            reason = forbidden_reason(combined)
            if reason is not None:
                raise PermissionError(
                    f"D5 strict guard blocked {reason}: {combined}"
                )
            pending = list(Path(combined).parts[1:])
            resolved = os.path.sep
        return os.path.normpath(resolved)

    def forbidden_reason(path: str) -> str | None:
        lowered = path.casefold()
        basename = os.path.basename(lowered)
        components = tuple(part.casefold() for part in Path(path).parts)
        if lowered.endswith(forbidden_suffixes):
            return "image/model path"
        if basename in forbidden_basenames:
            return "candidate reserve or model-weight material"
        if "ilsvrc2012_val" in components:
            return "reserve image tree"
        if (
            set(components) & forbidden_outcome_components
            or any(formal_component.fullmatch(part) for part in components)
            or any(
                part.startswith("stage_d_")
                and ("formal" in part or "smoke" in part)
                for part in components
            )
        ):
            return "Stage D outcome/formal/smoke path"
        if lowered == "/dev/nvidia" or lowered.startswith("/dev/nvidia"):
            return "GPU device access"
        return None

    def reject_forbidden_path(
        value: object, *, dir_fd: object = None, known_path: str | None = None,
    ) -> str | None:
        lexical = known_path or lexical_path(value, dir_fd=dir_fd)
        if lexical is None:
            return None
        reason = forbidden_reason(lexical)
        if reason is not None:
            raise PermissionError(
                f"D5 strict guard blocked {reason}: {lexical}"
            )
        resolved = resolved_path(lexical)
        reason = forbidden_reason(resolved)
        if reason is not None:
            raise PermissionError(
                f"D5 strict guard blocked {reason}: {resolved}"
            )
        return resolved

    def guarded_builtin_open(file: object, *args: Any, **kwargs: Any):
        reject_forbidden_path(file)
        return real_builtin_open(file, *args, **kwargs)

    def guarded_io_open(file: object, *args: Any, **kwargs: Any):
        reject_forbidden_path(file)
        return real_io_open(file, *args, **kwargs)

    def guarded_os_open(path: object, *args: Any, **kwargs: Any):
        dir_fd = kwargs.get("dir_fd")
        reject_forbidden_path(path, dir_fd=dir_fd)
        flags = args[0] if args else kwargs.get("flags")
        if (
            isinstance(flags, int)
            and not flags & os.O_CREAT
            and hasattr(os, "O_NOFOLLOW")
        ):
            flags |= os.O_NOFOLLOW
            if args:
                args = (flags, *args[1:])
            else:
                kwargs["flags"] = flags
        return real_os_open(path, *args, **kwargs)

    def guarded_os_fdopen(fd: int, *args: Any, **kwargs: Any):
        reject_forbidden_path(fd)
        return real_os_fdopen(fd, *args, **kwargs)

    def guarded_os_stat(path: object, *args: Any, **kwargs: Any):
        dir_fd = kwargs.get("dir_fd")
        reject_forbidden_path(path, dir_fd=dir_fd)
        return real_os_stat(path, *args, **kwargs)

    def guarded_os_lstat(path: object, *args: Any, **kwargs: Any):
        dir_fd = kwargs.get("dir_fd")
        reject_forbidden_path(path, dir_fd=dir_fd)
        return real_os_lstat(path, *args, **kwargs)

    def guarded_os_fstat(fd: int):
        reject_forbidden_path(fd)
        return real_os_fstat(fd)

    def guarded_os_listdir(path: object = "."):
        reject_forbidden_path(path)
        return real_os_listdir(path)

    class GuardedDirEntry:
        __slots__ = ("_entry", "_effective_path")

        def __init__(self, entry: os.DirEntry[Any], effective_path: str) -> None:
            self._entry = entry
            self._effective_path = effective_path

        @property
        def name(self):
            return self._entry.name

        @property
        def path(self):
            return self._entry.path

        def __fspath__(self):
            return os.fspath(self._entry)

        def __repr__(self) -> str:
            return repr(self._entry)

        def _check(self) -> None:
            reject_forbidden_path(
                self._effective_path, known_path=self._effective_path,
            )

        def inode(self):
            self._check()
            return self._entry.inode()

        def is_dir(self, *args: Any, **kwargs: Any):
            self._check()
            return self._entry.is_dir(*args, **kwargs)

        def is_file(self, *args: Any, **kwargs: Any):
            self._check()
            return self._entry.is_file(*args, **kwargs)

        def is_symlink(self):
            self._check()
            return self._entry.is_symlink()

        def stat(self, *args: Any, **kwargs: Any):
            self._check()
            return self._entry.stat(*args, **kwargs)

    class GuardedScandirIterator:
        __slots__ = ("_iterator", "_base")

        def __init__(self, iterator: Any, base: str | None) -> None:
            self._iterator = iterator
            self._base = base

        def __enter__(self):
            self._iterator.__enter__()
            return self

        def __exit__(self, *args: Any):
            return self._iterator.__exit__(*args)

        def __iter__(self):
            return self

        def __next__(self):
            entry = next(self._iterator)
            effective = (
                os.path.join(self._base, os.fsdecode(entry.name))
                if self._base is not None else lexical_path(entry.path)
            )
            effective = effective or os.fsdecode(entry.path)
            return GuardedDirEntry(entry, effective)

        def close(self) -> None:
            self._iterator.close()

    def guarded_os_scandir(path: object = "."):
        base = reject_forbidden_path(path)
        return GuardedScandirIterator(real_os_scandir(path), base)

    def guarded_os_readlink(path: object, *args: Any, **kwargs: Any):
        dir_fd = kwargs.get("dir_fd")
        reject_forbidden_path(path, dir_fd=dir_fd)
        return real_os_readlink(path, *args, **kwargs)

    def guarded_os_access(path: object, *args: Any, **kwargs: Any):
        dir_fd = kwargs.get("dir_fd")
        reject_forbidden_path(path, dir_fd=dir_fd)
        return real_os_access(path, *args, **kwargs)

    def guarded_os_read(fd: int, *args: Any, **kwargs: Any):
        reject_forbidden_path(fd)
        return real_os_read(fd, *args, **kwargs)

    def guarded_path_open(path: Path, *args: Any, **kwargs: Any):
        reject_forbidden_path(path)
        return real_path_open(path, *args, **kwargs)

    def guarded_path_stat(path: Path, *args: Any, **kwargs: Any):
        reject_forbidden_path(path)
        return real_path_stat(path, *args, **kwargs)

    def guarded_path_lstat(path: Path, *args: Any, **kwargs: Any):
        reject_forbidden_path(path)
        return real_path_stat(path, *args, follow_symlinks=False, **kwargs)

    def git_invocation_allowed(args: object, kwargs: dict[str, Any]) -> bool:
        if kwargs.get("shell") or kwargs.get("executable") is not None:
            return False
        if not isinstance(args, (list, tuple)) or len(args) < 2:
            return False
        rendered = [render_path(item) for item in args]
        if any(item is None for item in rendered):
            return False
        tokens = [item for item in rendered if item is not None]
        executable = tokens[0]
        if executable not in {"git", "/usr/bin/git", "/bin/git"}:
            return False
        index = 1
        if index >= len(tokens):
            return False
        command = tokens[index]
        arguments = tokens[index + 1:]
        if command == "rev-parse":
            if arguments in (["--show-toplevel"], ["--show-prefix"]):
                return True
            verify = len(arguments) == 2 and arguments[0] == "--verify"
            revision = arguments[-1] if arguments else ""
            return (len(arguments) == 1 or verify) and re.fullmatch(
                r"(?:HEAD|[0-9a-f]{40}|refs/archive/[A-Za-z0-9._/-]+)"
                r"(?:\^\{(?:commit|tree)\})?",
                revision,
            ) is not None
        if command == "status":
            return arguments in (
                ["--short"],
                ["--porcelain"],
                ["--porcelain=v1"],
                ["--porcelain=v1", "--untracked-files=all"],
                ["--porcelain=v1", "--untracked-files=no"],
            )
        return False

    def guarded_popen(args: object, *popen_args: Any, **popen_kwargs: Any):
        if popen_args:
            raise PermissionError(
                "D5 strict guard blocked positional subprocess controls"
            )
        if not git_invocation_allowed(args, popen_kwargs):
            raise PermissionError(f"D5 strict guard blocked subprocess: {args!r}")
        cwd = popen_kwargs.get("cwd")
        if cwd is not None:
            reject_forbidden_path(cwd)
        if popen_kwargs.get("preexec_fn") is not None:
            raise PermissionError(
                "D5 strict guard blocked subprocess pre-execution hook"
            )
        safe_environment = {
            "PATH": "/usr/bin:/bin",
            "GIT_CONFIG_COUNT": "2",
            "GIT_CONFIG_KEY_0": "core.fsmonitor",
            "GIT_CONFIG_VALUE_0": "false",
            "GIT_CONFIG_KEY_1": "core.hooksPath",
            "GIT_CONFIG_VALUE_1": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_LITERAL_PATHSPECS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_PAGER": "cat",
            "LANG": "C",
            "LC_ALL": "C",
        }
        popen_kwargs["env"] = safe_environment
        command = list(args)
        command[0] = "/usr/bin/git"
        return real_popen(command, **popen_kwargs)

    def blocked_process_api(*args: Any, **kwargs: Any):
        del args, kwargs
        raise PermissionError("D5 strict guard blocked external process creation")

    builtins.open = guarded_builtin_open
    io.open = guarded_io_open
    os.open = guarded_os_open
    os.fdopen = guarded_os_fdopen
    os.stat = guarded_os_stat
    os.lstat = guarded_os_lstat
    os.fstat = guarded_os_fstat
    os.listdir = guarded_os_listdir
    os.scandir = guarded_os_scandir
    os.readlink = guarded_os_readlink
    os.access = guarded_os_access
    os.read = guarded_os_read
    if real_os_pread is not None:
        def guarded_os_pread(fd: int, *args: Any, **kwargs: Any):
            reject_forbidden_path(fd)
            return real_os_pread(fd, *args, **kwargs)
        os.pread = guarded_os_pread
    if real_os_readv is not None:
        def guarded_os_readv(fd: int, *args: Any, **kwargs: Any):
            reject_forbidden_path(fd)
            return real_os_readv(fd, *args, **kwargs)
        os.readv = guarded_os_readv
    if real_os_preadv is not None:
        def guarded_os_preadv(fd: int, *args: Any, **kwargs: Any):
            reject_forbidden_path(fd)
            return real_os_preadv(fd, *args, **kwargs)
        os.preadv = guarded_os_preadv
    Path.open = guarded_path_open
    Path.stat = guarded_path_stat
    Path.lstat = guarded_path_lstat
    if path_accessor is not None:
        type(path_accessor).open = staticmethod(guarded_io_open)
        type(path_accessor).stat = staticmethod(guarded_os_stat)
        type(path_accessor).listdir = staticmethod(guarded_os_listdir)
        type(path_accessor).scandir = staticmethod(guarded_os_scandir)
        type(path_accessor).readlink = staticmethod(guarded_os_readlink)
    subprocess.Popen = guarded_popen
    os.system = blocked_process_api
    os.popen = blocked_process_api

    process_names = (
        "execl", "execle", "execlp", "execlpe",
        "execv", "execve", "execvp", "execvpe",
        "fork", "forkpty", "posix_spawn", "posix_spawnp",
        "spawnl", "spawnle", "spawnlp", "spawnlpe",
        "spawnv", "spawnve", "spawnvp", "spawnvpe",
    )
    for name in process_names:
        if hasattr(os, name):
            setattr(os, name, blocked_process_api)
    try:
        import posix
    except ImportError:
        pass
    else:
        for name, guarded in (
            ("open", guarded_os_open),
            ("stat", guarded_os_stat),
            ("lstat", guarded_os_lstat),
            ("fstat", guarded_os_fstat),
            ("listdir", guarded_os_listdir),
            ("scandir", guarded_os_scandir),
            ("readlink", guarded_os_readlink),
            ("access", guarded_os_access),
            ("read", guarded_os_read),
            ("pread", locals().get("guarded_os_pread")),
            ("readv", locals().get("guarded_os_readv")),
            ("preadv", locals().get("guarded_os_preadv")),
        ):
            if guarded is not None and hasattr(posix, name):
                setattr(posix, name, guarded)
        for name in process_names:
            if hasattr(posix, name):
                setattr(posix, name, blocked_process_api)

    # Torch is needed by shared type/resource modules.  Tensor-only use stays
    # available; model construction/execution, checkpoint loading, autograd,
    # and CUDA entry points are disabled.
    import torch

    def blocked_torch_api(*args: Any, **kwargs: Any):
        del args, kwargs
        raise PermissionError(
            "D5 strict guard blocked model, autograd, or CUDA execution"
        )

    torch.nn.Module.__init__ = blocked_torch_api
    torch.nn.Module.__call__ = blocked_torch_api
    torch.nn.Module.forward = blocked_torch_api
    if hasattr(torch.nn.Module, "_wrapped_call_impl"):
        torch.nn.Module._wrapped_call_impl = blocked_torch_api
    if hasattr(torch.nn.Module, "_call_impl"):
        torch.nn.Module._call_impl = blocked_torch_api
    torch.nn.Module.load_state_dict = blocked_torch_api
    torch.nn.Module._load_from_state_dict = blocked_torch_api
    torch.load = blocked_torch_api
    torch.serialization.load = blocked_torch_api
    for name in ("_load", "_legacy_load"):
        if hasattr(torch.serialization, name):
            setattr(torch.serialization, name, blocked_torch_api)
    torch.jit.load = blocked_torch_api
    torch.jit.script = blocked_torch_api
    torch.jit.trace = blocked_torch_api
    torch.hub.load = blocked_torch_api
    if hasattr(torch, "export") and hasattr(torch.export, "load"):
        torch.export.load = blocked_torch_api
    if hasattr(torch, "package") and hasattr(torch.package, "PackageImporter"):
        torch.package.PackageImporter = blocked_torch_api
    torch.inference_mode = blocked_torch_api
    torch.no_grad = blocked_torch_api
    for name in (
        "conv1d", "conv2d", "conv3d", "linear",
        "multi_head_attention_forward", "scaled_dot_product_attention",
    ):
        if hasattr(torch.nn.functional, name):
            setattr(torch.nn.functional, name, blocked_torch_api)
    for name in ("bmm", "einsum", "matmul", "mm"):
        if hasattr(torch, name):
            setattr(torch, name, blocked_torch_api)
    torch.Tensor.cuda = blocked_torch_api
    torch.Tensor.to = blocked_torch_api
    torch.autograd.grad = blocked_torch_api
    torch.autograd.backward = blocked_torch_api
    torch.Tensor.backward = blocked_torch_api
    torch_func = getattr(torch, "func", None)
    if torch_func is not None:
        for name in ("grad", "grad_and_value", "jacfwd", "jacrev", "vjp"):
            if hasattr(torch_func, name):
                setattr(torch_func, name, blocked_torch_api)
    autograd_functional = getattr(torch.autograd, "functional", None)
    if autograd_functional is not None:
        for name in ("hessian", "hvp", "jacobian", "jvp", "vhp", "vjp"):
            if hasattr(autograd_functional, name):
                setattr(autograd_functional, name, blocked_torch_api)
    for name in (
        "current_device", "empty_cache", "get_device_capability",
        "get_device_name", "get_device_properties", "init", "_lazy_init",
        "manual_seed", "manual_seed_all", "max_memory_allocated",
        "max_memory_reserved", "memory_allocated", "memory_reserved",
        "reset_peak_memory_stats", "set_device", "synchronize",
    ):
        if hasattr(torch.cuda, name):
            setattr(torch.cuda, name, blocked_torch_api)
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
    os.environ["VITNNS_D5_GUARD_ACTIVE"] = "1"


if os.environ.get("VITNNS_D5_STRICT") == "1":
    try:
        _install_d5_guard()
    except SystemExit:
        raise
    except Exception as error:
        # Python's site module reports and suppresses ordinary sitecustomize
        # exceptions.  Convert installation failures into a fatal startup so
        # a strict process can never continue with a partial or absent guard.
        raise SystemExit(f"D5 strict guard installation failed: {error}") from error
del _install_d5_guard
