from __future__ import annotations

import hashlib
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


_CACHE_ENVIRONMENT_KEYS = (
    "CUDA_CACHE_PATH",
    "CUTE_DSL_CACHE_DIR",
    "CUTLASS_CACHE_DIR",
    "XDG_CACHE_HOME",
)
_FINGERPRINT_PRIORITY = {
    ".cubin": 0,
    ".fatbin": 1,
    ".sass": 2,
    ".ptx": 3,
    ".so": 4,
}


@dataclass(frozen=True, slots=True)
class FileState:
    size: int
    modified_ns: int


@dataclass(frozen=True, slots=True)
class ArtifactChange:
    path: str
    change: str
    size: int
    sha256: str
    suffix: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def discover_cache_roots(
    environment: Mapping[str, str],
    *,
    home: Path | None = None,
    temporary_roots: Sequence[Path] = (Path("/tmp"), Path("/var/tmp")),
) -> tuple[Path, ...]:
    candidates: list[Path] = []
    cache_home = home or Path.home()
    candidates.append(cache_home / ".cache")
    for key in _CACHE_ENVIRONMENT_KEYS:
        value = environment.get(key)
        if value:
            candidates.append(Path(value))
    candidates.extend(temporary_roots)
    roots: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        roots.append(resolved)
    return tuple(roots)


def snapshot_files(
    roots: Sequence[Path], *, max_files: int = 50_000
) -> dict[str, FileState]:
    snapshot: dict[str, FileState] = {}
    for root in roots:
        for directory, _, filenames in os.walk(root, followlinks=False):
            for filename in filenames:
                path = Path(directory) / filename
                try:
                    if path.is_symlink() or not path.is_file():
                        continue
                    stat = path.stat()
                except OSError:
                    continue
                snapshot[str(path)] = FileState(stat.st_size, stat.st_mtime_ns)
                if len(snapshot) >= max_files:
                    raise RuntimeError(
                        f"CuTe diagnostic snapshot exceeded {max_files} files"
                    )
    return snapshot


def artifact_changes(
    before: Mapping[str, FileState],
    after: Mapping[str, FileState],
    *,
    maximum_hash_bytes: int = 1_000_000_000,
) -> tuple[ArtifactChange, ...]:
    changes: list[ArtifactChange] = []
    for path_text in sorted(after):
        state = after[path_text]
        previous = before.get(path_text)
        if previous == state:
            continue
        if state.size > maximum_hash_bytes:
            continue
        path = Path(path_text)
        try:
            digest = _sha256_file(path)
        except OSError:
            continue
        changes.append(
            ArtifactChange(
                path_text,
                "created" if previous is None else "modified",
                state.size,
                digest,
                path.suffix.casefold(),
            )
        )
    return tuple(changes)


def select_fingerprint_candidate(
    changes: Sequence[ArtifactChange],
) -> Mapping[str, object] | None:
    candidates = [item for item in changes if item.suffix in _FINGERPRINT_PRIORITY]
    if not candidates:
        return None
    selected = min(
        candidates,
        key=lambda item: (_FINGERPRINT_PRIORITY[item.suffix], item.path),
    )
    kind = {
        ".cubin": "cubin_sha256",
        ".fatbin": "fatbin_sha256",
        ".sass": "normalized_sass_candidate_sha256",
        ".ptx": "ptx_sha256",
        ".so": "shared_object_sha256",
    }[selected.suffix]
    return {"kind": kind, **selected.as_dict()}


def describe_kernel_callable(
    kernel_callable, keywords: Mapping[str, object]
) -> dict[str, object]:
    value: dict[str, object] = {
        "type_module": type(kernel_callable).__module__,
        "type_name": type(kernel_callable).__qualname__,
        "callable_module": getattr(kernel_callable, "__module__", None),
        "callable_name": getattr(kernel_callable, "__name__", None),
        "callable_qualname": getattr(kernel_callable, "__qualname__", None),
        "repr": _bounded_repr(kernel_callable),
        "benchmark_keyword_names": sorted(str(name) for name in keywords),
    }
    attributes = getattr(kernel_callable, "__dict__", None)
    if isinstance(attributes, Mapping):
        selected = {}
        for name, item in sorted(attributes.items(), key=lambda pair: str(pair[0])):
            folded = str(name).casefold()
            if not any(token in folded for token in ("kernel", "name", "module", "ptx", "cubin")):
                continue
            if isinstance(item, (str, int, float, bool, type(None))):
                selected[str(name)] = item
            elif isinstance(item, Path):
                selected[str(name)] = str(item)
            else:
                selected[str(name)] = _bounded_repr(item)
        value["selected_attributes"] = selected
        kernel_info = attributes.get("kernel_info")
        if isinstance(kernel_info, Mapping):
            value["kernel_names"] = [str(name) for name in kernel_info]
        ir_module = attributes.get("ir_module")
        if ir_module is not None:
            value["mlir"] = describe_mlir(ir_module)
        runtime_artifacts = []
        for attribute_name in ("jit_module", "gpu_module", "module"):
            runtime_object = attributes.get(attribute_name)
            if runtime_object is not None:
                runtime_artifacts.extend(
                    describe_runtime_artifacts(runtime_object, attribute_name)
                )
        value["runtime_artifacts"] = runtime_artifacts
    workspace = keywords.get("kernel_arguments")
    if workspace is not None:
        value["workspace_type"] = (
            f"{type(workspace).__module__}.{type(workspace).__qualname__}"
        )
        value["argument_types"] = [
            f"{type(item).__module__}.{type(item).__qualname__}"
            for item in getattr(workspace, "args", ())
        ]
        value["named_argument_types"] = {
            str(name): f"{type(item).__module__}.{type(item).__qualname__}"
            for name, item in getattr(workspace, "kwargs", {}).items()
        }
    return value


def describe_mlir(ir_module, *, maximum_text_bytes: int = 1_000_000) -> dict[str, object]:
    raw = str(ir_module)
    normalized = normalize_mlir(raw)
    encoded = normalized.encode("utf-8")
    return {
        "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "normalized_sha256": hashlib.sha256(encoded).hexdigest(),
        "normalized_text": normalized if len(encoded) <= maximum_text_bytes else None,
        "normalized_bytes": len(encoded),
        "text_omitted": len(encoded) > maximum_text_bytes,
    }


def normalize_mlir(value: str) -> str:
    normalized = re.sub(r"0x[0-9a-fA-F]+", "<address>", value)
    normalized = re.sub(r"/tmp/[^\s\"']+", "<tmp-path>", normalized)
    normalized = re.sub(
        r"object_at_+(?:address_)?[0-9a-fA-F_]+",
        "object_at_<identity>",
        normalized,
    )
    return "\n".join(line.rstrip() for line in normalized.splitlines()).strip() + "\n"


def describe_runtime_artifacts(
    runtime_object, prefix: str, *, depth: int = 0, maximum_depth: int = 2
) -> list[dict[str, object]]:
    if depth > maximum_depth:
        return []
    attributes = getattr(runtime_object, "__dict__", None)
    if not isinstance(attributes, Mapping):
        return []
    artifacts: list[dict[str, object]] = []
    for name, item in sorted(attributes.items(), key=lambda pair: str(pair[0])):
        path = f"{prefix}.{name}"
        folded = str(name).casefold()
        kind = _runtime_artifact_kind(folded, item)
        if kind is not None:
            encoded = (
                bytes(item)
                if isinstance(item, (bytes, bytearray, memoryview))
                else str(item).encode("utf-8")
            )
            artifacts.append(
                {
                    "attribute": path,
                    "kind": kind,
                    "size": len(encoded),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                }
            )
            continue
        if depth < maximum_depth and any(
            token in folded for token in ("module", "binary", "image", "code", "asm")
        ):
            artifacts.extend(
                describe_runtime_artifacts(
                    item, path, depth=depth + 1, maximum_depth=maximum_depth
                )
            )
    return artifacts


def select_runtime_fingerprint(
    diagnostics: Mapping[str, object],
) -> Mapping[str, object] | None:
    priorities = {"cubin": 0, "fatbin": 1, "sass": 2, "ptx": 3}
    candidates = diagnostics.get("runtime_artifacts", ())
    if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
        usable = [
            item
            for item in candidates
            if isinstance(item, Mapping) and item.get("kind") in priorities
        ]
        if usable:
            return dict(
                min(
                    usable,
                    key=lambda item: (
                        priorities[str(item["kind"])],
                        str(item.get("attribute", "")),
                    ),
                )
            )
    mlir = diagnostics.get("mlir")
    if isinstance(mlir, Mapping) and mlir.get("normalized_sha256"):
        return {
            "kind": "normalized_mlir",
            "sha256": mlir["normalized_sha256"],
            "size": mlir.get("normalized_bytes"),
            "attribute": "ir_module",
        }
    return None


def loaded_module_paths() -> tuple[str, ...]:
    try:
        lines = Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    paths = set()
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        path = fields[5]
        folded = path.casefold()
        if any(token in folded for token in ("cuda", "cutlass", "cute", "nvidia")):
            paths.add(path)
    return tuple(sorted(paths))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_repr(value: object, limit: int = 500) -> str:
    rendered = repr(value)
    return rendered if len(rendered) <= limit else rendered[:limit] + "..."


def _runtime_artifact_kind(name: str, value: object) -> str | None:
    if not isinstance(value, (str, bytes, bytearray, memoryview)):
        return None
    for kind in ("cubin", "fatbin", "sass", "ptx"):
        if kind in name:
            return kind
    if isinstance(value, str):
        stripped = value.lstrip()
        if stripped.startswith(".version") and ".target" in value:
            return "ptx"
    return None
