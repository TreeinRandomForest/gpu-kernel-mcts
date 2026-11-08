"""Opt-in local package overlay for version-pinned Nebius worker experiments."""

from __future__ import annotations

import io
import json
import shlex
import subprocess
import tarfile
import time
from pathlib import Path


def package_bundle(root: Path) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        archive.add(root / "src/kernel_mcts", arcname="kernel_mcts",
                    filter=lambda info: None if "__pycache__" in info.name else info)
    return stream.getvalue()


def _checked(command: list[str], *, input_bytes: bytes | None = None,
             timeout: float = 180) -> bytes:
    result = subprocess.run(command, input=input_bytes, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"worker overlay command failed ({result.returncode}): "
                           + result.stderr.decode(errors="replace")[-1200:])
    return result.stdout


def overlay_nebius_worker(worker, ssh_private_key: Path, username: str,
                          bundle: bytes) -> None:
    """Replace the worker package before root evaluation, then await readiness."""
    instance = json.loads(_checked([
        "nebius", "compute", "instance", "get", "--id", worker.instance.instance_id,
        "--format", "json", "--no-check-update",
    ], timeout=30))
    ip = instance["status"]["network_interfaces"][0]["public_ip_address"]["address"].split("/")[0]
    ssh = ["ssh", "-F", "/dev/null", "-i", str(ssh_private_key),
           "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
           f"{username}@{ip}"]
    package_path = _checked([*ssh, shlex.join([
        "sudo", "docker", "exec", "kernel-mcts-worker", "python", "-c",
        "import kernel_mcts,pathlib; print(pathlib.Path(kernel_mcts.__file__).parent.parent)",
    ])]).decode().strip()
    if not package_path.startswith("/opt/venv/") or not package_path.endswith("site-packages"):
        raise RuntimeError("unexpected worker package location")
    _checked([*ssh, "sudo", "docker", "exec", "-i", "kernel-mcts-worker", "tar", "-xz",
              "-C", package_path], input_bytes=bundle)
    _checked([*ssh, "sudo", "docker", "restart", "kernel-mcts-worker"], timeout=120)
    for _ in range(30):
        try:
            worker._transport.get_environment_manifest()
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError("overlaid worker did not become ready")
