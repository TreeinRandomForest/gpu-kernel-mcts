from __future__ import annotations

import json
import os
import re
import secrets
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterator, Mapping, Protocol, Sequence

from .domain import EvaluationResult, KernelProgram, ProposalStatus, WorkloadContract
from .providers import (
    EnvironmentManifest,
    GPUWorker,
    HardwareSpec,
    ManifestEventSink,
    WorkerEndpoint,
    WorkerTransport,
    WorkerTransportFactory,
    validate_environment,
)
from .worker_protocol import HTTPWorkerTransport


class NebiusError(RuntimeError):
    """A sanitized Nebius provisioning or remote-worker failure."""


class _NebiusCommandError(NebiusError):
    """Internal command failure retaining output for resource recovery."""

    def __init__(self, message: str, result: subprocess.CompletedProcess[str]) -> None:
        super().__init__(message)
        self.result = result


@dataclass(frozen=True, slots=True)
class NebiusConfig:
    image: str
    project_id: str
    subnet_id: str
    ssh_public_key: Path
    ssh_private_key: Path
    username: str
    platform: str = "gpu-h100-sxm"
    preset: str = "1gpu-16vcpu-200gb"
    image_family: str = "ubuntu24.04-cuda13.0"
    disk_size_gib: int = 200
    disk_type: str = "network_ssd"
    worker_port: int = 8000
    startup_timeout_seconds: float = 900.0
    poll_interval_seconds: float = 5.0
    resource_name_prefix: str = "gpu-kernel-mcts"
    nebius_command: str = "nebius"
    ssh_command: str = "ssh"
    project_git_commit: str | None = None
    project_dirty_tree: bool | None = None
    container_digest: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.image
            or not self.project_id
            or not self.subnet_id
            or not self.username
        ):
            raise ValueError(
                "Nebius image, project ID, subnet ID, and username are required"
            )
        if not re.fullmatch(r"[a-z_][a-z0-9_-]{0,31}", self.username):
            raise ValueError("Nebius SSH username is invalid")
        if not re.fullmatch(r"[A-Za-z0-9_./:@-]+", self.image):
            raise ValueError("Nebius image reference is invalid")
        if self.username in {"root", "admin"}:
            raise ValueError("Nebius SSH username cannot be root or admin")
        if self.disk_size_gib < 1 or self.worker_port not in range(1, 65536):
            raise ValueError("Nebius disk size and worker port must be positive")
        if self.startup_timeout_seconds <= 0 or self.poll_interval_seconds <= 0:
            raise ValueError("Nebius timeouts must be positive")


@dataclass(frozen=True, slots=True)
class NebiusInstance:
    instance_id: str
    disk_id: str
    public_ip: str | None = None


class NebiusClient(Protocol):
    def create_instance(self) -> NebiusInstance: ...

    def wait_until_ready(
        self,
        instance: NebiusInstance,
        timeout_seconds: float,
        *,
        progress: Callable[[str, float], None] | None = None,
    ) -> WorkerEndpoint: ...

    def terminate_instance(self, instance_id: str) -> None: ...

    def delete_disk(self, disk_id: str) -> None: ...


class CommandRunner(Protocol):
    def __call__(
        self,
        command: Sequence[str],
        *,
        input: str | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]: ...


class Tunnel(Protocol):
    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...


class NebiusCLIClient:
    """Provision a Nebius VM with the supported CLI and reach it over SSH."""

    def __init__(
        self,
        config: NebiusConfig,
        *,
        runner: CommandRunner | None = None,
        tunnel_factory: Callable[[Sequence[str]], Tunnel] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        port_factory: Callable[[], int] = lambda: _unused_local_port(),
        worker_status: Callable[[str, str], str | None] = lambda address, token: (
            _worker_status(address, token)
        ),
    ) -> None:
        self.config = config
        self._runner = runner or _run_command
        self._tunnel_factory = tunnel_factory or _start_tunnel
        self._monotonic = monotonic
        self._sleep = sleep
        self._port_factory = port_factory
        self._worker_status = worker_status
        self._tunnels: dict[str, Tunnel] = {}
        self._known_hosts: dict[str, Path] = {}

    def create_instance(self) -> NebiusInstance:
        public_key = self.config.ssh_public_key.read_text(encoding="utf-8").strip()
        if "\n" in public_key or not public_key.startswith(
            ("ssh-ed25519 ", "ssh-rsa ", "ecdsa-")
        ):
            raise ValueError("Nebius SSH public-key file is invalid")
        suffix = secrets.token_hex(4)
        disk = self._json_command(
            [
                self.config.nebius_command,
                "compute",
                "disk",
                "create",
                "--name",
                f"{self.config.resource_name_prefix}-disk-{suffix}",
                "--parent-id",
                self.config.project_id,
                "--size-gibibytes",
                str(self.config.disk_size_gib),
                "--type",
                self.config.disk_type,
                "--source-image-family-image-family",
                self.config.image_family,
                "--block-size-bytes",
                "4096",
                "--format",
                "json",
            ]
        )
        disk_id = _metadata_id(disk, "disk")
        cloud_init = (
            "users:\n"
            f"  - name: {self.config.username}\n"
            "    sudo: ALL=(ALL) NOPASSWD:ALL\n"
            "    shell: /bin/bash\n"
            "    ssh_authorized_keys:\n"
            f"      - {public_key}\n"
        )
        instance_name = f"{self.config.resource_name_prefix}-vm-{suffix}"
        request = {
            "metadata": {
                "name": instance_name,
                "parent_id": self.config.project_id,
            },
            "spec": {
                "stopped": False,
                "cloud_init_user_data": cloud_init,
                "resources": {
                    "platform": self.config.platform,
                    "preset": self.config.preset,
                },
                "boot_disk": {
                    "attach_mode": "READ_WRITE",
                    "existing_disk": {"id": disk_id},
                },
                "network_interfaces": [
                    {
                        "name": "kernel-mcts-worker",
                        "subnet_id": self.config.subnet_id,
                        "ip_address": {},
                        "public_ip_address": {},
                    }
                ],
            },
        }
        try:
            instance_value = self._json_command(
                [
                    self.config.nebius_command,
                    "compute",
                    "instance",
                    "create",
                    "--format",
                    "json",
                    "-",
                ],
                input=json.dumps(request),
            )
            instance_id = _metadata_id(instance_value, "instance")
        except Exception as create_error:
            instance_id = self._recover_failed_instance_id(
                create_error, instance_name
            )
            cleanup_errors: list[str] = []
            if instance_id is not None:
                try:
                    self.terminate_instance(instance_id)
                except Exception as error:
                    cleanup_errors.append(f"instance {instance_id}: {error}")
            try:
                self.delete_disk(disk_id)
            except Exception as error:
                cleanup_errors.append(f"disk {disk_id}: {error}")
            if cleanup_errors:
                residuals = "; ".join(cleanup_errors)
                raise NebiusError(
                    f"Nebius instance creation failed ({create_error}); "
                    f"cleanup was incomplete: {residuals}"
                ) from create_error
            raise
        return NebiusInstance(instance_id, disk_id)

    def _recover_failed_instance_id(
        self, error: Exception, instance_name: str
    ) -> str | None:
        if isinstance(error, _NebiusCommandError):
            instance_id = _resource_id_from_json(
                error.result.stdout, expected_name=instance_name
            )
            if instance_id is not None:
                return instance_id
        try:
            value = self._json_command(
                [
                    self.config.nebius_command,
                    "compute",
                    "instance",
                    "list",
                    "--parent-id",
                    self.config.project_id,
                    "--format",
                    "json",
                ]
            )
        except Exception:
            return None
        items = value.get("items", ())
        if not isinstance(items, list):
            return None
        for item in items:
            if not isinstance(item, Mapping):
                continue
            metadata = item.get("metadata")
            if not isinstance(metadata, Mapping):
                continue
            if metadata.get("name") == instance_name:
                resource_id = metadata.get("id")
                if isinstance(resource_id, str) and resource_id:
                    return resource_id
        return None

    def wait_until_ready(
        self,
        instance: NebiusInstance,
        timeout_seconds: float,
        *,
        progress: Callable[[str, float], None] | None = None,
    ) -> WorkerEndpoint:
        started = self._monotonic()
        deadline = started + timeout_seconds
        public_ip = instance.public_ip
        while public_ip is None:
            value = self._json_command(
                [
                    self.config.nebius_command,
                    "compute",
                    "instance",
                    "get",
                    "--id",
                    instance.instance_id,
                    "--format",
                    "json",
                ]
            )
            public_ip = _public_ip(value)
            if progress is not None:
                progress("PROVISIONING", self._monotonic() - started)
            if public_ip is None:
                self._check_deadline(deadline, "public IP")
                self._sleep(self.config.poll_interval_seconds)

        known_hosts = self._known_hosts_file(instance.instance_id)
        ssh_base = self._ssh_base(public_ip, known_hosts)
        while True:
            result = self._runner(
                [*ssh_base, f"{self.config.username}@{public_ip}", "true"],
                timeout=min(15.0, max(1.0, deadline - self._monotonic())),
            )
            if result.returncode == 0:
                break
            if progress is not None:
                progress("SSH", self._monotonic() - started)
            self._check_deadline(deadline, "SSH")
            self._sleep(self.config.poll_interval_seconds)

        token = secrets.token_urlsafe(32)
        remote_env_path = f"/tmp/kernel-mcts-worker-{instance.instance_id}.env"
        worker_environment = "\n".join(
            (
                f"KERNEL_MCTS_WORKER_TOKEN={token}",
                f"KERNEL_MCTS_WORKER_ID={instance.instance_id}",
                "KERNEL_MCTS_PROVIDER=nebius",
                f"KERNEL_MCTS_WORKER_PORT={self.config.worker_port}",
                f"KERNEL_MCTS_CONTAINER_IMAGE={self.config.image}",
                *(
                    (f"KERNEL_MCTS_GIT_COMMIT={self.config.project_git_commit}",)
                    if self.config.project_git_commit is not None
                    else ()
                ),
                *(
                    (
                        "KERNEL_MCTS_DIRTY_TREE="
                        f"{1 if self.config.project_dirty_tree else 0}",
                    )
                    if self.config.project_dirty_tree is not None
                    else ()
                ),
                *(
                    (f"KERNEL_MCTS_CONTAINER_DIGEST={self.config.container_digest}",)
                    if self.config.container_digest is not None
                    else ()
                ),
                "",
            )
        )
        uploaded = self._runner(
            [
                *ssh_base,
                f"{self.config.username}@{public_ip}",
                "sh",
                "-c",
                f"umask 077 && cat > {remote_env_path}",
            ],
            input=worker_environment,
            timeout=max(1.0, deadline - self._monotonic()),
        )
        if uploaded.returncode != 0:
            raise NebiusError("Nebius worker environment upload failed")
        docker_command = [
            "sudo",
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            "kernel-mcts-worker",
            "--gpus",
            "all",
            "--cap-add=SYS_ADMIN",
            "-p",
            f"127.0.0.1:{self.config.worker_port}:{self.config.worker_port}",
            "--env-file",
            remote_env_path,
            self.config.image,
        ]
        try:
            result = self._runner(
                [*ssh_base, f"{self.config.username}@{public_ip}", *docker_command],
                timeout=max(1.0, deadline - self._monotonic()),
            )
        finally:
            self._runner(
                [
                    *ssh_base,
                    f"{self.config.username}@{public_ip}",
                    "rm",
                    "-f",
                    remote_env_path,
                ],
                timeout=min(15.0, max(1.0, deadline - self._monotonic())),
            )
        if result.returncode != 0:
            raise NebiusError("Nebius worker container failed to start")

        local_port = self._port_factory()
        tunnel = self._tunnel_factory(
            [
                *ssh_base,
                "-N",
                "-L",
                f"127.0.0.1:{local_port}:127.0.0.1:{self.config.worker_port}",
                f"{self.config.username}@{public_ip}",
            ]
        )
        self._tunnels[instance.instance_id] = tunnel
        address = f"http://127.0.0.1:{local_port}"
        while True:
            if tunnel.poll() is not None:
                raise NebiusError("Nebius SSH tunnel exited before worker readiness")
            status = self._worker_status(address, token)
            if progress is not None:
                progress(status or "WORKER", self._monotonic() - started)
            if status == "ready":
                return WorkerEndpoint(instance.instance_id, address, token)
            self._check_deadline(deadline, "worker readiness")
            self._sleep(self.config.poll_interval_seconds)

    def terminate_instance(self, instance_id: str) -> None:
        tunnel = self._tunnels.pop(instance_id, None)
        try:
            if tunnel is not None:
                tunnel.terminate()
                try:
                    tunnel.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    tunnel.kill()
                    tunnel.wait(timeout=5.0)
        finally:
            try:
                self._mutation_command(
                    [
                        self.config.nebius_command,
                        "compute",
                        "instance",
                        "delete",
                        "--id",
                        instance_id,
                        "--format",
                        "json",
                    ]
                )
            finally:
                known_hosts = self._known_hosts.pop(instance_id, None)
                if known_hosts is not None:
                    known_hosts.unlink(missing_ok=True)

    def delete_disk(self, disk_id: str) -> None:
        self._mutation_command(
            [
                self.config.nebius_command,
                "compute",
                "disk",
                "delete",
                "--id",
                disk_id,
                "--format",
                "json",
            ]
        )

    def _ssh_base(self, public_ip: str, known_hosts: Path) -> list[str]:
        del public_ip
        return [
            self.config.ssh_command,
            "-i",
            str(self.config.ssh_private_key),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={known_hosts}",
            "-o",
            "GlobalKnownHostsFile=/dev/null",
            "-o",
            "ConnectTimeout=10",
        ]

    def _known_hosts_file(self, instance_id: str) -> Path:
        existing = self._known_hosts.get(instance_id)
        if existing is not None:
            return existing
        descriptor, name = tempfile.mkstemp(prefix="kernel-mcts-nebius-known-hosts-")
        os.close(descriptor)
        path = Path(name)
        path.chmod(0o600)
        self._known_hosts[instance_id] = path
        return path

    def _json_command(
        self, command: Sequence[str], *, input: str | None = None
    ) -> Mapping[str, object]:
        result = self._runner(
            command, input=input, timeout=self.config.startup_timeout_seconds
        )
        if result.returncode != 0:
            raise _NebiusCommandError(_command_error_message(result), result)
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise NebiusError("Nebius CLI returned malformed JSON") from error
        if not isinstance(value, Mapping):
            raise NebiusError("Nebius CLI response must be an object")
        return value

    def _mutation_command(self, command: Sequence[str]) -> None:
        result = self._runner(
            command,
            timeout=self.config.startup_timeout_seconds,
        )
        if result.returncode != 0:
            raise _NebiusCommandError(_command_error_message(result), result)

    def _check_deadline(self, deadline: float, stage: str) -> None:
        if self._monotonic() >= deadline:
            raise TimeoutError(f"Nebius worker was not ready during {stage}")


class NebiusWorker:
    def __init__(
        self,
        instance: NebiusInstance,
        endpoint: WorkerEndpoint,
        transport: WorkerTransport,
        manifest: EnvironmentManifest,
        owner_token: object,
    ) -> None:
        self.instance = instance
        self.endpoint = endpoint
        self._transport = transport
        self._manifest = manifest
        self._owner_token = owner_token
        self._released = False

    @property
    def worker_id(self) -> str:
        return self.endpoint.worker_id

    @property
    def released(self) -> bool:
        return self._released

    def get_environment_manifest(self) -> EnvironmentManifest:
        return self._manifest

    def evaluate(
        self,
        evaluation_id: str,
        program: KernelProgram,
        workload: WorkloadContract,
        profile_level: str,
    ) -> EvaluationResult:
        if self._released:
            raise RuntimeError("cannot evaluate on a released Nebius worker")
        try:
            result = self._transport.evaluate(
                evaluation_id, program, workload, profile_level
            )
        except Exception as error:
            return EvaluationResult(
                status=ProposalStatus.INFRASTRUCTURE_FAILURE,
                worker_id=self.worker_id,
                environment_manifest_id=self._manifest.manifest_id,
                metadata={
                    "evaluation_id": evaluation_id,
                    "error_type": type(error).__name__,
                },
            )
        return replace(
            result,
            worker_id=result.worker_id or self.worker_id,
            environment_manifest_id=result.environment_manifest_id
            or self._manifest.manifest_id,
        )

    def profile(
        self,
        evaluation_id: str,
        profile_level: str,
        metric_set: str = "lightweight_v1",
    ) -> Mapping[str, object]:
        if self._released:
            raise RuntimeError("cannot profile on a released Nebius worker")
        return self._transport.profile(evaluation_id, profile_level, metric_set)


class NebiusProvider:
    def __init__(
        self,
        config: NebiusConfig,
        *,
        client: NebiusClient,
        transport_factory: WorkerTransportFactory,
        readiness_progress: Callable[[str, float], None] | None = None,
    ) -> None:
        self.config = config
        self._client = client
        self._transport_factory = transport_factory
        self._readiness_progress = readiness_progress
        self._owner_token = object()
        self._active: dict[int, NebiusWorker] = {}

    def acquire_worker(self, hardware: HardwareSpec) -> NebiusWorker:
        instance: NebiusInstance | None = None
        transport: WorkerTransport | None = None
        try:
            instance = self._client.create_instance()
            endpoint = self._client.wait_until_ready(
                instance,
                self.config.startup_timeout_seconds,
                progress=self._readiness_progress,
            )
            transport = self._transport_factory(endpoint)
            manifest = transport.get_environment_manifest()
            if (
                manifest.worker_id != instance.instance_id
                or manifest.pod_id != instance.instance_id
            ):
                raise ValueError("Nebius worker manifest identity mismatch")
            if manifest.provider != "nebius":
                raise ValueError("Nebius worker manifest provider mismatch")
            validate_environment(hardware, manifest)
            worker = NebiusWorker(
                instance, endpoint, transport, manifest, self._owner_token
            )
            self._active[id(worker)] = worker
            return worker
        except Exception:
            if transport is not None:
                try:
                    transport.close()
                except Exception:
                    pass
            if instance is not None:
                self._cleanup(instance)
            raise

    def release_worker(self, worker: GPUWorker) -> None:
        if (
            not isinstance(worker, NebiusWorker)
            or worker._owner_token is not self._owner_token
        ):
            raise ValueError("worker is not owned by this Nebius provider")
        if worker.released:
            return
        if self._active.get(id(worker)) is not worker:
            raise ValueError("worker is not active in this Nebius provider")
        error: Exception | None = None
        try:
            worker._transport.close()
        except Exception as caught:
            error = caught
        try:
            self._cleanup(worker.instance)
        except Exception as caught:
            if error is None:
                error = caught
        worker._released = True
        self._active.pop(id(worker), None)
        if error is not None:
            raise error

    @contextmanager
    def worker_for_run(
        self, hardware: HardwareSpec, events: ManifestEventSink | None = None
    ) -> Iterator[NebiusWorker]:
        worker = self.acquire_worker(hardware)
        try:
            if events is not None:
                events.emit(
                    "environment_manifest", worker.get_environment_manifest().as_dict()
                )
            yield worker
        finally:
            self.release_worker(worker)

    def _cleanup(self, instance: NebiusInstance) -> None:
        error: Exception | None = None
        try:
            self._client.terminate_instance(instance.instance_id)
        except Exception as caught:
            error = caught
        try:
            self._client.delete_disk(instance.disk_id)
        except Exception as caught:
            if error is None:
                error = caught
        if error is not None:
            raise error


def create_nebius_provider(
    config: NebiusConfig,
    *,
    readiness_progress: Callable[[str, float], None] | None = None,
) -> NebiusProvider:
    client = NebiusCLIClient(config)
    return NebiusProvider(
        config,
        client=client,
        transport_factory=lambda endpoint: HTTPWorkerTransport(
            endpoint, timeout_seconds=config.startup_timeout_seconds
        ),
        readiness_progress=readiness_progress,
    )


def _run_command(
    command: Sequence[str], *, input: str | None = None, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        input=input,
        timeout=timeout,
        text=True,
        capture_output=True,
        check=False,
    )


def _start_tunnel(command: Sequence[str]) -> Tunnel:
    return subprocess.Popen(
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _unused_local_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _resource_id_from_json(output: str, *, expected_name: str) -> str | None:
    try:
        value = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(value, Mapping):
        return None
    metadata = value.get("metadata")
    name = metadata.get("name") if isinstance(metadata, Mapping) else None
    identifier = metadata.get("id") if isinstance(metadata, Mapping) else None
    if (
        name == expected_name
        and isinstance(identifier, str)
        and re.fullmatch(r"[A-Za-z0-9_-]+", identifier)
    ):
        return identifier
    return None


def _command_error_message(result: subprocess.CompletedProcess[str]) -> str:
    diagnostic = re.sub(r"[\x00-\x1f\x7f]+", " ", result.stderr).strip()
    diagnostic = re.sub(
        r"(?i)\b(api[_ -]?key|authorization|password|secret|token)\b"
        r"\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        diagnostic,
    )
    diagnostic = re.sub(r"\s+", " ", diagnostic)[:2000]
    suffix = f": {diagnostic}" if diagnostic else ""
    return f"Nebius CLI command failed with exit code {result.returncode}{suffix}"


def _metadata_id(value: Mapping[str, object], resource: str) -> str:
    metadata = value.get("metadata")
    identifier = metadata.get("id") if isinstance(metadata, Mapping) else None
    if not isinstance(identifier, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]+", identifier
    ):
        raise NebiusError(f"Nebius returned an invalid {resource} record")
    return identifier


def _public_ip(value: Mapping[str, object]) -> str | None:
    status = value.get("status")
    interfaces = (
        status.get("network_interfaces") if isinstance(status, Mapping) else None
    )
    if not isinstance(interfaces, list) or not interfaces:
        return None
    public = (
        interfaces[0].get("public_ip_address")
        if isinstance(interfaces[0], Mapping)
        else None
    )
    address = public.get("address") if isinstance(public, Mapping) else None
    if not isinstance(address, str) or not address:
        return None
    return address.split("/", 1)[0]


def _worker_status(address: str, token: str) -> str | None:
    request = urllib.request.Request(
        f"{address}/health",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            payload = json.loads(response.read())
    except (OSError, TimeoutError, urllib.error.HTTPError, json.JSONDecodeError):
        return None
    status = payload.get("status") if isinstance(payload, Mapping) else None
    if status == "starting":
        stage = payload.get("stage")
        return str(stage) if isinstance(stage, str) else "starting"
    if status == "failed":
        raise NebiusError("Nebius worker initialization failed")
    return "ready" if status == "ready" else None
