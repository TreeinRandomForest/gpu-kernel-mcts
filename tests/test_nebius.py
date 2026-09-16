from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from kernel_mcts.domain import (
    EvaluationResult,
    KernelProgram,
    ProposalStatus,
    ShapeCase,
    WorkloadContract,
)
from kernel_mcts.nebius import (
    NebiusCLIClient,
    NebiusConfig,
    NebiusError,
    NebiusInstance,
    NebiusProvider,
)
from kernel_mcts.providers import EnvironmentManifest, HardwareSpec, WorkerEndpoint


WORKLOAD = WorkloadContract("toy", "toy", "fp32", (ShapeCase({"n": 1}, 1.0),), 0.0, 0.0)


def manifest(**changes) -> EnvironmentManifest:
    values = {
        "worker_id": "instance-1",
        "provider": "nebius",
        "pod_id": "instance-1",
        "gpu_model": "NVIDIA H100 80GB HBM3",
        "gpu_count": 1,
        "compute_capability": "9.0",
        "form_factor": "SXM",
        "captured_at": "2026-09-14T00:00:00+00:00",
        "profiler_versions": {"ncu": "2025.1"},
    }
    values.update(changes)
    return EnvironmentManifest(**values)


def config(tmp_path: Path, **changes) -> NebiusConfig:
    public = tmp_path / "id_ed25519.pub"
    private = tmp_path / "id_ed25519"
    public.write_text("ssh-ed25519 AAAAtest user@test\n")
    private.write_text("private")
    values = {
        "image": "registry/worker:v1",
        "project_id": "project-1",
        "subnet_id": "subnet-1",
        "ssh_public_key": public,
        "ssh_private_key": private,
        "username": "sanjay",
    }
    values.update(changes)
    return NebiusConfig(**values)


class FakeNebiusClient:
    def __init__(self, startup_error: Exception | None = None) -> None:
        self.startup_error = startup_error
        self.created = 0
        self.terminated = []
        self.deleted_disks = []

    def create_instance(self):
        self.created += 1
        return NebiusInstance("instance-1", "disk-1", "203.0.113.2")

    def wait_until_ready(self, instance, timeout_seconds, *, progress=None):
        if self.startup_error:
            raise self.startup_error
        if progress:
            progress("ready", 1.0)
        return WorkerEndpoint("instance-1", "http://127.0.0.1:12345", "token")

    def terminate_instance(self, instance_id):
        self.terminated.append(instance_id)

    def delete_disk(self, disk_id):
        self.deleted_disks.append(disk_id)


class FakeTransport:
    def __init__(self, observed=None) -> None:
        self.observed = observed or manifest()
        self.evaluations = []
        self.profiles = []
        self.closed = 0

    def get_environment_manifest(self):
        return self.observed

    def evaluate(self, evaluation_id, program, workload, profile_level):
        self.evaluations.append(evaluation_id)
        return EvaluationResult(ProposalStatus.VALID, program, "key", 1.0)

    def profile(self, evaluation_id, profile_level):
        self.profiles.append(evaluation_id)
        return {"profiler": "ncu"}

    def close(self):
        self.closed += 1


def test_provider_reuses_one_vm_and_deletes_vm_then_disk(tmp_path) -> None:
    client = FakeNebiusClient()
    transport = FakeTransport()
    progress = []
    provider = NebiusProvider(
        config(tmp_path),
        client=client,
        transport_factory=lambda endpoint: transport,
        readiness_progress=lambda status, elapsed: progress.append((status, elapsed)),
    )

    worker = provider.acquire_worker(
        HardwareSpec("H100", form_factor="SXM", required_profilers=("ncu",))
    )
    first = worker.evaluate("one", KernelProgram("one"), WORKLOAD, "tier0")
    second = worker.evaluate("two", KernelProgram("two"), WORKLOAD, "tier0")
    worker.profile("one", "lightweight")
    provider.release_worker(worker)
    provider.release_worker(worker)

    assert client.created == 1
    assert transport.evaluations == ["one", "two"]
    assert transport.profiles == ["one"]
    assert first.worker_id == second.worker_id == "instance-1"
    assert progress == [("ready", 1.0)]
    assert transport.closed == 1
    assert client.terminated == ["instance-1"]
    assert client.deleted_disks == ["disk-1"]
    assert worker.released


def test_provider_cleans_up_startup_and_manifest_failures(tmp_path) -> None:
    startup_client = FakeNebiusClient(RuntimeError("startup failed"))
    provider = NebiusProvider(
        config(tmp_path),
        client=startup_client,
        transport_factory=lambda endpoint: pytest.fail("no transport"),
    )
    with pytest.raises(RuntimeError, match="startup failed"):
        provider.acquire_worker(HardwareSpec("H100"))
    assert startup_client.terminated == ["instance-1"]
    assert startup_client.deleted_disks == ["disk-1"]

    client = FakeNebiusClient()
    transport = FakeTransport(manifest(form_factor="PCIe"))
    provider = NebiusProvider(
        config(tmp_path), client=client, transport_factory=lambda endpoint: transport
    )
    with pytest.raises(ValueError, match="form factor"):
        provider.acquire_worker(HardwareSpec("H100", form_factor="SXM"))
    assert transport.closed == 1
    assert client.terminated == ["instance-1"]
    assert client.deleted_disks == ["disk-1"]


class RecordingRunner:
    def __init__(self) -> None:
        self.calls = []
        self.disk_created = False
        self.instance_created = False

    def __call__(self, command, *, input=None, timeout=None):
        command = list(command)
        self.calls.append((command, input, timeout))
        if command[1:4] == ["compute", "disk", "create"]:
            self.disk_created = True
            output = {"metadata": {"id": "disk-1"}}
        elif command[1:4] == ["compute", "instance", "create"]:
            self.instance_created = True
            output = {"metadata": {"id": "instance-1"}}
        else:
            output = {}
        return subprocess.CompletedProcess(command, 0, json.dumps(output), "")


def test_cli_client_creates_disk_and_sxm_vm_without_credentials_in_request(
    tmp_path,
) -> None:
    runner = RecordingRunner()
    client = NebiusCLIClient(config(tmp_path), runner=runner)

    instance = client.create_instance()

    assert instance == NebiusInstance("instance-1", "disk-1")
    request = json.loads(
        next(
            value
            for command, value, _ in runner.calls
            if command[2:4] == ["instance", "create"]
        )
    )
    assert request["spec"]["resources"] == {
        "platform": "gpu-h100-sxm",
        "preset": "1gpu-16vcpu-200gb",
    }
    assert request["spec"]["network_interfaces"][0]["subnet_id"] == "subnet-1"
    assert request["metadata"]["parent_id"] == "project-1"
    disk_command = next(
        command
        for command, _, _ in runner.calls
        if command[1:4] == ["compute", "disk", "create"]
    )
    assert disk_command[disk_command.index("--parent-id") + 1] == "project-1"
    serialized = json.dumps(request)
    assert "RUNPOD_API_KEY" not in serialized
    assert "KERNEL_MCTS_WORKER_TOKEN" not in serialized


def test_cli_client_deletes_disk_if_instance_creation_fails(tmp_path) -> None:
    class FailedRunner(RecordingRunner):
        def __call__(self, command, *, input=None, timeout=None):
            if list(command)[1:4] == ["compute", "instance", "create"]:
                return subprocess.CompletedProcess(
                    command, 1, "", "failure with details"
                )
            return super().__call__(command, input=input, timeout=timeout)

    runner = FailedRunner()
    client = NebiusCLIClient(config(tmp_path), runner=runner)

    with pytest.raises(NebiusError, match="CLI command failed"):
        client.create_instance()

    assert any(
        command[1:4] == ["compute", "disk", "delete"] for command, _, _ in runner.calls
    )


def test_cli_client_starts_privileged_worker_behind_ssh_tunnel(tmp_path) -> None:
    class Tunnel:
        def __init__(self):
            self.terminated = 0

        def poll(self):
            return None

        def terminate(self):
            self.terminated += 1

        def wait(self, timeout=None):
            return 0

        def kill(self):
            raise AssertionError("tunnel should terminate promptly")

    runner = RecordingRunner()
    tunnel = Tunnel()
    client = NebiusCLIClient(
        config(tmp_path),
        runner=runner,
        tunnel_factory=lambda command: tunnel,
        port_factory=lambda: 43210,
        worker_status=lambda address, token: "ready",
    )
    instance = NebiusInstance("instance-1", "disk-1", "203.0.113.2")

    endpoint = client.wait_until_ready(instance, 30.0)

    assert endpoint.worker_id == "instance-1"
    assert endpoint.address == "http://127.0.0.1:43210"
    commands = [command for command, _, _ in runner.calls]
    ssh_commands = [command for command in commands if command[0] == "ssh"]
    known_hosts_options = {
        item
        for command in ssh_commands
        for item in command
        if item.startswith("UserKnownHostsFile=")
    }
    assert len(known_hosts_options) == 1
    known_hosts = Path(known_hosts_options.pop().partition("=")[2])
    assert known_hosts.is_file()
    assert known_hosts != config(tmp_path).ssh_private_key.parent / "known_hosts"
    assert all("GlobalKnownHostsFile=/dev/null" in command for command in ssh_commands)
    assert all("StrictHostKeyChecking=accept-new" in command for command in ssh_commands)
    docker = next(command for command in commands if "docker" in command)
    assert "--gpus" in docker
    assert "--cap-add=SYS_ADMIN" in docker
    assert "--env-file" in docker
    assert not any("KERNEL_MCTS_WORKER_TOKEN=" in item for item in docker)
    upload = next(value for command, value, _ in runner.calls if "cat >" in command[-1])
    assert upload is not None and "KERNEL_MCTS_WORKER_TOKEN=" in upload
    assert any(command[-3:-1] == ["rm", "-f"] for command in commands)

    client.terminate_instance("instance-1")
    assert tunnel.terminated == 1
    assert not known_hosts.exists()
    assert any(
        command[1:4] == ["compute", "instance", "delete"]
        for command, _, _ in runner.calls
    )


@pytest.mark.parametrize("username", ["root", "admin"])
def test_config_rejects_reserved_ssh_users(tmp_path, username) -> None:
    with pytest.raises(ValueError, match="cannot be root or admin"):
        config(tmp_path, username=username)
