from __future__ import annotations

from typing import Mapping

from .providers import RunPodConfig, RunPodProvider
from .runpod_api import HTTPTransport, RunPodRESTClient
from .worker_protocol import HTTPWorkerTransport, WorkerHTTP


def create_runpod_provider(
    config: RunPodConfig,
    *,
    environ: Mapping[str, str] | None = None,
    runpod_http: HTTPTransport | None = None,
    worker_http: WorkerHTTP | None = None,
) -> RunPodProvider:
    """Wire the concrete RunPod lifecycle and authenticated worker transports."""
    return RunPodProvider(
        config,
        client_factory=lambda api_key: RunPodRESTClient(api_key, http=runpod_http),
        transport_factory=lambda endpoint: HTTPWorkerTransport(
            endpoint,
            timeout_seconds=config.startup_timeout_seconds,
            http=worker_http,
        ),
        environ=environ,
    )
