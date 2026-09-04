"""Sample allocating process trees; never label orchestrator peaks as model peaks."""
from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Callable

from lazybrick.evidence.vllm import EvidenceError


def process_tree_sample() -> dict[str, object]:
    import psutil  # Optional; already present in both smoke environment locks.
    root = psutil.Process(os.getpid())
    processes = [root, *root.children(recursive=True)]
    pids = set()
    host = 0
    vanished = 0
    for process in processes:
        try:
            host += process.memory_info().rss
            pids.add(process.pid)
        except psutil.NoSuchProcess:
            vanished += 1
    completed = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits"], text=True, capture_output=True, check=False, timeout=5)
    if completed.returncode:
        raise EvidenceError("GPU process memory sampling failed")
    gpu = 0
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        try:
            pid, memory_mib = (part.strip() for part in line.split(","))
            if int(pid) in pids:
                memory = int(memory_mib)
                if memory < 0:
                    raise ValueError()
                gpu += memory * 1024 * 1024
        except ValueError as error:
            raise EvidenceError("GPU memory sample is unavailable or malformed") from error
    return {"pids": sorted(pids), "sum_rss_bytes": host,
            "gpu_process_bytes": gpu, "vanished_processes": vanished}


class ProcessTreeResources:
    def __init__(self, *, interval_ms: int = 100,
                 reader: Callable[[], dict[str, object]] = process_tree_sample) -> None:
        if type(interval_ms) is not int or interval_ms < 10:
            raise EvidenceError("resource interval must be at least 10 ms")
        self.interval_ms = interval_ms
        self.reader = reader
        self.samples: list[dict[str, object]] = []
        self.error: Exception | None = None
        self.stop = threading.Event()
        self.finished = False

    def _sample(self) -> None:
        sample = self.reader()
        for key in ("sum_rss_bytes", "gpu_process_bytes", "vanished_processes"):
            if type(sample.get(key)) is not int or sample[key] < 0:
                raise EvidenceError("resource sample is incomplete")
        if not isinstance(sample.get("pids"), list) or not sample["pids"]:
            raise EvidenceError("resource sample has no observed process")
        self.samples.append({**sample, "elapsed_ns": time.monotonic_ns()-self.started})

    def _poll(self) -> None:
        while not self.stop.wait(self.interval_ms/1000):
            try:
                self._sample()
            except Exception as error:
                self.error = error
                return

    def __enter__(self):
        self.started = time.monotonic_ns()
        self._sample()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        self.thread.join(timeout=6)
        if self.thread.is_alive():
            raise EvidenceError("resource sampler did not terminate")
        if exc_type is not None:
            return False
        if self.error is not None:
            raise EvidenceError("resource sampling failed") from self.error
        self._sample()
        self.finished = True
        return False

    def record(self) -> dict[str, object]:
        if not self.finished or self.error or len(self.samples) < 2:
            raise EvidenceError("resource measurement did not complete")
        return {"schema_version": "0.1", "scope": "allocating_process_tree",
                "method": "psutil sum RSS and nvidia-smi per-process GPU memory",
                "interval_ms": self.interval_ms,
                "sampled_peak_sum_rss_bytes": max(s["sum_rss_bytes"] for s in self.samples),
                "sampled_peak_gpu_process_bytes": max(s["gpu_process_bytes"] for s in self.samples),
                "raw_samples": self.samples,
                "limitations": "sampled lower bound; sum RSS may double-count shared pages; short-lived processes and between-sample peaks may be missed"}
