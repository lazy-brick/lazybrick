from types import SimpleNamespace
import sys
import pytest
from lazybrick.evidence.resources import ProcessTreeResources, process_tree_sample
from lazybrick.evidence.vllm import EvidenceError


def test_resources_capture_raw_process_scope_and_sampled_peaks():
    values = iter([10, 30])
    def read():
        n=next(values)
        return {"pids":[7,8],"sum_rss_bytes":n,"gpu_process_bytes":n*2,"vanished_processes":0}
    with ProcessTreeResources(interval_ms=10000, reader=read) as sampler:
        pass
    result=sampler.record()
    assert result["sampled_peak_sum_rss_bytes"] == 30
    assert result["sampled_peak_gpu_process_bytes"] == 60
    assert len(result["raw_samples"]) == 2
    assert result["scope"] == "allocating_process_tree"
    assert "lower bound" in result["limitations"]


def test_missing_resource_data_fails_closed():
    with pytest.raises(EvidenceError, match="incomplete"):
        with ProcessTreeResources(reader=lambda: {"sum_rss_bytes":0}):
            pytest.fail("must not reach workload")


def test_incomplete_measurement_cannot_be_reported():
    with pytest.raises(EvidenceError, match="did not complete"):
        ProcessTreeResources().record()


def test_actual_reader_aggregates_only_current_process_tree(monkeypatch):
    class Process:
        def __init__(self, pid): self.pid=pid
        def children(self, recursive): return [Process(8)]
        def memory_info(self): return SimpleNamespace(rss=10 if self.pid==7 else 20)
    monkeypatch.setitem(sys.modules,"psutil",SimpleNamespace(Process=Process,NoSuchProcess=LookupError))
    monkeypatch.setattr("lazybrick.evidence.resources.os.getpid",lambda:7)
    monkeypatch.setattr("lazybrick.evidence.resources.subprocess.run",lambda *a,**kw:SimpleNamespace(returncode=0,stdout="7, 2\n8, 3\n999, 100\n"))
    sample=process_tree_sample()
    assert sample["sum_rss_bytes"] == 30
    assert sample["gpu_process_bytes"] == 5*1024*1024
    assert sample["pids"] == [7,8]


def test_unavailable_gpu_process_memory_is_not_zero(monkeypatch):
    class Process:
        pid=7
        def __init__(self,pid): pass
        def children(self,recursive): return []
        def memory_info(self): return SimpleNamespace(rss=10)
    monkeypatch.setitem(sys.modules,"psutil",SimpleNamespace(Process=Process,NoSuchProcess=LookupError))
    monkeypatch.setattr("lazybrick.evidence.resources.subprocess.run",lambda *a,**kw:SimpleNamespace(returncode=0,stdout="7, [N/A]\n"))
    with pytest.raises(EvidenceError,match="unavailable"):
        process_tree_sample()
