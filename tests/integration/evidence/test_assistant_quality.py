from types import SimpleNamespace
import pytest
from lazybrick.evidence.assistant_quality import assistant_samples, measure_assistant_loss, compare_assistant_quality
from lazybrick.evidence.vllm import EvidenceError, create_vllm_engine


class Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is True and enable_thinking is False
        return [10, 11, 12] if add_generation_prompt else [10, 11, 12, 20, 21]


def samples():
    return assistant_samples([[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]], Tokenizer(), max_model_len=20)


class Engine:
    def __init__(self, logprob=-2.0, ids=None):
        self.logprob = logprob
        self.ids = ids
    def generate(self, prompts, sampling):
        assert prompts == [{"prompt_token_ids": [10, 11, 12, 20, 21]}]
        return [SimpleNamespace(prompt_token_ids=self.ids or [10,11,12,20,21],
            prompt_logprobs=[None, {11: -100.0}, {12: -100.0}, {20: self.logprob}, {21: -3.0}])]


def measure(engine=None):
    return measure_assistant_loss(engine or Engine(), samples(), seed=42, sampling_factory=lambda **kw: kw)


def test_only_continuation_tokens_are_scored():
    result = measure()
    assert result.loss.token_count == 2
    assert result.loss.total_negative_log_likelihood == 5
    assert result.samples[0]["score_start"] == 3
    assert result.samples[0]["selected_logprobs"] == ["-2.0", "-3.0"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True, 0.5])
def test_invalid_selected_logprobs_rejected(value):
    with pytest.raises(EvidenceError, match="finite and nonpositive"):
        measure(Engine(value))


def test_tokenization_changes_fail_closed():
    with pytest.raises(EvidenceError, match="tokenization"):
        measure(Engine(ids=[10,11,99,20,21]))


def test_template_boundary_mismatch_rejected():
    class Ambiguous(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return [10,11,99] if kwargs["add_generation_prompt"] else [10,11,12,20,21]
    with pytest.raises(EvidenceError, match="ambiguous"):
        assistant_samples([[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]], Ambiguous(), max_model_len=20)


def test_context_truncation_forbidden():
    with pytest.raises(EvidenceError, match="truncation"):
        assistant_samples([[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]], Tokenizer(), max_model_len=5)


def test_equal_counts_do_not_make_different_samples_comparable():
    a = measure().to_dict()
    b = dict(a, sample_digest="0"*64)
    with pytest.raises(EvidenceError, match="samples differ"):
        compare_assistant_quality(a,b)


def test_protocol_and_raw_samples_are_explicit():
    a, b = measure().to_dict(), measure(Engine(-4.0)).to_dict()
    result = compare_assistant_quality(a,b)
    assert result["absolute_delta"] == 1
    assert result["protocol"]["id"].endswith("v2")
    assert result["baseline"]["raw_samples"]


def test_runtime_configuration_is_passed_exactly(tmp_path):
    captured = {}
    create_vllm_engine(tmp_path, seed=42, runtime={"dtype":"bfloat16", "max_model_len":2048,"gpu_memory_utilization":"0.85"}, llm_factory=lambda **kw: captured.update(kw))
    assert captured["dtype"] == "bfloat16"
    assert captured["max_model_len"] == 2048
    assert captured["gpu_memory_utilization"] == .85
    assert captured["seed"] == 42
