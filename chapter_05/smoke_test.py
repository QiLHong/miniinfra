"""不下载权重、不要求 GPU 的 Paged KV Cache 快速自检。"""

import torch

from cache_engine import make_request_specs, run_dense_cache, run_paged_cache
from paged_cache import PagedKVCache
from qwen3_model import Qwen3Config, Qwen3ForCausalLM


def build_tiny_model():
    """用缩小的词表、隐藏维度和层数创建随机权重模型，便于在 CPU 上快速验证算法。"""
    config = Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        torch_dtype="float32",
        tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(config).eval()


def check_block_pool(model, device):
    """验证分页存取、跨块追加、释放与块编号复用，确保逻辑顺序不依赖物理连续性。"""
    cache = PagedKVCache(
        model.config, block_size=4, max_blocks=8,
        device=device, dtype=torch.float32,
    )
    request_cache = []
    expected = []
    for layer_index in range(model.config.num_hidden_layers):
        key = torch.arange(2 * 6 * 8, dtype=torch.float32).view(1, 2, 6, 8)
        key = key + layer_index * 1000
        value = key + 500
        request_cache.append((key, value))
        expected.append((key, value))
    cache.store_prefill("a", request_cache)
    assert len(cache.block_tables["a"]) == 2
    assert cache.allocated_token_slots == 8
    assert cache.live_token_count == 6
    for layer_index in range(model.config.num_hidden_layers):
        key, value = cache.read_layer(layer_index, ["a"], 6)
        assert torch.equal(key, expected[layer_index][0])
        assert torch.equal(value, expected[layer_index][1])

    old_ids = list(cache.block_tables["a"])
    positions = cache.prepare_append(["a"])
    assert positions == {"a": 6}
    for layer_index in range(model.config.num_hidden_layers):
        token = torch.full((2, 1, 8), 7000.0 + layer_index)
        cache.write_token(layer_index, "a", token, token + 1, 6)
    cache.finish_append(["a"])
    assert cache.sequence_lengths["a"] == 7
    cache.release("a")

    short_cache = [
        (key[:, :, :2, :], value[:, :, :2, :])
        for key, value in expected
    ]
    cache.store_prefill("b", short_cache)
    assert cache.block_tables["b"][0] == min(old_ids)
    assert cache.reuse_count == 1


def main():
    """运行本章的小规模 CPU 冒烟检查，验证核心数据流、状态或输出一致性。

    使用本地微型模型或测试替身，不下载真实模型；通过这些检查不等于 GPU 性能验证。"""
    torch.manual_seed(7)
    device = torch.device("cpu")
    model = build_tiny_model()
    check_block_pool(model, device)

    sequences = [
        [3, 5, 7],
        [11, 13, 17, 19, 23, 29],
        [31, 37],
        [41, 43, 47, 53, 59],
    ]
    budgets = [5, 2, 4, 3]
    arrivals = [0.0, 0.0, 0.1, 0.2]
    specs = make_request_specs(sequences, budgets, arrivals)
    dense = run_dense_cache(
        model, specs, 2, -1, device, stop_on_eos=False
    )
    paged = run_paged_cache(
        model, specs, 2, -1, device, block_size=4, stop_on_eos=False
    )
    assert dense["new_token_ids"] == paged["new_token_ids"]
    assert paged["metrics"]["block_reuse_count"] > 0
    assert paged["metrics"]["peak_used_blocks"] > 0
    assert dense["metrics"]["visited_kv_token_slots"] > 0
    assert paged["metrics"]["visited_kv_token_slots"] > 0
    admissions = [
        request_id
        for event in paged["events"]
        for request_id in event["admitted"]
    ]
    assert admissions == ["0", "1", "2", "3"]

    print("Smoke test 通过")
    print("Block 跨界写入、读取、释放和复用检查通过")
    print("Dense 与 Paged Continuous Batching 输出一致")


if __name__ == "__main__":
    main()
