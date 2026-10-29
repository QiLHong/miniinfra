"""使用基础 PyTorch 组件实现 Qwen3-0.6B。

本文件故意不使用 Transformers 中的模型实现，也不加入 KV Cache、
FlashAttention 等优化，目的是清楚展示一次完整的模型前向计算。
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import nn


# 固定模型版本可以避免上游权重更新后，实验结果悄悄发生变化。
DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


@dataclass
class Qwen3Config:
    """保存模型结构参数；数值来自权重目录中的 config.json。

    注意区分隐藏维度、Query 头数、KV 头数和每头维度。GQA 允许多个 Query 头
    共享一组 K/V，因此 KV 头数通常少于 Query 头数。"""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    torch_dtype: str
    tie_word_embeddings: bool

    @classmethod
    def from_model_directory(cls, model_directory):
        """从 config.json 提取本章实际使用的字段，并在返回配置前检查支持范围。

        配置决定权重和中间张量的形状，不能只修改头数而继续使用不匹配的权重。"""
        config_path = Path(model_directory) / "config.json"
        with config_path.open("r", encoding="utf-8") as file:
            raw_config = json.load(file)

        config = cls(
            vocab_size=raw_config["vocab_size"],
            hidden_size=raw_config["hidden_size"],
            intermediate_size=raw_config["intermediate_size"],
            num_hidden_layers=raw_config["num_hidden_layers"],
            num_attention_heads=raw_config["num_attention_heads"],
            num_key_value_heads=raw_config["num_key_value_heads"],
            head_dim=raw_config["head_dim"],
            rms_norm_eps=raw_config["rms_norm_eps"],
            rope_theta=raw_config["rope_theta"],
            torch_dtype=raw_config.get("torch_dtype", "bfloat16"),
            tie_word_embeddings=raw_config.get("tie_word_embeddings", True),
        )
        config.validate()
        return config

    def validate(self):
        """检查本章模型支持的配置约束，提前拒绝不兼容的头数或权重共享设置。"""

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("Attention 头数必须能被 KV 头数整除")
        if not self.tie_word_embeddings:
            raise ValueError("本期实现只支持共享 Embedding 和 LM Head 权重的模型")


class RMSNorm(nn.Module):
    """沿最后一维做均方根归一化，再乘可学习的逐维缩放系数。

    这里不减均值：计算 x / sqrt(mean(x²) + eps)。eps 用来避免零向量除零；
    统计过程转为 float32，随后将归一化结果转回输入精度。"""

    def __init__(self, hidden_size, eps):
        """创建长度为 hidden_size 的缩放参数，并保存数值稳定项 eps。"""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        # 平方和求均值使用 float32，防止 bfloat16 精度不足。
        """保持输入形状不变，只对最后一维归一化。

        先以 float32 求平方均值，再用 rsqrt 计算倒数平方根；keepdim=True
        保留末尾单维度，便于把每个 token 的缩放因子广播回它的全部特征。"""
        input_dtype = hidden_states.dtype
        float_states = hidden_states.float()
        # 这里只沿特征维求均方值；保留末尾维度后，可将每个 token 的系数广播回全部特征。
        variance = float_states.pow(2).mean(dim=-1, keepdim=True)
        normalized = float_states * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """按 token 位置生成 RoPE 的余弦和正弦系数，供 Q/K 旋转使用。"""

    def __init__(self, head_dim, rope_theta):
        """建立每对特征使用的逆频率；注册为 buffer，使其随模型移动但不参与训练。"""
        super().__init__()

        # 每两个维度共享一个旋转频率。inv_freq 不参与训练，所以注册为 buffer。
        dimensions = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (rope_theta ** (dimensions / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids, output_dtype):
        # 旋转角度 freqs 的形状：[批次, 序列长度, 每头维度/2]。
        """将 [批次, 长度] 的位置编号转换为 [批次, 长度, 每头维度] 的 cos/sin。

        位置乘逆频率得到角度。复制两份频率对应 Qwen3 将向量前后两半配对的布局，
        并不是为相邻两维交替排列的另一种 RoPE 布局生成系数。"""
        # 位置表补一个末维，与每对特征的频率相乘，得到 [批次, 长度, 每头维度/2] 的旋转角度。
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq
        # Qwen3 将同一组频率分别应用到向量的前半部分和后半部分。
        embedding = torch.cat((freqs, freqs), dim=-1)
        return embedding.cos().to(output_dtype), embedding.sin().to(output_dtype)


def rotate_half(hidden_states):
    """把最后一维拆为两半，返回 [-后半, 前半]；输出形状与输入相同。

    它与 cos/sin 相乘后构成二维旋转，必须和模型使用的 RoPE 配对布局一致。"""

    half = hidden_states.shape[-1] // 2
    first_half = hidden_states[..., :half]
    second_half = hidden_states[..., half:]
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rotary_embedding(query, key, cosine, sine):
    """将位置旋转应用到 Q 和 K，V 不参与旋转。

    Q/K 为 [批次, 头数, 长度, 每头维度]；cos/sin 在头数位置补一个单维度，
    于是同一 token 的位置系数可以广播到全部头。"""

    # Query/Key 多一个 attention head 维度，需要在这里补一个维度用于广播。
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    rotated_query = query * cosine + rotate_half(query) * sine
    rotated_key = key * cosine + rotate_half(key) * sine
    return rotated_query, rotated_key


def repeat_key_value(hidden_states, repeat_count):
    """沿头数维度重复 K/V，使它们能与 Query 的头数对应。

    输入为 [批次, KV头数, 长度, 每头维度]，repeat_count 是 Query/KV 头数之比。
    这是直观的 GQA 展开实现，会产生展开后的张量；缓存本身仍应保存较少的 KV 头。"""

    if repeat_count == 1:
        return hidden_states
    # 这里选择直观的 repeat_interleave；Paged Attention 等优化留到后续章节。
    return hidden_states.repeat_interleave(repeat_count, dim=1)


class Qwen3Attention(nn.Module):
    """实现 Qwen3 的分组查询注意力：Q/K 归一化、位置旋转和按头注意力计算。

    阅读 forward 时依次跟踪投影后的头维度、历史 K/V 的处理、Mask 与输出合并；
    不同章节的缓存接口不同，以本文件的 forward 参数为准。"""

    def __init__(self, config):
        """按 Query 与 KV 的头数分别创建投影层，并为每个 Q/K 头创建 RMSNorm。

        q_proj 的输出宽度是 Query头数×每头维度，k_proj/v_proj 则使用 KV头数；
        o_proj 将所有 Query 头的结果合并回模型隐藏维度。"""
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )

        query_size = config.num_attention_heads * config.head_dim
        key_value_size = config.num_key_value_heads * config.head_dim

        self.q_proj = nn.Linear(config.hidden_size, query_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, key_value_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, key_value_size, bias=False)
        self.o_proj = nn.Linear(query_size, config.hidden_size, bias=False)

        # Qwen3 会在每个 attention head 内单独归一化 Query 和 Key。
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(self, hidden_states, cosine, sine, causal_mask):
        """将 [批次, 当前长度, 隐藏维度] 投影为分头 Q/K/V，计算注意力后合并回隐藏维度。

        本章没有 KV Cache，每次使用传入序列的全部 K/V。缩放 QKᵀ 后加因果 Mask，
        用 float32 Softmax 求概率，再与 V 相乘，最后通过 o_proj 合并各头结果。"""
        batch_size, sequence_length, _ = hidden_states.shape

        # 投影后拆分 head，形状变为 [batch, heads, sequence, head_dim]。
        query = self.q_proj(hidden_states)
        query = query.view(
            batch_size, sequence_length, self.num_attention_heads, self.head_dim
        )
        query = self.q_norm(query).transpose(1, 2)

        key = self.k_proj(hidden_states)
        key = key.view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        )
        key = self.k_norm(key).transpose(1, 2)

        value = self.v_proj(hidden_states)
        value = value.view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        )
        value = value.transpose(1, 2)

        query, key = apply_rotary_embedding(query, key, cosine, sine)
        key = repeat_key_value(key, self.num_key_value_groups)
        value = repeat_key_value(value, self.num_key_value_groups)

        # causal_mask 保证当前位置只能看到自己和之前的 Token。
        attention_scores = torch.matmul(query, key.transpose(2, 3))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        attention_scores = attention_scores + causal_mask

        # Softmax 使用 float32 计算，再转回模型 dtype，与参考实现保持一致。
        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32)
        attention_probs = attention_probs.to(query.dtype)
        attention_output = torch.matmul(attention_probs, value)

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, sequence_length, -1)
        return self.o_proj(attention_output)


class Qwen3MLP(nn.Module):
    """用 SwiGLU 门控前馈网络变换每个 token 的特征，输入输出隐藏维度相同。"""

    def __init__(self, config):
        """创建 gate/up 两个升维投影和 down 降维投影；中间维度来自模型配置。"""
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, hidden_states):
        """计算 down_proj(silu(gate_proj(x)) * up_proj(x))。

        乘号是逐元素相乘：一条分支生成门值，另一条分支提供内容；这里只变换特征，
        不在不同 token 之间交换信息。"""
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


class Qwen3DecoderLayer(nn.Module):
    """串联 Attention 与 MLP 两个子层，每个子层都采用归一化后计算、再加残差的结构。"""

    def __init__(self, config):
        """构造本层的注意力、前馈网络及两套独立的 Pre-Norm 参数。"""
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(self, hidden_states, cosine, sine, causal_mask):
        # Attention 子层：Pre-Norm -> Attention -> Residual。
        """依次执行 Attention 和 MLP 两个残差子层。

        每个子层先归一化输入、计算变换，再加回未经该子层变换的原隐藏状态；
        本章处理完整输入序列，尚未引入 KV Cache。"""
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(
            normalized, cosine, sine, causal_mask
        )

        # MLP 子层：Pre-Norm -> MLP -> Residual。
        residual = hidden_states
        normalized = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(normalized)
        return hidden_states


class Qwen3Model(nn.Module):
    """由 token 嵌入、Decoder 层堆叠和末尾 RMSNorm 组成的模型主干。

    它输出隐藏状态；转换为词表上的 logits 由外层 Qwen3ForCausalLM 完成。"""

    def __init__(self, config):
        """按配置建立嵌入表、各层 Decoder、最终归一化和全模型共用的 RoPE 系数生成器。"""
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)

    def make_causal_mask(self, sequence_length, dtype, device):
        """构造 [1, 1, Q长度, KV长度] 的加法 Mask，未来位置填极小值，允许位置为 0。

        主对角线及其左侧可见，上三角被屏蔽；前两个单维度用于广播到批次与各头。"""

        smallest_value = torch.finfo(dtype).min
        mask = torch.full(
            (sequence_length, sequence_length),
            smallest_value,
            dtype=dtype,
            device=device,
        )
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids):
        """把 token ID 嵌入为向量，准备位置与注意力 Mask，然后依次执行各层并做最终归一化。

        有缓存接口的章节按层传递历史并收集 present；缓存只保存 K/V，不保存 logits。"""
        hidden_states = self.embed_tokens(input_ids)
        batch_size, sequence_length = input_ids.shape

        position_ids = torch.arange(sequence_length, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        cosine, sine = self.rotary_emb(position_ids, hidden_states.dtype)
        causal_mask = self.make_causal_mask(
            sequence_length, hidden_states.dtype, hidden_states.device
        )

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states, cosine, sine, causal_mask
            )

        return self.norm(hidden_states)


class Qwen3ForCausalLM(nn.Module):
    """在主干网络后接词表投影，将每个位置的隐藏状态转换为下一 token 的 logits。

    logits 是未归一化分数；推理循环再选择 argmax 或其他采样方法。"""

    def __init__(self, config):
        """建立模型主干和词表输出层，然后让输出层与输入嵌入表共享权重。"""
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_weights()

    def tie_weights(self):
        # Qwen3-0.6B 的输入 Embedding 与输出 LM Head 使用同一份权重。
        """让 lm_head.weight 引用同一个嵌入参数，避免保存两份相同的词表权重。"""
        # 绑定同一个 Parameter 对象，实现权重共享，而不是复制一份数值相同的张量。
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids):
        """执行主干并将隐藏状态映射为 [批次, 当前长度, 词表大小] 的 logits。

        生成循环通常只取最后位置的分数来选择下一 token。"""
        hidden_states = self.model(input_ids)
        return self.lm_head(hidden_states)


def resolve_model_directory(model_name_or_path, revision=DEFAULT_MODEL_REVISION):
    """本地目录存在时直接使用；否则按固定 revision 下载所需的模型与分词文件。

    固定版本使不同实验使用相同权重，下载耗时应与推理计时分开。"""

    local_path = Path(model_name_or_path).expanduser()
    if local_path.is_dir():
        return str(local_path.resolve())

    return snapshot_download(
        repo_id=model_name_or_path,
        revision=revision,
        # 测试服务器到 Hugging Face 的连接偶尔会被重置。降低并发数可以
        # 提高首次下载的稳定性，且不会影响模型加载和推理性能。
        max_workers=2,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "merges.txt",
            "vocab.json",
        ],
    )


def model_dtype_from_config(config):
    """将配置中的精度名称映射到 torch.dtype；不支持的名称显式报错。"""

    supported_dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if config.torch_dtype not in supported_dtypes:
        raise ValueError("不支持的模型 dtype: " + config.torch_dtype)
    return supported_dtypes[config.torch_dtype]


def load_handwritten_model(model_directory, device):
    """读取配置和 Safetensors 权重，核对参数名称后返回 eval 模式的模型。

    共享的 lm_head 权重允许不单独出现在文件中，加载后重新绑定。设备与精度
    转换还涉及 RoPE buffer，因此要关注这里对 inv_freq 的单独处理。"""

    config = Qwen3Config.from_model_directory(model_directory)
    model = Qwen3ForCausalLM(config)

    weight_path = Path(model_directory) / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError("没有找到模型权重: " + str(weight_path))

    state_dict = load_file(str(weight_path), device="cpu")
    load_result = model.load_state_dict(state_dict, strict=False)

    # 共享权重只在 safetensors 中保存一份，所以 lm_head.weight 允许缺失。
    allowed_missing_keys = {"lm_head.weight"}
    unexpected_missing = set(load_result.missing_keys) - allowed_missing_keys
    if unexpected_missing or load_result.unexpected_keys:
        raise RuntimeError(
            "权重名称不匹配，missing="
            + str(sorted(unexpected_missing))
            + ", unexpected="
            + str(sorted(load_result.unexpected_keys))
        )

    model.tie_weights()
    model_dtype = model_dtype_from_config(config)

    # model.to(dtype=...) 会连 float buffer 一起转换。RoPE 的频率必须保留
    # float32 精度，因此先保存一份，再在模型移动完成后放回目标设备。
    rotary_inv_freq = model.model.rotary_emb.inv_freq.clone()
    model = model.to(device=device, dtype=model_dtype)
    model.model.rotary_emb.inv_freq = rotary_inv_freq.to(device)
    model.eval()
    return model

