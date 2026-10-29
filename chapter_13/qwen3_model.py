"""第 13 期独立使用的 Qwen3-0.6B 推理模型。

只依赖 PyTorch、huggingface-hub 和 safetensors，不引用其他期代码。
模型显式接收二维 Attention Mask 与 Position IDs，以支持左 Padding、
Batched Prefill 和分页 Cache 的权重计算组件。
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
        with (Path(model_directory) / "config.json").open(
            "r", encoding="utf-8"
        ) as file:
            raw = json.load(file)
        config = cls(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            head_dim=raw["head_dim"],
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw["rope_theta"],
            torch_dtype=raw.get("torch_dtype", "bfloat16"),
            tie_word_embeddings=raw.get("tie_word_embeddings", True),
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
        dimensions = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (rope_theta ** (dimensions / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids, output_dtype):
        """将 [批次, 长度] 的位置编号转换为 [批次, 长度, 每头维度] 的 cos/sin。

        位置乘逆频率得到角度。复制两份频率对应 Qwen3 将向量前后两半配对的布局，
        并不是为相邻两维交替排列的另一种 RoPE 布局生成系数。"""
        # 位置表补一个末维，与每对特征的频率相乘，得到 [批次, 长度, 每头维度/2] 的旋转角度。
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq
        embedding = torch.cat((freqs, freqs), dim=-1)
        return embedding.cos().to(output_dtype), embedding.sin().to(output_dtype)


def rotate_half(hidden_states):
    """把最后一维拆为两半，返回 [-后半, 前半]；输出形状与输入相同。

    它与 cos/sin 相乘后构成二维旋转，必须和模型使用的 RoPE 配对布局一致。"""
    half = hidden_states.shape[-1] // 2
    return torch.cat(
        (-hidden_states[..., half:], hidden_states[..., :half]), dim=-1
    )


def apply_rotary_embedding(query, key, cosine, sine):
    """将位置旋转应用到 Q 和 K，V 不参与旋转。

    Q/K 为 [批次, 头数, 长度, 每头维度]；cos/sin 在头数位置补一个单维度，
    于是同一 token 的位置系数可以广播到全部头。"""
    # 在头数维插入 1，让同一个位置的系数供所有 Query/KV 头共享。
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    return (
        query * cosine + rotate_half(query) * sine,
        key * cosine + rotate_half(key) * sine,
    )


def repeat_key_value(hidden_states, repeat_count):
    """沿头数维度重复 K/V，使它们能与 Query 的头数对应。

    输入为 [批次, KV头数, 长度, 每头维度]，repeat_count 是 Query/KV 头数之比。
    这是直观的 GQA 展开实现，会产生展开后的张量；缓存本身仍应保存较少的 KV 头。"""
    if repeat_count == 1:
        return hidden_states
    return hidden_states.repeat_interleave(repeat_count, dim=1)


def cache_size_bytes(past_key_values):
    """将逐层 K/V 的元素数量乘以每元素字节数，求和得到缓存张量的逻辑字节数。

    返回值不包含分配器预留、临时 Attention 张量或模型权重。"""
    if past_key_values is None:
        return 0
    return sum(
        key.numel() * key.element_size() + value.numel() * value.element_size()
        for key, value in past_key_values
    )


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
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        cosine,
        sine,
        attention_mask,
        past_key_value=None,
        use_cache=False,
    ):
        """将 [批次, 当前长度, 隐藏维度] 投影为分头 Q/K/V，计算注意力后合并回隐藏维度。

        先对当前 Q/K 做 RoPE，再沿序列维拼接历史 K/V。present_key_value 保存
        未展开的 KV 头；只有计算 Attention 时才按 GQA 组展开，以减少持久缓存。
        返回当前输出及可选的新缓存；Mask 控制可见历史和无效位置。"""
        batch_size, query_length, _ = hidden_states.shape
        query = self.q_norm(
            self.q_proj(hidden_states).view(
                batch_size, query_length, self.num_attention_heads, self.head_dim
            )
        ).transpose(1, 2)
        key = self.k_norm(
            self.k_proj(hidden_states).view(
                batch_size, query_length, self.num_key_value_heads, self.head_dim
            )
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        query, key = apply_rotary_embedding(query, key, cosine, sine)
        if past_key_value is not None:
            past_key, past_value = past_key_value
            # 沿序列维接在旧历史之后；此时 K 已做 RoPE，仍只保存原始 KV 头数。
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)
        # 先保留紧凑的 K/V，再为本次注意力展开 GQA，避免持久缓存膨胀到 Query 头数。
        present_key_value = (key, value) if use_cache else None

        attention_key = repeat_key_value(key, self.num_key_value_groups)
        attention_value = repeat_key_value(value, self.num_key_value_groups)
        scores = torch.matmul(query, attention_key.transpose(2, 3))
        scores = scores / math.sqrt(self.head_dim)
        scores = scores + attention_mask
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(probabilities, attention_value)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, query_length, -1
        )
        return self.o_proj(output), present_key_value


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
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


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

    def forward(
        self,
        hidden_states,
        cosine,
        sine,
        attention_mask,
        past_key_value=None,
        use_cache=False,
    ):
        """依次执行 Attention 残差块和 MLP 残差块。

        每个子层先保存原 hidden_states，再对归一化后的结果进行计算，最后加回原值；
        缓存参数传给 Attention，MLP 只处理当前 token 的隐藏状态。"""
        residual = hidden_states
        attention_output, present = self.self_attn(
            self.input_layernorm(hidden_states),
            cosine,
            sine,
            attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + attention_output
        residual = hidden_states
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present


class Qwen3Model(nn.Module):
    """由 token 嵌入、Decoder 层堆叠和末尾 RMSNorm 组成的模型主干。

    它输出隐藏状态；转换为词表上的 logits 由外层 Qwen3ForCausalLM 完成。"""
    def __init__(self, config):
        """按配置建立嵌入表、各层 Decoder、最终归一化和全模型共用的 RoPE 系数生成器。"""
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)

    def validate_past_key_values(self, past_key_values, batch_size):
        """核对历史缓存的层数与 [批次, KV头数, 历史长度, 每头维度] 形状，返回历史长度。

        所有层的 K/V 必须有同一历史长度；没有缓存时返回 0，表示首次 Prefill。"""
        if past_key_values is None:
            return 0
        if len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values 层数与模型层数不一致")
        past_length = past_key_values[0][0].shape[2]
        expected_prefix = (batch_size, self.config.num_key_value_heads)
        for layer_index, (key, value) in enumerate(past_key_values):
            expected = expected_prefix + (past_length, self.config.head_dim)
            if tuple(key.shape) != expected or tuple(value.shape) != expected:
                raise ValueError(
                    "第 %d 层 KV Cache 形状错误: key=%s, value=%s, expected=%s"
                    % (layer_index, tuple(key.shape), tuple(value.shape), expected)
                )
        return past_length

    def prepare_inputs(
        self, input_ids, attention_mask, position_ids, past_key_values
    ):
        """检查输入、历史缓存、Mask 与位置编号的形状，统一设备和 dtype。

        将物理因果关系与有效 Key 掩码合并，返回位置编号及 [批次, 1, Q长度, KV长度] 的允许矩阵。
        左 Padding 的真实位置需由调用方提供；默认位置按物理缓存长度连续编号。"""
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids 必须是 [batch, sequence] 非空 Tensor")
        batch_size, query_length = input_ids.shape
        past_length = self.validate_past_key_values(past_key_values, batch_size)
        key_value_length = past_length + query_length

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, key_value_length),
                dtype=torch.bool,
                device=input_ids.device,
            )
        if tuple(attention_mask.shape) != (batch_size, key_value_length):
            raise ValueError(
                "attention_mask 应为 %s，实际为 %s"
                % ((batch_size, key_value_length), tuple(attention_mask.shape))
            )
        attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)

        if position_ids is None:
            position_ids = torch.arange(
                past_length, key_value_length, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)
        if tuple(position_ids.shape) != (batch_size, query_length):
            raise ValueError(
                "position_ids 应为 %s，实际为 %s"
                % ((batch_size, query_length), tuple(position_ids.shape))
            )
        position_ids = position_ids.to(device=input_ids.device, dtype=torch.long)

        # 因果关系按缓存中的物理先后判断；RoPE 使用的逻辑 position_ids 另行传递。
        query_physical_positions = torch.arange(
            past_length, key_value_length, device=input_ids.device
        )
        key_physical_positions = torch.arange(
            key_value_length, device=input_ids.device
        )
        future = key_physical_positions.unsqueeze(0) > query_physical_positions.unsqueeze(1)
        # 一个 Key 必须既不是未来位置、又不是 Padding，才能被当前 Query 看到。
        allowed = (~future).unsqueeze(0) & attention_mask.unsqueeze(1)
        return position_ids, allowed.unsqueeze(1)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
    ):
        """把 token ID 嵌入为向量，准备位置与注意力 Mask，然后依次执行各层并做最终归一化。

        有缓存接口的章节按层传递历史并收集 present；缓存只保存 K/V，不保存 logits。"""
        if past_key_values is not None and not use_cache:
            raise ValueError("传入 past_key_values 时必须设置 use_cache=True")
        hidden_states = self.embed_tokens(input_ids)
        position_ids, allowed = self.prepare_inputs(
            input_ids, attention_mask, position_ids, past_key_values
        )
        cosine, sine = self.rotary_emb(position_ids, hidden_states.dtype)
        additive_mask = torch.zeros(
            allowed.shape, dtype=hidden_states.dtype, device=hidden_states.device
        )
        additive_mask.masked_fill_(~allowed, torch.finfo(hidden_states.dtype).min)

        # 每一层有独立 K/V，必须按层序收集；不能把上一层缓存传给下一层。
        present_key_values = [] if use_cache else None
        for layer_index, layer in enumerate(self.layers):
            layer_past = (
                None if past_key_values is None else past_key_values[layer_index]
            )
            hidden_states, present = layer(
                hidden_states,
                cosine,
                sine,
                additive_mask,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            if use_cache:
                present_key_values.append(present)
        return self.norm(hidden_states), present_key_values


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
        """让 lm_head.weight 引用同一个嵌入参数，避免保存两份相同的词表权重。"""
        # 绑定同一个 Parameter 对象，实现权重共享，而不是复制一份数值相同的张量。
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
    ):
        """执行主干并将隐藏状态映射为 [批次, 当前长度, 词表大小] 的 logits。

        同时返回更新后的逐层缓存；use_cache 决定是否保留缓存结果。"""
        hidden_states, present = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return self.lm_head(hidden_states), present


def resolve_model_directory(model_name_or_path, revision=DEFAULT_MODEL_REVISION):
    """本地目录存在时直接使用；否则按固定 revision 下载所需的模型与分词文件。

    固定版本使不同实验使用相同权重，下载耗时应与推理计时分开。"""
    local_path = Path(model_name_or_path).expanduser()
    if local_path.is_dir():
        return str(local_path.resolve())
    return snapshot_download(
        repo_id=model_name_or_path,
        revision=revision,
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
    supported = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if config.torch_dtype not in supported:
        raise ValueError("不支持的模型 dtype: " + config.torch_dtype)
    return supported[config.torch_dtype]


def load_handwritten_model(model_directory, device, dtype=None):
    """读取配置和 Safetensors 权重，核对参数名称后返回 eval 模式的模型。

    共享的 lm_head 权重允许不单独出现在文件中，加载后重新绑定。设备与精度
    转换还涉及 RoPE buffer，因此要关注这里对 inv_freq 的单独处理。"""
    config = Qwen3Config.from_model_directory(model_directory)
    model = Qwen3ForCausalLM(config)
    weight_path = Path(model_directory) / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError("没有找到模型权重: " + str(weight_path))
    state_dict = load_file(str(weight_path), device="cpu")
    result = model.load_state_dict(state_dict, strict=False)
    unexpected_missing = set(result.missing_keys) - {"lm_head.weight"}
    if unexpected_missing or result.unexpected_keys:
        raise RuntimeError(
            "权重名称不匹配，missing=%s, unexpected=%s"
            % (sorted(unexpected_missing), sorted(result.unexpected_keys))
        )
    model.tie_weights()
    target_dtype = model_dtype_from_config(config) if dtype is None else dtype
    if target_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("不支持的模型 dtype: " + str(target_dtype))
    rotary_inv_freq = model.model.rotary_emb.inv_freq.clone()
    model = model.to(device=device, dtype=target_dtype)
    model.model.rotary_emb.inv_freq = rotary_inv_freq.to(device)
    model.eval()
    return model
