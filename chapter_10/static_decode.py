"""第 10 期：可由 CUDA Graph 捕获的静态 Paged Decode 执行器。

核心对照 ``static_eager`` 与 ``cuda_graph`` 调用完全相同的 ``_decode_step``。
两者使用相同的静态 Batch、Context Bucket、Block Table、KV Pool 和 Attention；
唯一差别是前者逐个提交 CUDA Kernel，后者重放已经捕获的 CUDA Graph。
"""

import math
import time

import torch

from qwen3_model import (
    apply_rotary_embedding,
    dense_attention_forward,
    repeat_key_value,
)


class StaticPagedKVCache:
    """为静态 Decode Bucket 预留物理块和固定宽度 Block Table。

    教学实现仍通过 Block Table 间接寻址物理 KV Block，但为了保证 Graph 捕获期间
    Tensor 地址和形状稳定，会提前为每个 Slot 预留整个 Context Bucket。它不是
    生产级动态 Block 分配器，这一预留代价会在实验中单独报告。
    """

    def __init__(self, config, capacity, context_bucket, block_size, device, dtype):
        """提前为每个槽位预留整个上下文桶，并另分配一个接收无效写入的 sink 块。

        这样块表宽度和底层地址在图重放时固定；代价是空槽和短请求也占预留容量。"""
        if capacity < 1:
            raise ValueError("capacity 必须大于 0")
        if context_bucket < 1:
            raise ValueError("context_bucket 必须大于 0")
        if block_size < 1:
            raise ValueError("block_size 必须大于 0")
        self.config = config
        self.capacity = int(capacity)
        self.context_bucket = int(context_bucket)
        self.block_size = int(block_size)
        self.device = torch.device(device)
        self.dtype = dtype
        self.blocks_per_slot = math.ceil(context_bucket / block_size)
        self.data_block_count = self.capacity * self.blocks_per_slot
        self.sink_block_count = self.capacity
        self.total_block_count = self.data_block_count + self.sink_block_count

        self.blocks = torch.zeros(
            (
                config.num_hidden_layers,
                2,
                self.total_block_count,
                config.num_key_value_heads,
                block_size,
                config.head_dim,
            ),
            dtype=dtype,
            device=self.device,
        )
        self.block_table = torch.arange(
            self.data_block_count, dtype=torch.long, device=self.device
        ).reshape(self.capacity, self.blocks_per_slot)
        self.sink_block_ids = torch.arange(
            self.data_block_count,
            self.total_block_count,
            dtype=torch.long,
            device=self.device,
        )

    @property
    def bytes_per_block(self):
        """计算一个包含所有层 K/V 的物理块所需的字节数。"""
        values = (
            self.config.num_hidden_layers
            * 2
            * self.config.num_key_value_heads
            * self.block_size
            * self.config.head_dim
        )
        return values * torch.empty((), dtype=self.dtype).element_size()

    @property
    def pool_bytes(self):
        """返回数据块与 sink 块共同占用的总池容量。"""
        return self.total_block_count * self.bytes_per_block

    @property
    def reserved_token_slots(self):
        """统计为真实上下文预留的 token 槽位，不包含用于丢弃无效写入的 sink 块。"""
        return self.data_block_count * self.block_size

    def store_prompt(self, slot, prompt_cache):
        """将单请求 Prefill 缓存逐块复制进指定槽位的固定块表。

        这是图捕获外的准备步骤，可以读取 Python 整数块编号；重放阶段不重新分配块。"""
        if not 0 <= slot < self.capacity:
            raise IndexError("slot 超出范围")
        if len(prompt_cache) != self.config.num_hidden_layers:
            raise ValueError("Prompt KV 层数与模型配置不一致")
        prompt_length = prompt_cache[0][0].shape[2]
        if prompt_length > self.context_bucket:
            raise ValueError("Prompt 长度超过 Context Bucket")
        table = self.block_table[slot]
        for layer_index, (key, value) in enumerate(prompt_cache):
            expected = (
                1,
                self.config.num_key_value_heads,
                prompt_length,
                self.config.head_dim,
            )
            if tuple(key.shape) != expected or tuple(value.shape) != expected:
                raise ValueError("Prompt KV 形状错误")
            for logical_block in range(math.ceil(prompt_length / self.block_size)):
                start = logical_block * self.block_size
                end = min(prompt_length, start + self.block_size)
                count = end - start
                block_id = int(table[logical_block].item())
                self.blocks[
                    layer_index, 0, block_id, :, :count, :
                ].copy_(key[0, :, start:end, :])
                self.blocks[
                    layer_index, 1, block_id, :, :count, :
                ].copy_(value[0, :, start:end, :])

    def dense_slot_cache(self, slot, logical_length):
        """按指定逻辑长度重建单槽的逐层稠密 K/V，用于与常规模型路径逐项比较。"""
        block_count = math.ceil(logical_length / self.block_size)
        block_ids = self.block_table[slot:slot + 1, :block_count]
        result = []
        for layer_index in range(self.config.num_hidden_layers):
            key = self.blocks[layer_index, 0, block_ids]
            value = self.blocks[layer_index, 1, block_ids]
            key = key.permute(0, 2, 1, 3, 4).reshape(
                1,
                self.config.num_key_value_heads,
                block_count * self.block_size,
                self.config.head_dim,
            )[:, :, :logical_length, :]
            value = value.permute(0, 2, 1, 3, 4).reshape(
                1,
                self.config.num_key_value_heads,
                block_count * self.block_size,
                self.config.head_dim,
            )[:, :, :logical_length, :]
            result.append((key, value))
        return result


class StaticDecodeRunner:
    """固定 Batch/Context Bucket 的 Eager 与 CUDA Graph 共用执行器。"""

    MODES = ("static_eager", "cuda_graph")

    def __init__(
        self,
        model,
        capacity,
        context_bucket,
        block_size=16,
        pad_token_id=0,
    ):
        """分配固定地址的输入、位置、长度、活跃掩码、输出和块池，供 eager 与图重放共用。"""
        self.model = model
        self.config = model.config
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.capacity = int(capacity)
        self.context_bucket = int(context_bucket)
        self.block_size = int(block_size)
        self.pad_token_id = int(pad_token_id)
        if not 0 <= self.pad_token_id < self.config.vocab_size:
            raise ValueError("pad_token_id 超出词表范围")

        self.cache = StaticPagedKVCache(
            self.config,
            self.capacity,
            self.context_bucket,
            self.block_size,
            self.device,
            self.dtype,
        )
        self.input_ids = torch.full(
            (self.capacity, 1),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        self.position_ids = torch.zeros(
            (self.capacity, 1), dtype=torch.long, device=self.device
        )
        self.sequence_lengths = torch.zeros(
            self.capacity, dtype=torch.long, device=self.device
        )
        self.active_mask = torch.zeros(
            self.capacity, dtype=torch.bool, device=self.device
        )
        self.slot_indices = torch.arange(
            self.capacity, dtype=torch.long, device=self.device
        )
        self.key_positions = torch.arange(
            self.context_bucket, dtype=torch.long, device=self.device
        )
        self.output_tokens = torch.full(
            (self.capacity,),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        self.last_logits = torch.zeros(
            (self.capacity, self.config.vocab_size),
            dtype=self.dtype,
            device=self.device,
        )
        self.graph = None
        self.capture_ms = 0.0
        self.capture_allocated_bytes = 0
        self.capture_reserved_bytes = 0
        self.active_count = 0
        self.prompt_first_tokens = []
        self.remaining_decode_steps = 0

    @property
    def static_buffer_bytes(self):
        """统计固定控制与输入输出张量的字节数；大型 K/V 块池由 pool_bytes 单独报告。"""
        tensors = (
            self.input_ids,
            self.position_ids,
            self.sequence_lengths,
            self.active_mask,
            self.slot_indices,
            self.key_positions,
            self.output_tokens,
            self.last_logits,
            self.cache.block_table,
            self.cache.sink_block_ids,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def _write_layer_token(self, layer_index, key, value):
        """将有效槽的新 K/V 写到逻辑末尾，无效槽写到专用 sink 块。

        地址选择全部在设备张量上进行，避免在捕获路径中依赖 .item() 或动态 Python 分支。"""
        logical_blocks = torch.div(
            self.sequence_lengths, self.block_size, rounding_mode="floor"
        )
        offsets = torch.remainder(self.sequence_lengths, self.block_size)
        physical_blocks = self.cache.block_table[
            self.slot_indices, logical_blocks
        ]
        # 活跃槽写真实块，无效槽写专用 sink 块；用设备上的选择替代依赖 Python 的动态分支。
        write_blocks = torch.where(
            self.active_mask, physical_blocks, self.cache.sink_block_ids
        )
        write_offsets = torch.where(
            self.active_mask, offsets, torch.zeros_like(offsets)
        )
        self.cache.blocks[
            layer_index, 0, write_blocks, :, write_offsets, :
        ] = key[:, :, 0, :]
        self.cache.blocks[
            layer_index, 1, write_blocks, :, write_offsets, :
        ] = value[:, :, 0, :]

    def _read_layer_bucket(self, layer_index):
        """按固定块表收集整桶 K/V，输出长度保持 context_bucket，实际有效范围由 Mask 限制。"""
        block_ids = self.cache.block_table
        key = self.cache.blocks[layer_index, 0, block_ids]
        value = self.cache.blocks[layer_index, 1, block_ids]
        key = key.permute(0, 2, 1, 3, 4).reshape(
            self.capacity,
            self.config.num_key_value_heads,
            self.cache.blocks_per_slot * self.block_size,
            self.config.head_dim,
        )[:, :, :self.context_bucket, :]
        value = value.permute(0, 2, 1, 3, 4).reshape(
            self.capacity,
            self.config.num_key_value_heads,
            self.cache.blocks_per_slot * self.block_size,
            self.config.head_dim,
        )[:, :, :self.context_bucket, :]
        return key, value

    def _decode_step(self):
        """执行固定批大小的一次 Decode，并原位更新下一轮输入、位置、长度及输出缓冲。

        每层先写入本轮 K/V，再读取固定上下文桶做 Attention；空槽通过 active_mask
        屏蔽。固定形状加原位更新使同一段操作可被 CUDA Graph 捕获并反复重放。"""
        hidden_states = self.model.model.embed_tokens(self.input_ids)
        cosine, sine = self.model.model.rotary_emb(
            self.position_ids, hidden_states.dtype
        )
        key_valid = self.key_positions.unsqueeze(0) < (
            self.sequence_lengths + 1
        ).unsqueeze(1)
        key_valid = key_valid & self.active_mask.unsqueeze(1)
        query_valid = self.active_mask.unsqueeze(1)

        for layer_index, layer in enumerate(self.model.model.layers):
            residual = hidden_states
            normalized = layer.input_layernorm(hidden_states)
            attention = layer.self_attn
            query = attention.q_norm(
                attention.q_proj(normalized).view(
                    self.capacity,
                    1,
                    attention.num_attention_heads,
                    attention.head_dim,
                )
            ).transpose(1, 2)
            key = attention.k_norm(
                attention.k_proj(normalized).view(
                    self.capacity,
                    1,
                    attention.num_key_value_heads,
                    attention.head_dim,
                )
            ).transpose(1, 2)
            value = attention.v_proj(normalized).view(
                self.capacity,
                1,
                attention.num_key_value_heads,
                attention.head_dim,
            ).transpose(1, 2)
            query, key = apply_rotary_embedding(query, key, cosine, sine)
            self._write_layer_token(layer_index, key, value)

            bucket_key, bucket_value = self._read_layer_bucket(layer_index)
            bucket_key = repeat_key_value(
                bucket_key, attention.num_key_value_groups
            )
            bucket_value = repeat_key_value(
                bucket_value, attention.num_key_value_groups
            )
            attention_output = dense_attention_forward(
                attention.attention_backend,
                query,
                bucket_key,
                bucket_value,
                key_valid,
                query_valid,
                self.context_bucket - 1,
            )
            attention_output = attention_output.transpose(1, 2).contiguous().view(
                self.capacity, 1, -1
            )
            hidden_states = residual + attention.o_proj(attention_output)
            residual = hidden_states
            hidden_states = residual + layer.mlp(
                layer.post_attention_layernorm(hidden_states)
            )

        hidden_states = self.model.model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)[:, -1, :]
        next_tokens = torch.argmax(logits, dim=-1)
        next_tokens = torch.where(
            self.active_mask,
            next_tokens,
            torch.full_like(next_tokens, self.pad_token_id),
        )
        self.last_logits.copy_(logits)
        # 输出复用同一缓冲，下次重放会覆盖；要保存完整输出历史，需要在调用方复制。
        self.output_tokens.copy_(next_tokens)
        self.input_ids[:, 0].copy_(next_tokens)
        # 只为活跃请求推进长度，空槽保持不变；原位修改使图重放继续使用固定地址。
        self.sequence_lengths.add_(self.active_mask.to(torch.long))
        self.position_ids[:, 0].copy_(
            torch.where(
                self.active_mask,
                self.sequence_lengths,
                torch.zeros_like(self.sequence_lengths),
            )
        )

    @torch.inference_mode()
    def prepare_prompts(self, prompt_token_ids):
        """在捕获路径外完成每个提示的 Prefill，将首 token 和对应长度填入固定缓冲。

        返回首 token 列表；这些 token 将作为下一次 step 的输入，尚未写入缓存。"""
        if len(prompt_token_ids) < 1:
            raise ValueError("至少需要一条 Prompt")
        if len(prompt_token_ids) > self.capacity:
            raise ValueError("Prompt 数量超过 Batch Capacity")
        self.input_ids.fill_(self.pad_token_id)
        self.position_ids.zero_()
        self.sequence_lengths.zero_()
        self.active_mask.zero_()
        first_tokens = []
        for slot, raw_tokens in enumerate(prompt_token_ids):
            tokens = [int(token) for token in raw_tokens]
            if not tokens:
                raise ValueError("Prompt 不能为空")
            if len(tokens) >= self.context_bucket:
                raise ValueError("Prompt 必须为后续 Decode 留出至少一个 Token")
            input_ids = torch.tensor(
                tokens, dtype=torch.long, device=self.device
            ).unsqueeze(0)
            positions = torch.arange(
                len(tokens), dtype=torch.long, device=self.device
            ).unsqueeze(0)
            mask = torch.ones(
                (1, len(tokens)), dtype=torch.bool, device=self.device
            )
            logits, prompt_cache = self.model(
                input_ids,
                attention_mask=mask,
                position_ids=positions,
                use_cache=True,
            )
            self.cache.store_prompt(slot, prompt_cache)
            token = torch.argmax(logits[0, -1, :], dim=-1)
            self.input_ids[slot, 0].copy_(token)
            self.position_ids[slot, 0] = len(tokens)
            self.sequence_lengths[slot] = len(tokens)
            self.active_mask[slot] = True
            first_tokens.append(int(token.item()))
            del input_ids, positions, mask, logits, prompt_cache
        self.active_count = len(prompt_token_ids)
        self.prompt_first_tokens = first_tokens
        self.remaining_decode_steps = min(
            self.context_bucket - len(tokens) for tokens in prompt_token_ids
        )
        return list(first_tokens)

    @torch.inference_mode()
    def capture(self, warmup_steps=3):
        """用无活跃槽的状态预热并捕获一次 _decode_step，记录捕获时间和内存增量。

        捕获会重置输入及长度；真实请求需随后通过 prepare_prompts 填入。预热使用
        独立流并建立等待关系，确保捕获开始前相关工作已经完成。"""
        if self.device.type != "cuda":
            raise RuntimeError("CUDA Graph 只能在 CUDA 设备上捕获")
        if warmup_steps < 1:
            raise ValueError("warmup_steps 必须大于 0")

        self.input_ids.fill_(self.pad_token_id)
        self.position_ids.zero_()
        self.sequence_lengths.zero_()
        self.active_mask.zero_()
        current_stream = torch.cuda.current_stream(self.device)
        warmup_stream = torch.cuda.Stream(device=self.device)
        # 先建立流依赖，确保预热不会抢在当前流的初始化操作之前读取缓冲。
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            for _ in range(warmup_steps):
                self._decode_step()
        current_stream.wait_stream(warmup_stream)
        torch.cuda.synchronize(self.device)

        allocated_before = torch.cuda.memory_allocated(self.device)
        reserved_before = torch.cuda.memory_reserved(self.device)
        start = time.perf_counter()
        self.graph = torch.cuda.CUDAGraph()
        # 捕获的是固定地址与形状上的操作序列；之后只更新缓冲内容并 replay。
        with torch.cuda.graph(self.graph):
            self._decode_step()
        torch.cuda.synchronize(self.device)
        self.capture_ms = (time.perf_counter() - start) * 1000
        self.capture_allocated_bytes = max(
            0, torch.cuda.memory_allocated(self.device) - allocated_before
        )
        self.capture_reserved_bytes = max(
            0, torch.cuda.memory_reserved(self.device) - reserved_before
        )
        return {
            "capture_ms": self.capture_ms,
            "capture_allocated_bytes": self.capture_allocated_bytes,
            "capture_reserved_bytes": self.capture_reserved_bytes,
        }

    @torch.inference_mode()
    def step(self, mode):
        """检查容量后执行一次 static_eager 或图重放，返回固定输出缓冲。

        返回张量下次 step 会被覆盖；需要保留历史输出的调用方应自行复制。"""
        if mode not in self.MODES:
            raise ValueError("mode 必须是 static_eager 或 cuda_graph")
        if self.remaining_decode_steps < 1:
            raise RuntimeError("Decode 将超过 Context Bucket")
        if mode == "static_eager":
            self._decode_step()
        else:
            if self.graph is None:
                raise RuntimeError("cuda_graph 模式必须先调用 capture()")
            self.graph.replay()
        self.remaining_decode_steps -= 1
        return self.output_tokens

    def snapshot(self):
        """汇总静态桶容量、活跃槽数、预留内存与捕获成本，便于解释图加速的额外代价。"""
        return {
            "capacity": self.capacity,
            "active_count": self.active_count,
            "context_bucket": self.context_bucket,
            "block_size": self.block_size,
            "blocks_per_slot": self.cache.blocks_per_slot,
            "pool_blocks": self.cache.total_block_count,
            "pool_bytes": self.cache.pool_bytes,
            "reserved_token_slots": self.cache.reserved_token_slots,
            "static_buffer_bytes": self.static_buffer_bytes,
            "capture_ms": self.capture_ms,
            "capture_allocated_bytes": self.capture_allocated_bytes,
            "capture_reserved_bytes": self.capture_reserved_bytes,
        }
