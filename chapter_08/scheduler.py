"""第 08 期独立使用的 Chunked Prefill 调度器。

Prefix Cache disabled/enabled 两条路径共用完全相同的 FCFS、硬 Token Budget
和 Decode 保护规则。调度器只读取 ``prefill_cursor``，不参与缓存匹配。
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RequestSpec:
    """保存请求不可变的输入：标识、提示 token、最大输出数量和到达时间。"""
    request_id: str
    token_ids: tuple
    max_new_tokens: int
    arrival_ms: float = 0.0

    def validate(self):
        """在进入调度队列前检查标识、非空提示、正输出预算及非负到达时间。"""
        if not self.request_id:
            raise ValueError("request_id 不能为空")
        if not self.token_ids:
            raise ValueError("Prompt Token 不能为空")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens 必须大于 0")
        if self.arrival_ms < 0:
            raise ValueError("arrival_ms 不能小于 0")


@dataclass
class RequestState:
    """保存请求执行中的可变状态，与只描述输入的 RequestSpec 分开。

    关注 generated、缓存长度、槽位和各阶段时间戳；生成的最后一个 token
    通常还没有送入下一轮前向，因此输出长度不能直接当作已写入的缓存长度。"""
    spec: RequestSpec
    status: str = "waiting"
    generated: list = field(default_factory=list)
    slot_index: int | None = None
    cache_length: int = 0
    prefill_cursor: int = 0
    prefix_hit_tokens: int = 0
    prefix_lookup_ms: float = 0.0
    admitted_ms: float | None = None
    prefill_started_ms: float | None = None
    prefill_completed_ms: float | None = None
    first_token_ms: float | None = None
    completion_ms: float | None = None
    token_times_ms: list = field(default_factory=list)


def make_request_specs(token_sequences, max_new_tokens, arrival_times_ms=None):
    """把 token 序列、输出预算和到达时间对齐为 RequestSpec 列表，并逐个校验。"""
    sequences = [tuple(int(token) for token in sequence) for sequence in token_sequences]
    if isinstance(max_new_tokens, int):
        budgets = [max_new_tokens] * len(sequences)
    else:
        budgets = list(max_new_tokens)
    arrivals = (
        [0.0] * len(sequences)
        if arrival_times_ms is None else list(arrival_times_ms)
    )
    if len(budgets) != len(sequences) or len(arrivals) != len(sequences):
        raise ValueError("Prompt、输出预算和到达时间的数量必须一致")
    specs = [
        RequestSpec(str(index), sequence, int(budget), float(arrival))
        for index, (sequence, budget, arrival) in enumerate(
            zip(sequences, budgets, arrivals)
        )
    ]
    for spec in specs:
        spec.validate()
    return specs


@dataclass(frozen=True)
class SchedulerConfig:
    """描述调度器的并发槽位与 token 预算；不同策略是否允许超预算由本章规则决定。"""
    max_running_requests: int
    token_budget: int

    def validate(self):
        """检查策略支持范围、并发数与预算，确保一次完整 Decode 至少能容纳所有运行请求。"""
        if self.max_running_requests < 1:
            raise ValueError("max_running_requests 必须大于 0")
        if self.token_budget < self.max_running_requests:
            raise ValueError("token_budget 不能小于 max_running_requests")


@dataclass(frozen=True)
class PrefillPlan:
    """描述单个请求本轮要处理的提示区间 [start, end)，不包含 end 位置。"""
    state: RequestState
    start: int
    end: int

    @property
    def token_count(self):
        """返回本轮分块的有效 token 数，即 end-start，不含批内补齐位置。"""
        return self.end - self.start


@dataclass(frozen=True)
class SchedulerOutput:
    """记录一次迭代选中的阶段、请求与预算开销，交给引擎执行。

    scheduled_tokens 可包含补齐开销；logical_tokens 只统计实际内容，二者不要混用。"""
    phase: str
    selected: tuple
    prefill_plans: tuple = ()
    scheduled_tokens: int = 0
    logical_tokens: int = 0
    token_budget: int | None = None

    @property
    def request_ids(self):
        """按本轮选择顺序提取请求 ID，便于日志和结果对齐。"""
        return [state.spec.request_id for state in self.selected]


class ChunkedPrefillScheduler:
    """把长提示拆为多个可调度片段，在有限 token 预算下穿插 Decode。

    调度器返回计划，不写缓存；prefill_cursor 由引擎在相应片段执行完成后推进。"""
    def __init__(self, config):
        """检查本章的预算与并发配置，并初始化用于阶段选择的历史状态。"""
        config.validate()
        self.config = config
        self.last_phase = None

    @staticmethod
    def _padded_cost(lengths):
        """按片段数量×最长片段长度计算实际批大小，空列表开销为 0。"""
        return len(lengths) * max(lengths) if lengths else 0

    def _chunked_plans(self, candidates):
        """按候选请求数平分本轮 token 预算，从各自游标开始截取片段。

        使用向下取整的单请求上限，使补齐后的批大小也不超过硬预算；短请求
        只取实际剩余 token，不需要把未使用额度填成内容。"""
        # 均分的是实际批计算预算；向下取整保证补齐后的候选批也不会超过硬上限。
        chunk_cap = self.config.token_budget // len(candidates)
        plans = []
        for state in candidates:
            # 游标表示已经处理或复用的前缀长度，本轮从这里继续，而不是每次从提示开头重算。
            start = state.prefill_cursor
            remaining = len(state.spec.token_ids) - start
            if remaining < 1:
                raise RuntimeError("Prefill 请求没有剩余 Prompt Token")
            plans.append(PrefillPlan(state, start, start + min(remaining, chunk_cap)))
        if self._padded_cost([plan.token_count for plan in plans]) > self.config.token_budget:
            raise RuntimeError("Chunked Prefill 生成了超出硬预算的计划")
        return plans

    def schedule(self, waiting, prefilling, running, clock_ms):
        """结合等待、正在 Prefill 和正在 Decode 的队列选择本轮计划。

        先过滤尚未到达的请求，再检查槽位；已有 Decode 时通过上一阶段记录
        避免连续 Prefill 阻塞输出。具体完整/分块策略以本章配置分支为准。"""
        # 只允许已经到达的请求参与决策；尚未到达的请求即使排在列表中也不能提前执行。
        ready = [state for state in waiting if state.spec.arrival_ms <= clock_ms]
        free_slots = self.config.max_running_requests - len(prefilling) - len(running)
        if free_slots < 0:
            raise RuntimeError("运行请求数超过 max_running_requests")
        # 已有请求等着输出时，上一轮若做了 Prefill，本轮先让 Decode 前进，减少连续阻塞。
        if running and self.last_phase == "prefill":
            return self._decode_output(running)
        candidates = list(prefilling) + ready[:free_slots]
        if candidates:
            plans = self._chunked_plans(candidates)
            lengths = [plan.token_count for plan in plans]
            self.last_phase = "prefill"
            return SchedulerOutput(
                "prefill", tuple(plan.state for plan in plans), tuple(plans),
                self._padded_cost(lengths), sum(lengths), self.config.token_budget,
            )
        if running:
            return self._decode_output(running)
        return None

    def _decode_output(self, running):
        """选择运行中的请求各处理一个 token，并将上一阶段更新为 Decode。"""
        self.last_phase = "decode"
        return SchedulerOutput(
            "decode", tuple(running), scheduled_tokens=len(running),
            logical_tokens=len(running), token_budget=self.config.token_budget,
        )
