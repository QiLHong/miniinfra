"""第 06 期 Prefill burst 与 Token Budget 受控实验。"""

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch

from engine import percentile, run_scheduler
from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_handwritten_model,
    resolve_model_directory,
)
from scheduler import make_request_specs


def parse_args():
    """解析本脚本的实验参数并返回配置；默认值、可选项和参数间约束以下方定义为准。"""
    parser = argparse.ArgumentParser(description="迭代级调度器受控实验")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--max-running-requests", type=int, default=8)
    parser.add_argument("--initial-requests", type=int, default=4)
    parser.add_argument("--initial-prompt-length", type=int, default=128)
    parser.add_argument("--initial-output", type=int, default=64)
    parser.add_argument(
        "--burst-prompt-lengths", type=int, nargs="+",
        default=[128, 512, 128, 1024, 128, 2048, 128, 512,
                 256, 1024, 256, 2048],
    )
    parser.add_argument("--burst-output", type=int, default=8)
    parser.add_argument("--burst-arrival-ms", type=float, default=200.0)
    parser.add_argument(
        "--token-budgets", type=int, nargs="+",
        default=[256, 512, 1024, 2048, 4096],
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", help="可选 JSON 输出路径")
    return parser.parse_args()


def make_sequence(length, vocab_size, salt):
    """按目标长度构造可重复的 token 序列，控制实验输入长度而不依赖生成内容。"""
    values = torch.arange(length, dtype=torch.long)
    return ((values * (7919 + salt) + 17 + salt) % (vocab_size - 1) + 1).tolist()


def build_specs(args, model):
    """按本实验设定的提示长度、输出预算与到达时间构造固定请求列表。"""
    lengths = (
        [args.initial_prompt_length] * args.initial_requests
        + args.burst_prompt_lengths
    )
    output_budgets = (
        [args.initial_output] * args.initial_requests
        + [args.burst_output] * len(args.burst_prompt_lengths)
    )
    arrivals = (
        [0.0] * args.initial_requests
        + [args.burst_arrival_ms] * len(args.burst_prompt_lengths)
    )
    sequences = [
        make_sequence(length, model.config.vocab_size, 41 * (index + 1))
        for index, length in enumerate(lengths)
    ]
    return make_request_specs(sequences, output_budgets, arrivals)


def mean(values):
    """求样本的算术平均值；调用方应提供非空样本。"""
    return sum(values) / len(values)


def initial_itls(run, initial_requests):
    """只收集最初一批请求的相邻输出时间差，观察后来的突发请求是否拖慢它们。"""
    values = []
    initial_ids = {str(index) for index in range(initial_requests)}
    for request in run["request_metrics"]:
        if request["request_id"] not in initial_ids:
            continue
        values.extend(
            right - left
            for left, right in zip(
                request["token_times_ms"], request["token_times_ms"][1:]
            )
        )
    return values


def burst_metric(run, initial_requests, key):
    """从后到达的突发请求中提取指定指标，与已有请求的 ITL 分开统计。"""
    return [
        item[key] for item in run["request_metrics"]
        if int(item["request_id"]) >= initial_requests
    ]


SUMMARY_KEYS = [
    "makespan_ms",
    "output_token_throughput_per_second",
    "queue_ms_p95",
    "ttft_ms_p95",
    "end_to_end_ms_p95",
    "itl_ms_p50",
    "itl_ms_p95",
    "itl_ms_max",
    "scheduler_ms",
    "scheduler_fraction",
    "prefill_iterations",
    "decode_iterations",
    "max_consecutive_prefill_iterations",
    "prefill_logical_tokens",
    "prefill_padded_tokens",
    "prefill_padding_fraction",
    "prefill_budget_utilization",
    "oversize_prefill_iterations",
    "peak_memory_bytes",
]


def summarize(label, policy, token_budget, runs, initial_requests):
    """汇总本组重复测量，保留脚本所需的均值、分位数或资源指标。

    单位沿用输入字段；多次运行的分位数平均值与合并原始样本后重算的分位数不同。"""
    row = {
        "label": label,
        "policy": policy,
        "token_budget": token_budget,
    }
    for key in SUMMARY_KEYS:
        row[key + "_mean"] = mean([run["metrics"][key] for run in runs])
    existing_p95 = []
    existing_max = []
    burst_ttft_p95 = []
    burst_queue_p95 = []
    for run in runs:
        itls = initial_itls(run, initial_requests)
        existing_p95.append(percentile(itls, 95))
        existing_max.append(max(itls) if itls else 0.0)
        burst_ttft_p95.append(
            percentile(burst_metric(run, initial_requests, "ttft_ms"), 95)
        )
        burst_queue_p95.append(
            percentile(burst_metric(run, initial_requests, "queue_ms"), 95)
        )
    row["existing_itl_ms_p95_mean"] = mean(existing_p95)
    row["existing_itl_ms_max_mean"] = mean(existing_max)
    row["burst_ttft_ms_p95_mean"] = mean(burst_ttft_p95)
    row["burst_queue_ms_p95_mean"] = mean(burst_queue_p95)
    row["repeat_samples"] = [
        {
            "makespan_ms": run["metrics"]["makespan_ms"],
            "output_token_throughput_per_second": run["metrics"][
                "output_token_throughput_per_second"
            ],
            "existing_itl_ms_p95": existing_p95[index],
            "existing_itl_ms_max": existing_max[index],
            "burst_ttft_ms_p95": burst_ttft_p95[index],
            "prefill_padded_tokens": run["metrics"]["prefill_padded_tokens"],
        }
        for index, run in enumerate(runs)
    ]
    row["sample_schedule_trace"] = [
        {
            "iteration": event["iteration"],
            "phase": event["phase"],
            "admitted": event["admitted"],
            "completed": event["completed"],
            "scheduled_tokens": event["scheduled_tokens"],
            "token_budget": event["token_budget"],
            "oversize_singleton": event["oversize_singleton"],
            "total_ms": event["total_ms"],
        }
        for event in runs[0]["events"]
    ]
    return row


def clean_cuda(device):
    """清理实验间不再使用的对象与 CUDA 缓存，减少上一组测量对下一组的影响。"""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def system_environment(device):
    """记录软件和硬件环境，帮助解释不同机器或版本之间的性能差异。"""
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        ).stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.SubprocessError, IndexError):
        driver = "unknown"
    cpu = platform.processor()
    if not cpu or cpu.lower() in ("x86_64", "amd64"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as file:
                cpu = next(
                    line.split(":", 1)[1].strip() for line in file
                    if line.startswith("model name")
                )
        except (OSError, StopIteration):
            cpu = "unknown"
    properties = torch.cuda.get_device_properties(device)
    return {
        "gpu": torch.cuda.get_device_name(device),
        "gpu_memory_gib": properties.total_memory / 1024**3,
        "cpu": cpu,
        "system_memory_gib": (
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
        ),
        "operating_system": platform.platform(),
        "python": sys.version.split()[0],
        "gpu_driver": driver,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def main():
    """在固定请求到达序列上比较调度策略，观察吞吐和已有请求的输出间隔。"""
    args = parse_args()
    positive = [
        args.max_running_requests, args.initial_requests,
        args.initial_prompt_length, args.initial_output, args.burst_output,
        args.block_size, args.repeats,
    ] + args.burst_prompt_lengths + args.token_budgets
    if any(value < 1 for value in positive) or args.warmup < 0:
        raise ValueError("长度、数量、预算必须为正，warmup 不能小于 0")
    if args.initial_requests >= args.max_running_requests:
        raise ValueError("initial_requests 必须小于 max_running_requests")
    if args.burst_arrival_ms < 0:
        raise ValueError("burst_arrival_ms 不能小于 0")
    if any(budget < args.max_running_requests for budget in args.token_budgets):
        raise ValueError("Token Budget 不能小于最大运行请求数")
    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要可用的 NVIDIA GPU")

    torch.manual_seed(0)
    device = torch.device("cuda")
    model_directory = resolve_model_directory(args.model, args.revision)
    model = load_handwritten_model(model_directory, device)
    specs = build_specs(args, model)
    configurations = [("baseline", "baseline", None)] + [
        ("budget_%d" % budget, "budgeted", budget)
        for budget in args.token_budgets
    ]

    for _, policy, budget in configurations:
        for _ in range(args.warmup):
            run_scheduler(
                model, specs, args.max_running_requests, -1, device,
                policy=policy, token_budget=budget,
                block_size=args.block_size, stop_on_eos=False,
            )
            clean_cuda(device)

    collected = {label: [] for label, _, _ in configurations}
    for repeat in range(args.repeats):
        order = configurations if repeat % 2 == 0 else list(reversed(configurations))
        for label, policy, budget in order:
            collected[label].append(run_scheduler(
                model, specs, args.max_running_requests, -1, device,
                policy=policy, token_budget=budget,
                block_size=args.block_size, stop_on_eos=False,
            ))
            clean_cuda(device)

    rows = [
        summarize(label, policy, budget, collected[label], args.initial_requests)
        for label, policy, budget in configurations
    ]
    baseline = rows[0]
    for row in rows:
        row["comparison_to_baseline"] = {
            "throughput_gain": (
                row["output_token_throughput_per_second_mean"]
                / baseline["output_token_throughput_per_second_mean"] - 1
            ),
            "existing_itl_p95_reduction": 1 - (
                row["existing_itl_ms_p95_mean"]
                / baseline["existing_itl_ms_p95_mean"]
            ),
            "existing_itl_max_reduction": 1 - (
                row["existing_itl_ms_max_mean"]
                / baseline["existing_itl_ms_max_mean"]
            ),
            "burst_ttft_change": (
                row["burst_ttft_ms_p95_mean"]
                / baseline["burst_ttft_ms_p95_mean"] - 1
            ),
            "prefill_padding_reduction": 1 - (
                row["prefill_padded_tokens_mean"]
                / baseline["prefill_padded_tokens_mean"]
            ),
        }
        print(
            "%s throughput=%.2f token/s existing-ITL-p95=%.2f ms "
            "burst-TTFT-p95=%.2f ms padding=%.1f%% oversize=%.1f"
            % (
                row["label"],
                row["output_token_throughput_per_second_mean"],
                row["existing_itl_ms_p95_mean"],
                row["burst_ttft_ms_p95_mean"],
                row["prefill_padding_fraction_mean"] * 100,
                row["oversize_prefill_iterations_mean"],
            )
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "environment": {
                **system_environment(device),
                "model": args.model,
                "revision": args.revision,
                "dtype": str(next(model.parameters()).dtype),
                "warmup": args.warmup,
                "repeats": args.repeats,
                "decoding": "greedy, forced output budget, EOS disabled",
                "input": "synthetic exact-length token IDs",
                "tokenizer_included": False,
                "network_included": False,
                "arrival_time": "logical timeline; no real sleep",
                "cuda_allocator_config": os.environ.get(
                    "PYTORCH_CUDA_ALLOC_CONF", "default"
                ),
            },
            "workload": {
                "max_running_requests": args.max_running_requests,
                "initial_requests": args.initial_requests,
                "initial_prompt_length": args.initial_prompt_length,
                "initial_output": args.initial_output,
                "burst_prompt_lengths": args.burst_prompt_lengths,
                "burst_output": args.burst_output,
                "burst_arrival_ms": args.burst_arrival_ms,
                "block_size": args.block_size,
            },
            "results": rows,
        }
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        print("JSON 结果已写入:", output_path)


if __name__ == "__main__":
    main()
