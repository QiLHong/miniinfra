"""第 11 期实验入口共用的模型加载、环境记录和计时辅助函数。"""

import json
import os
import platform
import random
import statistics
import time
from pathlib import Path

import torch

from qwen3_model import (
    DEFAULT_DRAFT_MODEL_ID,
    DEFAULT_TARGET_MODEL_ID,
    MODEL_REVISIONS,
    load_handwritten_model,
    resolve_model_directory,
)


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def set_seed(seed):
    """固定实验使用的随机数生成器种子，减少不同模式输入不一致造成的干扰。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_name_or_path, device, dtype, revision=None):
    """解析模型目录并按实验指定的设备和精度加载手写模型，返回目录及所需模型信息。"""
    model_directory = resolve_model_directory(model_name_or_path, revision=revision)
    model = load_handwritten_model(
        model_directory,
        device=torch.device(device),
        dtype=DTYPES[dtype],
    )
    return model_directory, model


def load_target_and_draft(
    target_model,
    draft_model,
    device,
    dtype,
    target_revision=None,
    draft_revision=None,
):
    """分别加载目标与草稿模型，并检查词表大小一致后返回两者及其模型目录。"""
    target_directory, target = load_model(
        target_model, device, dtype, revision=target_revision
    )
    draft_directory, draft = load_model(
        draft_model, device, dtype, revision=draft_revision
    )
    if target.config.vocab_size != draft.config.vocab_size:
        raise ValueError("Target 与 Draft 的词表大小不同")
    return target_directory, target, draft_directory, draft


def synchronize(device):
    """CUDA 设备上等待已提交工作完成，CPU 路径直接返回。

    GPU 调用通常异步返回；在主机计时边界同步，才能把实际执行时间纳入测量。"""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def timed_call(function, device):
    """在设备同步边界内计时一次函数调用，返回函数结果及以秒为单位的耗时。"""
    synchronize(device)
    start = time.perf_counter()
    result = function()
    synchronize(device)
    return result, time.perf_counter() - start


def summarize(values):
    """统计非空样本的数量、均值、中位数和极值；各项保持输入样本的单位。"""
    if not values:
        raise ValueError("values 不能为空")
    ordered = sorted(values)
    return {
        "samples": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": ordered[0],
        "max": ordered[-1],
    }


def model_parameter_bytes(model):
    """按底层存储地址和大小去重后统计参数字节数，避免共享权重重复计数；不包括 KV 缓存。"""
    storages = {}
    for parameter in model.parameters():
        storage = parameter.untyped_storage()
        storages[(storage.data_ptr(), storage.nbytes())] = storage.nbytes()
    return sum(storages.values())


def environment_snapshot(
    target_model_path,
    draft_model_path,
    dtype,
    target_model=DEFAULT_TARGET_MODEL_ID,
    draft_model=DEFAULT_DRAFT_MODEL_ID,
):
    """记录本次运行的软件版本、系统与设备信息，供实验结果复现。"""
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": "cpu",
        "target_model": target_model,
        "target_revision": MODEL_REVISIONS.get(target_model),
        "target_path_kind": (
            "directory" if Path(target_model_path).is_dir() else "model_id"
        ),
        "draft_model": draft_model,
        "draft_revision": MODEL_REVISIONS.get(draft_model),
        "draft_path_kind": (
            "directory" if Path(draft_model_path).is_dir() else "model_id"
        ),
        "dtype": dtype,
        "pid": os.getpid(),
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        result.update(
            {
                "device": properties.name,
                "compute_capability": "%d.%d"
                % (properties.major, properties.minor),
                "total_memory_bytes": properties.total_memory,
            }
        )
    return result


def memory_snapshot():
    """读取当前设备内存统计，区分实际张量分配与分配器预留空间。"""
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def write_json(path, payload):
    """把实验结果写为 UTF-8 JSON，便于保留原始参数、环境和指标。"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def bytes_to_mib(value):
    """将字节数除以 1024² 转为 MiB；MiB 与十进制 MB 不同。"""
    return value / (1024 * 1024)
