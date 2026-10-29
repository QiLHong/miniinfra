"""第 14 期实验公共入口：固定版本模型、Cache Service 生命周期和 JSON。"""

import json
import os
import platform
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from cache_protocol import ExternalCacheClient
from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    Qwen3Config,
    load_handwritten_model,
    resolve_model_directory,
)


def parse_dtype(value):
    """把命令行精度名称转换为 torch.dtype，供模型与缓存使用一致的数据类型。"""
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if value not in mapping:
        raise ValueError("dtype 必须是 float32、float16 或 bfloat16")
    return mapping[value]


def free_local_port():
    """临时绑定本机的自动分配端口，读取端口号后关闭探测 socket。

    返回值不是持续占用的端口预留，真正的服务仍需成功绑定才能使用。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def cache_service(capacity_mib=1024.0, port=None):
    """启动本机独立缓存子进程，等待 ping 成功后提供客户端，并在退出上下文时关闭服务。"""
    port = free_local_port() if port is None else int(port)
    script = Path(__file__).with_name("cache_server.py")
    process = subprocess.Popen(
        [
            sys.executable,
            str(script),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--capacity-mib",
            str(capacity_mib),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    client = ExternalCacheClient(port=port)
    deadline = time.time() + 10
    while time.time() < deadline:
        if process.poll() is not None:
            error = process.stderr.read()
            raise RuntimeError("Cache Service 启动失败: " + error)
        try:
            client.ping()
            break
        except (ConnectionError, OSError):
            time.sleep(0.05)
    else:
        process.terminate()
        raise RuntimeError("Cache Service 启动超时")
    try:
        yield client, port
    finally:
        if process.poll() is None:
            try:
                client.shutdown()
                process.wait(timeout=5)
            except Exception:
                process.terminate()
                process.wait(timeout=5)


def load_model(model_name, revision, device, dtype):
    """解析模型目录并按实验指定的设备和精度加载手写模型，返回目录及所需模型信息。"""
    directory = resolve_model_directory(model_name, revision)
    config = Qwen3Config.from_model_directory(directory)
    model = load_handwritten_model(directory, device, dtype=dtype)
    return directory, config, model


def environment_record(model_name, revision, device, dtype):
    """生成本次实验的软件与硬件环境记录，便于回溯结果产生条件。"""
    record = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "model": model_name,
        "revision": revision,
        "pid": os.getpid(),
    }
    if torch.cuda.is_available():
        record["gpu"] = torch.cuda.get_device_name(torch.device(device))
        record["gpu_memory_bytes"] = torch.cuda.get_device_properties(
            torch.device(device)
        ).total_memory
    return record


def write_json(path, value):
    """把实验结果写为 UTF-8 JSON，便于保留原始参数、环境和指标。"""
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def add_model_arguments(parser):
    """为实验入口注册共用的模型、版本、设备与精度参数，避免各入口配置不一致。"""
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
