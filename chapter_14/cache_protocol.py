"""第 14 期外部 KV Cache 的进程间协议与 Token Chunk 标识。

协议刻意保持简单：TCP 只承载长度前缀 JSON 和可选原始字节。CacheKey 由模型
命名空间、父 Chunk 摘要和当前 Chunk Token IDs 共同决定，不能使用进程内 Block ID。
"""

import hashlib
import json
import socket
import struct
from dataclasses import dataclass


PROTOCOL_VERSION = 1
HEADER = struct.Struct("!Q")


def _canonical_json(value):
    """按固定键顺序和紧凑格式编码 UTF-8 JSON，使相同元数据得到稳定的字节表示。"""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def recv_exact(connection, size):
    """循环接收直到凑齐指定字节数，连接提前关闭时报错。

    TCP 是字节流，一次 recv 不保证返回一条完整消息，必须自行处理短读。"""
    chunks = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError("连接在消息接收完成前关闭")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(connection, metadata, payload=b""):
    """依次发送 JSON 长度、JSON 元数据和原始 payload，接收方据长度还原消息边界。"""
    metadata = dict(metadata)
    metadata["protocol_version"] = PROTOCOL_VERSION
    metadata["payload_bytes"] = len(payload)
    header = _canonical_json(metadata)
    connection.sendall(HEADER.pack(len(header)))
    connection.sendall(header)
    if payload:
        connection.sendall(payload)


def receive_message(connection, max_header_bytes=1024 * 1024):
    """解析长度前缀与 JSON，检查协议版本后按 payload_bytes 读取二进制数据。"""
    header_size = HEADER.unpack(recv_exact(connection, HEADER.size))[0]
    if header_size < 2 or header_size > max_header_bytes:
        raise ValueError("协议 Header 长度非法: %d" % header_size)
    metadata = json.loads(recv_exact(connection, header_size).decode("utf-8"))
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("外部 KV Cache 协议版本不匹配")
    payload_size = int(metadata.get("payload_bytes", 0))
    if payload_size < 0:
        raise ValueError("协议 Payload 长度不能为负数")
    return metadata, recv_exact(connection, payload_size) if payload_size else b""


@dataclass(frozen=True)
class ChunkIdentity:
    """描述逻辑 token 块的位置、内容和父摘要；身份与任何进程内的物理块编号无关。"""
    digest: str
    parent_digest: str
    token_start: int
    token_count: int
    token_ids: tuple


class TokenChunker:
    """生成与引擎物理 Block ID 无关的链式 CacheKey。"""

    def __init__(self, namespace, chunk_size):
        """保存命名空间和外部 Chunk 大小，并将格式版本加入哈希输入。"""
        if chunk_size < 1:
            raise ValueError("external_chunk_size 必须大于 0")
        self.namespace = dict(namespace)
        self.chunk_size = int(chunk_size)
        namespace_with_protocol = dict(self.namespace)
        namespace_with_protocol["external_chunk_size"] = self.chunk_size
        namespace_with_protocol["format_version"] = 1
        self.namespace_bytes = _canonical_json(namespace_with_protocol)

    @property
    def namespace_digest(self):
        """返回当前命名空间的摘要，隔离模型配置、格式或 Chunk 大小不同的缓存。"""
        return hashlib.sha256(self.namespace_bytes).hexdigest()

    def identities(self, token_ids, leave_last_token=True):
        """从提示开头生成完整 Chunk 的链式身份；每个摘要都依赖前一个 Chunk。

        默认留下最后一个提示 token 继续前向，保证即使大部分前缀命中，也能计算
        首个输出 token 的 logits。外部 Chunk 大小不必等于引擎物理块大小。"""
        reusable = len(token_ids) - 1 if leave_last_token else len(token_ids)
        full_chunks = max(0, reusable // self.chunk_size)
        parent = ""
        result = []
        for index in range(full_chunks):
            start = index * self.chunk_size
            values = tuple(int(value) for value in token_ids[start:start + self.chunk_size])
            hasher = hashlib.sha256()
            hasher.update(self.namespace_bytes)
            hasher.update(bytes.fromhex(parent) if parent else b"")
            for value in values:
                hasher.update(value.to_bytes(8, "little", signed=True))
            digest = hasher.hexdigest()
            result.append(ChunkIdentity(
                digest=digest,
                parent_digest=parent,
                token_start=start,
                token_count=self.chunk_size,
                token_ids=values,
            ))
            parent = digest
        return result


class ExternalCacheClient:
    """把缓存查找、读取和发布封装成独立的 TCP 请求，返回元数据或 KV 字节流。"""
    def __init__(self, host="127.0.0.1", port=65432, timeout=120.0):
        """保存服务地址和超时；构造对象本身不会连接或启动服务。"""
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)

    def _request(self, metadata, payload=b""):
        """为一次操作建立连接、发送请求并读取完整响应，将服务端错误转换为异常。"""
        with socket.create_connection(
            (self.host, self.port), timeout=self.timeout
        ) as connection:
            connection.settimeout(self.timeout)
            send_message(connection, metadata, payload)
            response, response_payload = receive_message(connection)
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "外部 KV Cache 请求失败"))
        return response, response_payload

    def ping(self):
        """检查服务是否可响应，供实验启动阶段等待就绪。"""
        response, _ = self._request({"op": "ping"})
        return response

    def lookup(self, identities):
        """查询从开头连续命中的 Chunk 数，只交换身份元数据，不读取 K/V 内容。"""
        response, _ = self._request({
            "op": "lookup",
            "entries": [
                {
                    "digest": item.digest,
                    "parent_digest": item.parent_digest,
                    "token_count": item.token_count,
                    "token_ids": list(item.token_ids),
                }
                for item in identities
            ],
        })
        return int(response["hit_chunks"])

    def load(self, identities):
        """加载指定 Chunk 链，按各条目的长度拆分 payload，并逐块验证内容摘要。

        lookup 与 load 是不同请求；两者之间条目可能已被淘汰，因此 load 仍可能失败。"""
        response, payload = self._request({
            "op": "load",
            "entries": [
                {
                    "digest": item.digest,
                    "parent_digest": item.parent_digest,
                    "token_count": item.token_count,
                    "token_ids": list(item.token_ids),
                }
                for item in identities
            ],
        })
        sizes = [int(value) for value in response["entry_sizes"]]
        checksums = response["checksums"]
        if sum(sizes) != len(payload):
            raise RuntimeError("Load 返回的 Payload 长度与元数据不一致")
        chunks = []
        offset = 0
        for size, checksum in zip(sizes, checksums):
            value = payload[offset:offset + size]
            offset += size
            if hashlib.sha256(value).hexdigest() != checksum:
                raise RuntimeError("Load Payload 校验和不一致")
            chunks.append(value)
        return chunks

    def store(self, identity, payload, namespace_digest, dtype, shape, ancestors):
        """发送一个不可变 Chunk 及其模型、布局和校验元数据，并声明需要保护的祖先块。"""
        response, _ = self._request({
            "op": "store",
            "digest": identity.digest,
            "parent_digest": identity.parent_digest,
            "token_start": identity.token_start,
            "token_count": identity.token_count,
            "token_ids": list(identity.token_ids),
            "namespace_digest": namespace_digest,
            "dtype": dtype,
            "shape": list(shape),
            "checksum": hashlib.sha256(payload).hexdigest(),
            "protected_ancestors": list(ancestors),
        }, payload)
        return response

    def stats(self):
        """读取服务端容量、命中与淘汰计数，供实验报告记录。"""
        response, _ = self._request({"op": "stats"})
        return response["stats"]

    def clear(self):
        """请求清空服务端缓存内容，供冷缓存对照实验使用。"""
        response, _ = self._request({"op": "clear"})
        return response

    def shutdown(self):
        """请求结束本次实验使用的缓存服务进程。"""
        response, _ = self._request({"op": "shutdown"})
        return response
