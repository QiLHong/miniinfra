"""请求级 KV Handoff 的长度前缀 TCP 协议与客户端。"""

import hashlib
import json
import socket
import struct


MAX_HEADER_BYTES = 4 * 1024 * 1024
MAX_PAYLOAD_BYTES = 2 * 1024 * 1024 * 1024


def payload_digest(payload):
    """计算 payload 原始字节的 SHA-256，供发送端和接收端核对内容是否一致。"""
    return hashlib.sha256(payload).hexdigest()


def namespace_digest(namespace):
    """对稳定编码后的模型命名空间求摘要，用于拒绝版本、精度或布局不兼容的 K/V。"""
    encoded = json.dumps(
        namespace, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def recv_exact(sock, length):
    """循环接收直到凑齐指定字节数，连接提前关闭时报错。

    TCP 是字节流，一次 recv 不保证返回一条完整消息，必须自行处理短读。"""
    chunks = []
    received = 0
    while received < length:
        chunk = sock.recv(min(1024 * 1024, length - received))
        if not chunk:
            raise ConnectionError("连接在消息接收完成前关闭")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def send_message(sock, header, payload=b""):
    """依次发送 JSON 长度、JSON 元数据和原始 payload，接收方据长度还原消息边界。"""
    header = dict(header)
    header["payload_bytes"] = len(payload)
    encoded = json.dumps(
        header, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if len(encoded) > MAX_HEADER_BYTES or len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("协议消息超过教学实现的大小限制")
    sock.sendall(struct.pack("!I", len(encoded)))
    sock.sendall(encoded)
    if payload:
        sock.sendall(payload)


def recv_message(sock):
    """验证头部与 payload 的长度上限后，分别读取 JSON 元数据和二进制内容。"""
    header_length = struct.unpack("!I", recv_exact(sock, 4))[0]
    if header_length < 2 or header_length > MAX_HEADER_BYTES:
        raise ValueError("非法 Header 长度")
    header = json.loads(recv_exact(sock, header_length).decode("utf-8"))
    payload_length = int(header.get("payload_bytes", 0))
    if payload_length < 0 or payload_length > MAX_PAYLOAD_BYTES:
        raise ValueError("非法 Payload 长度")
    return header, recv_exact(sock, payload_length) if payload_length else b""


class HandoffClient:
    """封装单次请求的发布、接收、确认和释放，用 request_id/attempt_id 隔离不同尝试。"""
    def __init__(self, host="127.0.0.1", port=0, timeout=120.0):
        """保存交接服务地址与超时，实际连接在每次请求时建立。"""
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)

    def _request(self, header, payload=b""):
        """发送一次交接操作并读取响应，将服务返回的 error 状态转为异常。"""
        with socket.create_connection(
            (self.host, self.port), timeout=self.timeout
        ) as sock:
            sock.settimeout(self.timeout)
            send_message(sock, header, payload)
            response, response_payload = recv_message(sock)
        if response.get("status") == "error":
            raise RuntimeError(response.get("error", "Handoff Service 错误"))
        return response, response_payload

    def ping(self):
        """检查交接服务是否已经启动并能够响应。"""
        return self._request({"op": "ping"})[0]

    def publish(self, manifest, payload):
        """发布交接清单和提示 K/V 字节；发布成功尚不表示接收方已导入。"""
        return self._request(
            {"op": "publish", "manifest": manifest}, payload
        )[0]

    def status(self, request_id, attempt_id):
        """查询指定请求尝试的当前状态，供发送方等待确认或回退。"""
        return self._request({
            "op": "status", "request_id": request_id,
            "attempt_id": attempt_id,
        })[0]

    def receive(self, request_id, attempt_id):
        """读取指定尝试的清单与 K/V；接收方还必须验证、导入并发送确认。"""
        return self._request({
            "op": "receive", "request_id": request_id,
            "attempt_id": attempt_id,
        })

    def acknowledge(self, request_id, attempt_id, accepted, reason=None):
        """告知服务接收端是否接受交接；accepted=False 时携带回退原因。"""
        return self._request({
            "op": "ack", "request_id": request_id,
            "attempt_id": attempt_id, "accepted": bool(accepted),
            "reason": reason,
        })[0]

    def abort(self, request_id, attempt_id, reason):
        """将仍在等待或接收中的交接标记为回退，记录终止原因。"""
        return self._request({
            "op": "abort", "request_id": request_id,
            "attempt_id": attempt_id, "reason": reason,
        })[0]

    def release(self, request_id, attempt_id):
        """请求释放已经确认或回退的交接对象，不能在接收确认之前释放。"""
        return self._request({
            "op": "release", "request_id": request_id,
            "attempt_id": attempt_id,
        })[0]

    def stats(self):
        """读取服务中剩余交接对象及发布、接收、回退等计数。"""
        return self._request({"op": "stats"})[0]

    def shutdown(self):
        """请求关闭本次实验的交接服务。"""
        return self._request({"op": "shutdown"})[0]
