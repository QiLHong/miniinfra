"""只保存单次请求交接对象的本机 Handoff Service。"""

import argparse
import socketserver
import threading
import time

from handoff_protocol import payload_digest, recv_message, send_message


class HandoffStore:
    """管理一次请求交接的状态机，KV 对象在 ACK 或回退前保持可用。

    典型顺序为 kv_ready→received→acknowledged→释放；失败尝试转为 fallback。"""
    def __init__(self):
        """创建交接索引、互斥锁和生命周期计数。"""
        self.lock = threading.Lock()
        self.entries = {}
        self.counters = {
            "published": 0, "received": 0, "accepted": 0,
            "fallback": 0, "released": 0,
        }

    @staticmethod
    def key(request_id, attempt_id):
        """将请求 ID 与尝试 ID 共同组成键，避免同一请求的不同尝试互相覆盖。"""
        return str(request_id), str(attempt_id)

    def publish(self, manifest, payload):
        """核对清单必需字段、字节数和摘要后，登记为 kv_ready；拒绝重复发布相同尝试。"""
        required = (
            "request_id", "attempt_id", "namespace_digest", "payload_sha256",
            "payload_bytes", "prompt_tokens", "first_token",
        )
        missing = [key for key in required if key not in manifest]
        if missing:
            raise ValueError("Manifest 缺少字段: " + ",".join(missing))
        if int(manifest["payload_bytes"]) != len(payload):
            raise ValueError("Manifest 与实际 Payload 字节数不一致")
        if payload_digest(payload) != manifest["payload_sha256"]:
            raise ValueError("发布时 Payload SHA-256 校验失败")
        key = self.key(manifest["request_id"], manifest["attempt_id"])
        with self.lock:
            if key in self.entries:
                raise ValueError("同一 Request Attempt 已经发布")
            self.entries[key] = {
                "manifest": dict(manifest), "payload": payload,
                "state": "kv_ready", "reason": None,
                "published_ns": time.monotonic_ns(),
            }
            self.counters["published"] += 1
        return {"status": "published", "state": "kv_ready"}

    def status(self, request_id, attempt_id):
        """在锁内读取尝试状态和原因，不存在时明确返回 missing。"""
        key = self.key(request_id, attempt_id)
        with self.lock:
            entry = self.entries.get(key)
            if entry is None:
                return {"status": "missing", "state": "missing"}
            return {
                "status": "ok", "state": entry["state"],
                "reason": entry["reason"],
            }

    def receive(self, request_id, attempt_id):
        """只允许读取就绪或已接收的对象，并将状态记录为 received；尚不释放 payload。"""
        key = self.key(request_id, attempt_id)
        with self.lock:
            entry = self.entries.get(key)
            if entry is None:
                raise KeyError("请求交接对象不存在")
            if entry["state"] not in ("kv_ready", "received"):
                raise ValueError("当前状态不允许 Receive: " + entry["state"])
            entry["state"] = "received"
            self.counters["received"] += 1
            return dict(entry["manifest"]), entry["payload"]

    def acknowledge(self, request_id, attempt_id, accepted, reason):
        """只对 received 对象处理确认，按是否接受切换到 acknowledged 或 fallback。"""
        key = self.key(request_id, attempt_id)
        with self.lock:
            entry = self.entries.get(key)
            if entry is None:
                raise KeyError("请求交接对象不存在")
            if entry["state"] != "received":
                raise ValueError("只有 received 状态可以确认")
            entry["state"] = "acknowledged" if accepted else "fallback"
            entry["reason"] = reason
            self.counters["accepted" if accepted else "fallback"] += 1
            return {"status": "ok", "state": entry["state"]}

    def abort(self, request_id, attempt_id, reason):
        """将仍可取消的交接转为 fallback，保留失败原因供发送方处理。"""
        key = self.key(request_id, attempt_id)
        with self.lock:
            entry = self.entries.get(key)
            if entry is None:
                raise KeyError("请求交接对象不存在")
            if entry["state"] not in ("kv_ready", "received"):
                raise ValueError("当前状态不允许 Abort: " + entry["state"])
            entry["state"] = "fallback"
            entry["reason"] = reason
            self.counters["fallback"] += 1
            return {"status": "ok", "state": "fallback"}

    def release(self, request_id, attempt_id):
        """仅在确认或回退后删除对象；不存在返回 False，提前释放则报错。"""
        key = self.key(request_id, attempt_id)
        with self.lock:
            entry = self.entries.get(key)
            if entry is None:
                return False
            if entry["state"] not in ("acknowledged", "fallback"):
                raise ValueError("ACK 前不能释放交接对象")
            del self.entries[key]
            self.counters["released"] += 1
            return True

    def stats(self):
        """返回当前对象数、状态及各阶段累计计数，供验证资源最终是否释放。"""
        with self.lock:
            return {
                "status": "ok", "entries": len(self.entries),
                "states": [entry["state"] for entry in self.entries.values()],
                **self.counters,
            }


class Handler(socketserver.BaseRequestHandler):
    """将 TCP 协议操作映射到交接状态机，负责返回清单、payload 或错误。"""
    def handle(self):
        """处理一次交接请求；服务关闭放到另一线程，避免请求处理线程等待自身退出。"""
        try:
            header, payload = recv_message(self.request)
            op = header.get("op")
            store = self.server.store
            if op == "ping":
                response, output = {"status": "ok"}, b""
            elif op == "publish":
                response, output = store.publish(header["manifest"], payload), b""
            elif op == "status":
                response, output = store.status(
                    header["request_id"], header["attempt_id"]
                ), b""
            elif op == "receive":
                manifest, output = store.receive(
                    header["request_id"], header["attempt_id"]
                )
                response = {"status": "ok", "manifest": manifest}
            elif op == "ack":
                response = store.acknowledge(
                    header["request_id"], header["attempt_id"],
                    header["accepted"], header.get("reason"),
                )
                output = b""
            elif op == "abort":
                response = store.abort(
                    header["request_id"], header["attempt_id"],
                    header.get("reason", "Handoff 被中止"),
                )
                output = b""
            elif op == "release":
                released = store.release(
                    header["request_id"], header["attempt_id"]
                )
                response, output = {"status": "ok", "released": released}, b""
            elif op == "stats":
                response, output = store.stats(), b""
            elif op == "shutdown":
                response, output = {"status": "ok"}, b""
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                raise ValueError("未知操作: %s" % op)
            send_message(self.request, response, output)
        except Exception as error:
            send_message(self.request, {
                "status": "error", "error": "%s: %s" % (
                    type(error).__name__, error
                )
            })


class Server(socketserver.ThreadingTCPServer):
    """允许复用监听地址的多线程 TCP 服务，处理线程作为守护线程随进程退出。"""
    allow_reuse_address = True
    daemon_threads = True


def main():
    """创建交接存储并监听指定端口，直到收到关闭请求或进程结束。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    with Server((args.host, args.port), Handler) as server:
        server.store = HandoffStore()
        server.serve_forever()


if __name__ == "__main__":
    main()
