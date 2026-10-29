"""独立 CPU 外部 KV Cache Service。

服务进程只认识不可变 Chunk、字节容量和父子依赖，不导入模型或 CUDA。固定使用
leaf-LRU，淘汰策略不是本期实验变量。
"""

import argparse
import hashlib
import socketserver
import threading
from dataclasses import dataclass

from cache_protocol import receive_message, send_message


@dataclass
class CacheEntry:
    """保存不可变 K/V 字节及其逻辑身份、父依赖、形状、精度和最近访问顺序。"""
    digest: str
    parent_digest: str
    token_start: int
    token_count: int
    token_ids: tuple
    namespace_digest: str
    dtype: str
    shape: tuple
    checksum: str
    payload: bytes
    last_access: int


class CacheStore:
    """以固定字节容量管理外部 Chunk，使用锁保护并发请求下的元数据一致性。

    淘汰只选择无子节点的叶子，并遵守当前发布请求的祖先保护集合。"""
    def __init__(self, capacity_bytes):
        """建立条目索引、父子关系、容量计数和互斥锁，初始缓存为空。"""
        if capacity_bytes < 1:
            raise ValueError("capacity_bytes 必须大于 0")
        self.capacity_bytes = int(capacity_bytes)
        self.entries = {}
        self.children = {}
        self.clock = 0
        self.used_bytes = 0
        self.lookup_count = 0
        self.hit_chunks = 0
        self.load_count = 0
        self.store_count = 0
        self.duplicate_store_count = 0
        self.eviction_count = 0
        self.rejected_store_count = 0
        self.lock = threading.Lock()

    def _touch(self, entry):
        """推进逻辑访问时钟，记录条目最近使用次序，供 LRU 选择淘汰对象。"""
        self.clock += 1
        entry.last_access = self.clock

    def _remove(self, digest):
        """删除条目并同步维护字节占用和父子索引；调用方先保证它可被淘汰。"""
        entry = self.entries.pop(digest)
        self.used_bytes -= len(entry.payload)
        self.children.pop(digest, None)
        if entry.parent_digest:
            siblings = self.children.get(entry.parent_digest)
            if siblings is not None:
                siblings.discard(digest)
        self.eviction_count += 1

    def _make_room(self, required, protected):
        """逐个淘汰未受保护的最旧叶子，直到能容纳新内容；无可淘汰条目时返回 False。"""
        if required > self.capacity_bytes:
            return False
        while self.used_bytes + required > self.capacity_bytes:
            leaves = [
                entry for digest, entry in self.entries.items()
                if not self.children.get(digest) and digest not in protected
            ]
            if not leaves:
                return False
            self._remove(min(leaves, key=lambda entry: entry.last_access).digest)
        return True

    @staticmethod
    def _matches(entry, requested):
        """除摘要外再次比对父摘要、token 数及可选 token 内容，防止错误身份被当作命中。"""
        return (
            entry.digest == requested["digest"]
            and entry.parent_digest == requested.get("parent_digest", "")
            and entry.token_count == int(requested["token_count"])
            and (
                "token_ids" not in requested
                or entry.token_ids == tuple(int(value) for value in requested["token_ids"])
            )
        )

    def lookup(self, requested):
        """在锁内从根开始检查连续 Chunk 链，首个缺失或不匹配处停止并返回命中数。"""
        with self.lock:
            self.lookup_count += 1
            parent = ""
            count = 0
            for item in requested:
                entry = self.entries.get(item["digest"])
                if (
                    entry is None or not self._matches(entry, item)
                    or entry.parent_digest != parent
                ):
                    break
                self._touch(entry)
                parent = entry.digest
                count += 1
            self.hit_chunks += count
            return count

    def load(self, requested):
        """重新验证整条请求链与 payload 校验和，更新访问顺序后返回条目。

        先前 lookup 成功不保证此刻还存在，因此加载阶段不能跳过检查。"""
        with self.lock:
            entries = []
            parent = ""
            for item in requested:
                entry = self.entries.get(item["digest"])
                if (
                    entry is None or not self._matches(entry, item)
                    or entry.parent_digest != parent
                ):
                    raise KeyError("请求的 Chunk 链在 Load 前失效")
                if hashlib.sha256(entry.payload).hexdigest() != entry.checksum:
                    raise RuntimeError("服务端 Payload 校验和不一致")
                self._touch(entry)
                entries.append(entry)
                parent = entry.digest
            self.load_count += len(entries)
            return entries

    def store(self, request, payload):
        """校验 payload 与身份，处理重复发布并腾出容量，再登记不可变条目。

        父 Chunk 必须已存在；同一键对应不同内容会报错，容量不足返回 no_space，
        不会通过覆盖旧内容伪装为成功。"""
        checksum = hashlib.sha256(payload).hexdigest()
        if checksum != request["checksum"]:
            raise ValueError("Store Payload 校验和不一致")
        digest = request["digest"]
        parent = request.get("parent_digest", "")
        with self.lock:
            existing = self.entries.get(digest)
            if existing is not None:
                if (
                    existing.checksum != checksum
                    or existing.parent_digest != parent
                    or existing.token_ids != tuple(request["token_ids"])
                    or existing.namespace_digest != request["namespace_digest"]
                    or existing.dtype != request["dtype"]
                    or existing.shape != tuple(request["shape"])
                ):
                    raise RuntimeError("相同 CacheKey 对应了不同内容")
                self._touch(existing)
                self.duplicate_store_count += 1
                return "exists"
            if parent and parent not in self.entries:
                raise ValueError("父 Chunk 尚未存在，拒绝发布不可达子 Chunk")
            protected = set(request.get("protected_ancestors", []))
            if parent:
                protected.add(parent)
            if not self._make_room(len(payload), protected):
                self.rejected_store_count += 1
                return "no_space"
            self.clock += 1
            entry = CacheEntry(
                digest=digest,
                parent_digest=parent,
                token_start=int(request["token_start"]),
                token_count=int(request["token_count"]),
                token_ids=tuple(int(value) for value in request["token_ids"]),
                namespace_digest=request["namespace_digest"],
                dtype=request["dtype"],
                shape=tuple(int(value) for value in request["shape"]),
                checksum=checksum,
                payload=payload,
                last_access=self.clock,
            )
            self.entries[digest] = entry
            self.children.setdefault(digest, set())
            if parent:
                self.children.setdefault(parent, set()).add(digest)
            self.used_bytes += len(payload)
            self.store_count += 1
            return "stored"

    def clear(self):
        """在锁内清空条目和父子关系、归零当前占用；累计实验计数仍保留。"""
        with self.lock:
            self.entries.clear()
            self.children.clear()
            self.used_bytes = 0

    def stats(self):
        """在锁内生成容量、命中、发布与淘汰统计快照，避免读到一半更新的元数据。"""
        with self.lock:
            return {
                "capacity_bytes": self.capacity_bytes,
                "used_bytes": self.used_bytes,
                "entry_count": len(self.entries),
                "lookup_count": self.lookup_count,
                "hit_chunks": self.hit_chunks,
                "load_count": self.load_count,
                "store_count": self.store_count,
                "duplicate_store_count": self.duplicate_store_count,
                "eviction_count": self.eviction_count,
                "rejected_store_count": self.rejected_store_count,
                "digests": list(self.entries),
            }


class CacheRequestHandler(socketserver.BaseRequestHandler):
    """把一条网络请求分派到缓存存储操作，并统一编码成功响应或异常信息。"""
    def handle(self):
        """接收一次协议消息，按 op 调用 CacheStore，再发送元数据与可选二进制结果。"""
        try:
            request, payload = receive_message(self.request)
            operation = request.get("op")
            store = self.server.cache_store
            if operation == "ping":
                response, output = {"ok": True, "status": "ready"}, b""
            elif operation == "lookup":
                count = store.lookup(request.get("entries", []))
                response, output = {"ok": True, "hit_chunks": count}, b""
            elif operation == "load":
                entries = store.load(request.get("entries", []))
                response = {
                    "ok": True,
                    "entry_sizes": [len(entry.payload) for entry in entries],
                    "checksums": [entry.checksum for entry in entries],
                }
                output = b"".join(entry.payload for entry in entries)
            elif operation == "store":
                status = store.store(request, payload)
                response, output = {"ok": True, "status": status}, b""
            elif operation == "stats":
                response, output = {"ok": True, "stats": store.stats()}, b""
            elif operation == "clear":
                store.clear()
                response, output = {"ok": True}, b""
            elif operation == "shutdown":
                response, output = {"ok": True}, b""
                threading.Thread(
                    target=self.server.shutdown, daemon=True
                ).start()
            else:
                raise ValueError("未知操作: %r" % operation)
            send_message(self.request, response, output)
        except Exception as error:
            send_message(self.request, {
                "ok": False,
                "error": "%s: %s" % (type(error).__name__, error),
            })


class ThreadedCacheServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """为连接使用独立处理线程的 TCP 服务；共享缓存的数据一致性由 CacheStore 的锁保证。"""
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, capacity_bytes):
        """绑定监听地址和请求处理器，并创建本服务独享的固定容量缓存。"""
        super().__init__(address, CacheRequestHandler)
        self.cache_store = CacheStore(capacity_bytes)


def main():
    """解析监听地址和容量，启动 CPU 缓存服务；该进程只管理字节，不加载模型。"""
    parser = argparse.ArgumentParser(description="第 14 期外部 KV Cache Service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=65432)
    parser.add_argument("--capacity-mib", type=float, default=1024.0)
    args = parser.parse_args()
    capacity = int(args.capacity_mib * 1024 * 1024)
    with ThreadedCacheServer((args.host, args.port), capacity) as server:
        print(
            "cache_server_ready host=%s port=%d capacity_bytes=%d"
            % (args.host, args.port, capacity),
            flush=True,
        )
        server.serve_forever(poll_interval=0.1)


if __name__ == "__main__":
    main()
