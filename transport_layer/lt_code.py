"""
LT 喷泉码：编码 + peeling 解码。

编码：content → K 个源块 → 逐帧 XOR 编码。
解码：收到的帧 → peeling decoder → 还原 K 个源块 → 截断至 total_length。
"""

import struct
from typing import List, Optional, Tuple

from .prng import derive_seed, xorshift32
from .rsd import robust_soliton_cdf, sample_degree, sample_indices


def split_blocks(content: bytes, chunk_size: int) -> List[bytes]:
    """将 content 按 chunk_size 切成 K 块，末块 zero-pad。"""
    K = (len(content) + chunk_size - 1) // chunk_size
    if K == 0:
        K = 1
    blocks = []
    for i in range(K):
        chunk = content[i * chunk_size:(i + 1) * chunk_size]
        if len(chunk) < chunk_size:
            chunk = chunk + b"\x00" * (chunk_size - len(chunk))
        blocks.append(chunk)
    return blocks


def lt_encode_block(blocks: List[bytes], seq: int, total_length: int,
                    chunk_size: int) -> bytes:
    """编码单个 LT 帧（payload 部分）。

    返回 payload（chunk_size bytes）。
    """
    K = len(blocks)
    cdf = robust_soliton_cdf(K)
    degree, indices = _derive(seq, total_length, K, cdf)

    payload = bytearray(chunk_size)
    for idx in indices:
        block = blocks[idx]
        for i in range(chunk_size):
            payload[i] ^= block[i]
    return bytes(payload)


def lt_encode_frame(blocks: List[bytes], seq: int, total_length: int,
                    chunk_size: int, L_max: int) -> bytes:
    """编码完整的 LT 帧：total_length(4) + seq(4) + payload。"""
    payload = lt_encode_block(blocks, seq, total_length, chunk_size)
    frame = struct.pack(">II", total_length, seq) + payload
    assert len(frame) == L_max, f"帧长度 {len(frame)} != L_max={L_max}"
    return frame


def lt_encode(content: bytes, L_max: int, num_frames: int) -> Tuple[List[bytes], int, int]:
    """LT 喷泉码编码。

    Args:
        content: 原始内容
        L_max: 数据层容量（帧长度）
        num_frames: 生成帧数
    Returns:
        (frames, K, chunk_size)
    """
    total_length = len(content)
    chunk_size = L_max - 8
    if chunk_size <= 0:
        raise ValueError(f"L_max={L_max} 太小，chunk_size={chunk_size}")

    blocks = split_blocks(content, chunk_size)
    K = len(blocks)

    frames = []
    for seq in range(num_frames):
        frame = lt_encode_frame(blocks, seq, total_length, chunk_size, L_max)
        frames.append(frame)

    return frames, K, chunk_size


class LTDecoder:
    """LT peeling decoder。

    增量接收帧，逐步还原源块。
    """

    def __init__(self, total_length: int, K: int, chunk_size: int) -> None:
        self.total_length = total_length
        self.K = K
        self.chunk_size = chunk_size
        self.cdf = robust_soliton_cdf(K)
        self.decoded: List[Optional[bytes]] = [None] * K
        self.received: List[Tuple[List[int], bytearray]] = []
        self.seen_seqs: set = set()

    def add_frame(self, frame: bytes) -> bool:
        """接收一帧，返回是否为新帧（True=新，False=重复/无效）。"""
        if len(frame) < 8:
            return False

        tl, seq = struct.unpack(">II", frame[:8])
        if tl != self.total_length:
            return False

        if seq in self.seen_seqs:
            return False
        self.seen_seqs.add(seq)

        payload = bytearray(frame[8:8 + self.chunk_size])
        if len(payload) < self.chunk_size:
            payload.extend(b"\x00" * (self.chunk_size - len(payload)))

        degree, indices = _derive(seq, self.total_length, self.K, self.cdf)
        self.received.append((indices, payload))
        return True

    def decode(self) -> Optional[bytes]:
        """Peeling decoder：尝试还原所有源块。

        成功返回 content（截断至 total_length），失败返回 None。
        """
        progress = True
        while progress:
            progress = False
            for indices, payload in self.received:
                unknown = [(i, idx) for i, idx in enumerate(indices)
                           if self.decoded[idx] is None]
                if len(unknown) == 0:
                    continue
                if len(unknown) == 1:
                    pos, target_idx = unknown[0]
                    result = bytearray(payload)
                    for i, idx in enumerate(indices):
                        if i != pos:
                            block = self.decoded[idx]
                            if block is not None:
                                for j in range(self.chunk_size):
                                    result[j] ^= block[j]
                    self.decoded[target_idx] = bytes(result)
                    progress = True

        if any(d is None for d in self.decoded):
            return None

        content = b"".join(self.decoded)
        return content[:self.total_length]

    @property
    def decoded_count(self) -> int:
        return sum(1 for d in self.decoded if d is not None)

    @property
    def is_complete(self) -> bool:
        return all(d is not None for d in self.decoded)


def _derive(seq: int, total_length: int, K: int,
            cdf: List[float]) -> Tuple[int, List[int]]:
    """从 (seq, total_length) 推导 (degree, indices)。"""
    seed = derive_seed(seq, total_length)
    rng = xorshift32(seed)
    degree = sample_degree(rng, K, cdf)
    indices = sample_indices(rng, K, degree)
    return degree, indices
