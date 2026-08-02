"""
TransportCodec：传输层主类，串联帧结构 + LT 喷泉码 + 去重。

编码方向：content → LT 编码 → 帧列表（每帧 L_max bytes）
解码方向：帧列表 → LT 解码 → content
"""

import struct
from typing import List, Optional, Tuple

from .lt_code import LTDecoder, lt_encode, split_blocks
from .prng import derive_seed


class TransportCodec:
    """传输层编解码器。

    一次传输过程中 L_max 恒定。
    """

    def __init__(self, L_max: int) -> None:
        if L_max < 9:
            raise ValueError(f"L_max={L_max} 太小，至少需要 9 bytes")
        self.L_max = L_max
        self.chunk_size = L_max - 8  # 减去 total_length(4) + seq(4)

    def encode(self, content: bytes, num_frames: int) -> List[bytes]:
        """编码：content → num_frames 个帧（每个 L_max bytes）。

        Args:
            content: 原始内容
            num_frames: 生成的帧数（>= K）
        Returns:
            帧列表，每帧长度 L_max
        """
        frames, K, _ = lt_encode(content, self.L_max, num_frames)
        return frames

    def decode(self, frames: List[bytes]) -> Optional[bytes]:
        """解码：从帧列表恢复 content。

        Args:
            frames: 收到的帧列表
        Returns:
            成功返回 content（长度 total_length），失败返回 None
        """
        if not frames:
            return None

        # 从第一帧解析 total_length
        first_frame = frames[0]
        if len(first_frame) < 8:
            return None

        total_length, _ = struct.unpack(">II", first_frame[:8])
        if total_length <= 0:
            return b""

        K = (total_length + self.chunk_size - 1) // self.chunk_size
        if K == 0:
            K = 1

        decoder = LTDecoder(total_length, K, self.chunk_size)
        for frame in frames:
            decoder.add_frame(frame)

        return decoder.decode()

    def get_params(self, total_length: int) -> Tuple[int, int]:
        """根据 total_length 推导 (K, chunk_size)。"""
        K = (total_length + self.chunk_size - 1) // self.chunk_size
        if K == 0:
            K = 1
        return K, self.chunk_size
