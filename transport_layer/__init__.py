"""
传输层（Transport Layer）

负责将字节流用 LT 喷泉码切片、编码为帧、并在接收端重组恢复。

上层：应用层
- 编码方向：接收 content（byte 流）及其长度 total_length
- 解码方向：交付恢复的 content（长度为 total_length）

下层：数据层
- 编码方向：交付 byte 块（长度恒为 L_max）
- 解码方向：接收 byte 块（RS 纠错后）
"""

from .prng import mix32, derive_seed, xorshift32
from .rsd import robust_soliton_cdf, sample_degree, sample_indices
from .lt_code import LTDecoder, lt_encode, lt_encode_block, lt_encode_frame, split_blocks
from .codec import TransportCodec

__all__ = [
    # prng
    "mix32", "derive_seed", "xorshift32",
    # rsd
    "robust_soliton_cdf", "sample_degree", "sample_indices",
    # lt_code
    "LTDecoder", "lt_encode", "lt_encode_block", "lt_encode_frame", "split_blocks",
    # codec
    "TransportCodec",
]
