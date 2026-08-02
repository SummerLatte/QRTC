"""
数据层（Data Layer）

负责符号块与字节块之间的双向转换：多符号联合打包、RS 纠错、CRC16 完整性校验。

上层：传输层
- 编码方向：接收 byte 块（长度恒为 L_max），打包为符号块
- 解码方向：交付 byte 块（RS 纠错后）

下层：符号层
- 编码方向：交付符号块（恰好 S 个符号，每个值 0 到 M-1）
- 解码方向：接收符号块（S 个）+ 置信度（erasure 标记）
"""

from .packing import PACKING_TABLE, select_packing, pack_bytes_to_symbols, unpack_symbols_to_bytes
from .rs import rs_encode_block, rs_decode_block
from .crc import crc16, crc16_check, crc16_append, crc16_verify
from .codec import DataCodec, RS_NSYM, RS_MAX_DATA, RS_BLOCK_SIZE

__all__ = [
    # packing
    "PACKING_TABLE", "select_packing", "pack_bytes_to_symbols", "unpack_symbols_to_bytes",
    # rs
    "rs_encode_block", "rs_decode_block",
    # crc
    "crc16", "crc16_check", "crc16_append", "crc16_verify",
    # codec
    "DataCodec", "RS_NSYM", "RS_MAX_DATA", "RS_BLOCK_SIZE",
]
