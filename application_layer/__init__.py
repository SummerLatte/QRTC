"""
应用层（Application Layer）

负责定义 content 的内部结构（文件名、内容类型等元信息如何与实际数据一起编码）。

content 格式：
    | version (1) | content_type (1) | filename_len (1) | filename | data |

- version: 0x01
- content_type: 0x00=文件, 0x01=文本
- filename: UTF-8，不含路径
- data: 剩余全部，长度 = total_length - 3 - filename_len
"""

from dataclasses import dataclass
from typing import Optional


VERSION = 0x01
CONTENT_TYPE_FILE = 0x00
CONTENT_TYPE_TEXT = 0x01


@dataclass
class AppMessage:
    """解码后的应用层消息。"""
    version: int
    content_type: int
    filename: str
    data: bytes

    @property
    def is_file(self) -> bool:
        return self.content_type == CONTENT_TYPE_FILE

    @property
    def is_text(self) -> bool:
        return self.content_type == CONTENT_TYPE_TEXT

    @property
    def text(self) -> str:
        return self.data.decode("utf-8")


def app_encode(data: bytes, content_type: int = CONTENT_TYPE_FILE,
               filename: str = "") -> bytes:
    """编码 content。

    Args:
        data: 原始文件/文本内容
        content_type: 0x00=文件, 0x01=文本
        filename: 文件名（文件时必填，文本时可为空）
    Returns:
        content bytes
    """
    fn_bytes = filename.encode("utf-8")
    fn_len = len(fn_bytes)
    if fn_len > 255:
        raise ValueError(f"文件名过长: {fn_len} bytes, 最大 255")
    return bytes([VERSION, content_type, fn_len]) + fn_bytes + data


def app_decode(content: bytes) -> AppMessage:
    """解码 content。

    Args:
        content: 从传输层收到的完整 content
    Returns:
        AppMessage
    Raises:
        ValueError: 未知版本或格式错误
    """
    if len(content) < 3:
        raise ValueError(f"content 过短: {len(content)} bytes")

    version = content[0]
    if version != VERSION:
        raise ValueError(f"未知版本: {version:#04x}, 期望 {VERSION:#04x}")

    content_type = content[1]
    fn_len = content[2]
    if len(content) < 3 + fn_len:
        raise ValueError(f"filename 越界: 需要 {3 + fn_len} bytes, 实际 {len(content)}")

    filename = content[3:3 + fn_len].decode("utf-8")
    data = content[3 + fn_len:]

    return AppMessage(
        version=version,
        content_type=content_type,
        filename=filename,
        data=data,
    )


__all__ = [
    "VERSION", "CONTENT_TYPE_FILE", "CONTENT_TYPE_TEXT",
    "AppMessage", "app_encode", "app_decode",
]
