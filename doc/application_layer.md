# 应用层协议规范（Application Layer）

应用层负责定义 content 的内部结构（文件名、内容类型等元信息如何与实际数据一起编码），并通过传输层的接口收发。应用层协议本身与 Cimbar 符号层/数据层/传输层无关，理论上可替换为任意应用协议。

**下层**：传输层（见 [transport_layer.md](transport_layer.md) 第 3 节）
- 编码方向：交付 content（byte 流）及其长度 total_length
- 解码方向：接收还原后的 content（长度为 total_length）

---

## 1. content 格式

### Specification

```
| version (1 byte) | content_type (1 byte) | filename_len (1 byte) | filename (filename_len bytes) | data (剩余全部) |
```

| 字段 | 长度 | 说明 |
|------|------|------|
| version | 1 | 协议版本号，当前为 0x01，用于未来格式变更时的兼容判断 |
| content_type | 1 | 0x00 = 文件, 0x01 = 文本 |
| filename_len | 1 | 文件名字节数（0~255），content_type=0x01 时通常为 0 |
| filename | filename_len | UTF-8 文件名，不含路径 |
| data | 剩余全部 | 实际文件/文本内容，长度 = total_length − 3 − filename_len |

- **不单独传输 data 的长度**：data 长度由传输层已知的 total_length 减去头部固定开销和 filename_len 推导得到，无需冗余字段
- 偏移和长度单位均为 byte

### Proposal

- data 长度复用 total_length 推导而非单独传输 filesize 字段：total_length 本就是传输层原生提供的信息，重复传输是浪费
- filename_len 用 1 byte（最长 255）：文件名场景足够，若需要更长可将来提升 version 兼容处理
- version 字段预留将来扩展（如加密、压缩标记等），当前解析器遇到未知 version 应拒绝解析而不是硬解，避免静默出错

---

## 2. 编码过程

```
1. 应用准备好 data（原始文件/文本 bytes）、content_type、filename（文件时必填，文本时可为空）
2. content = version(1) + content_type(1) + len(filename)(1) + filename + data
3. total_length = len(content)
4. 交付 (content, total_length) 给传输层
```

## 3. 解码过程

```
1. 从传输层收到 (content, total_length)
2. 解析 version，若不识别则拒绝
3. 解析 content_type、filename_len、filename
4. data = content[3+filename_len : total_length]
5. 按 content_type 交付给上层（文件落盘为 filename，或文本直接展示）
```
