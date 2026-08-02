"""
QRTC Cimbar 解码器 GUI

功能：
- 选择屏幕区域，实时录屏读取 Cimbar 帧
- 自动检测网格等级、颜色档、图形档（或手动指定）
- 喷泉码增量解码，收到足够帧后还原原始内容
- 支持文本和文件内容还原

用法：
    python decoder_app.py

依赖：
    pip install mss          # 快速屏幕捕获（可选，有 PIL 后备）
    pip install opencv-python numpy pillow
"""

from __future__ import annotations

import contextlib
import io
import os
import queue
import struct
import sys
import threading
import time
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Dict, Optional, Tuple

from PIL import Image, ImageTk
import numpy as np

from application_layer import app_decode, CONTENT_TYPE_TEXT, CONTENT_TYPE_FILE
from data_layer import DataCodec
from symbol_layer import (
    ColorPalette,
    Grid,
    ShapeSet,
    SymbolDecoder,
    OpenCVSampler,
)
from transport_layer.lt_code import LTDecoder

# 屏幕捕获：优先 mss（快 ~10x），后备 PIL.ImageGrab
try:
    import mss
    _HAS_MSS = True
except ImportError:
    _HAS_MSS = False
    from PIL import ImageGrab


@contextlib.contextmanager
def _capture_stdout():
    """捕获 stdout（OpenCVSampler 内有 print 调试语句），返回捕获内容。"""
    old = sys.stdout
    buf = io.StringIO()
    sys.stdout = buf
    try:
        yield buf
    finally:
        sys.stdout = old


# ---------------------------------------------------------------------------
# 屏幕区域选择器
# ---------------------------------------------------------------------------

class RegionSelector:
    """全屏半透明覆盖窗口，拖拽选择屏幕区域。"""

    def __init__(self, root: tk.Tk, callback) -> None:
        self.callback = callback
        self.start_x = 0
        self.start_y = 0
        self.rect_id = None

        self.top = tk.Toplevel(root)
        self.top.attributes("-fullscreen", True)
        self.top.attributes("-alpha", 0.25)
        self.top.attributes("-topmost", True)
        self.top.configure(bg="black")
        self.top.title("选择区域")

        self.canvas = tk.Canvas(
            self.top, cursor="cross", bg="black", highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.label = tk.Label(
            self.top,
            text="拖拽选择要捕获的区域 · ESC 取消",
            fg="white", bg="black", font=("", 18),
        )
        self.label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.top.bind("<Escape>", lambda e: self._cancel())
        self.top.focus_force()

    def _on_press(self, event):
        self.label.place_forget()
        self.start_x = event.x
        self.start_y = event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="red", width=3,
        )

    def _on_drag(self, event):
        self.canvas.coords(
            self.rect_id, self.start_x, self.start_y, event.x, event.y,
        )

    def _on_release(self, event):
        x1, y1 = min(self.start_x, event.x), min(self.start_y, event.y)
        x2, y2 = max(self.start_x, event.x), max(self.start_y, event.y)
        self.top.destroy()
        if x2 - x1 > 20 and y2 - y1 > 20:
            self.callback((x1, y1, x2, y2))

    def _cancel(self):
        self.top.destroy()


# ---------------------------------------------------------------------------
# 解码器主应用
# ---------------------------------------------------------------------------

class CimbarDecoderApp:
    """Cimbar 解码器主窗口。"""

    MAX_PREVIEW_SIZE = 400

    # 自动检测时尝试的参数组合（按常见度排序）
    _ALL_COMBOS = [
        (15, 8, 4), (15, 4, 4), (14, 8, 4), (14, 4, 4),
        (13, 8, 4), (13, 4, 4), (12, 8, 4), (12, 4, 4),
        (11, 8, 4), (11, 4, 4), (10, 8, 4), (10, 4, 4),
        (9, 8, 4), (9, 4, 4), (8, 8, 4), (8, 4, 4),
        (7, 8, 4), (7, 4, 4), (6, 8, 4), (6, 4, 4),
        (5, 8, 4), (5, 4, 4), (4, 8, 4), (4, 4, 4),
        (3, 8, 4), (3, 4, 4), (2, 8, 4), (2, 4, 4),
        (1, 8, 4), (1, 4, 4),
        (15, 2, 4), (14, 2, 4), (13, 2, 4), (12, 2, 4), (11, 2, 4), (10, 2, 4),
        (9, 2, 4), (8, 2, 4), (7, 2, 4), (6, 2, 4), (5, 2, 4), (4, 2, 4), (3, 2, 4), (2, 2, 4), (1, 2, 4),
        (15, 8, 2), (14, 8, 2), (13, 8, 2), (12, 8, 2), (11, 8, 2), (10, 8, 2),
        (9, 8, 2), (8, 8, 2), (7, 8, 2), (6, 8, 2), (5, 8, 2), (4, 8, 2), (3, 8, 2), (2, 8, 2), (1, 8, 2),
        (15, 4, 2), (14, 4, 2), (13, 4, 2), (12, 4, 2), (11, 4, 2), (10, 4, 2),
        (9, 4, 2), (8, 4, 2), (7, 4, 2), (6, 4, 2), (5, 4, 2), (4, 4, 2), (3, 4, 2), (2, 4, 2), (1, 4, 2),
        (15, 2, 2), (14, 2, 2), (13, 2, 2), (12, 2, 2), (11, 2, 2), (10, 2, 2),
        (9, 2, 2), (8, 2, 2), (7, 2, 2), (6, 2, 2), (5, 2, 2), (4, 2, 2), (3, 2, 2), (2, 2, 2), (1, 2, 2),
    ]

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("QRTC Cimbar 解码器")
        self.root.minsize(720, 660)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # --- 状态 ---
        self.region: Optional[Tuple[int, int, int, int]] = None
        self.capturing = False
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._ui_queue: queue.Queue = queue.Queue()

        # 解码器状态
        self._lt_decoder: Optional[LTDecoder] = None
        self._detected_params: Optional[Dict] = None
        self._frame_count = 0
        self._decoded_count = 0
        self._last_msg = None

        # 速率统计
        self._rate_bytes = 0          # 当前窗口内有效帧字节数
        self._rate_window_start = 0.0  # 窗口起始时间
        self._rate_kbps = 0.0

        # 传输统计
        self._transfer_start = 0.0     # 首帧时间
        self._total_valid_bytes = 0    # 有效帧总字节数

        # 屏幕捕获
        self._sct = None
        if _HAS_MSS:
            self._sct = mss.mss()

        self.photo_image: Optional[ImageTk.PhotoImage] = None

        # 预览限速
        self._last_preview_ts = 0.0

        self._build_ui()
        self.root.after(50, self._poll_ui_queue)

    # ---- UI 构建 ----

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # 配置面板
        config = ttk.LabelFrame(main, text="配置", padding=10)
        config.pack(fill=tk.X, pady=(0, 8))
        self._build_config(config)

        # 控制按钮
        ctrl = ttk.Frame(main)
        ctrl.pack(fill=tk.X, pady=(0, 8))
        self._build_controls(ctrl)

        # 进度条 + 速率
        prog_row = ttk.Frame(main)
        prog_row.pack(fill=tk.X, pady=(0, 4))
        self.progress = ttk.Progressbar(prog_row, mode="determinate")
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.var_rate = tk.StringVar(value="0.0 kB/s")
        ttk.Label(prog_row, textvariable=self.var_rate, width=14,
                  anchor=tk.E).pack(side=tk.RIGHT, padx=(8, 0))
        self.var_progress = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.var_progress).pack(anchor=tk.W)

        # 预览 + 结果 并排
        bottom = ttk.Frame(main)
        bottom.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        # 捕获预览
        preview = ttk.LabelFrame(bottom, text="捕获预览", padding=10)
        preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_label = ttk.Label(
            preview, text="未开始", anchor=tk.CENTER, font=("", 14),
        )
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # 解码结果
        result = ttk.LabelFrame(bottom, text="解码结果", padding=10)
        result.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.text_result = tk.Text(result, height=10, width=40, state=tk.DISABLED)
        self.text_result.pack(fill=tk.BOTH, expand=True)
        result_btns = ttk.Frame(result)
        result_btns.pack(fill=tk.X, pady=(4, 0))
        self.btn_save_result = ttk.Button(
            result_btns, text="保存内容", command=self._save_result,
            state=tk.DISABLED,
        )
        self.btn_save_result.pack(side=tk.LEFT)
        self.btn_copy = ttk.Button(
            result_btns, text="复制文本", command=self._copy_text,
            state=tk.DISABLED,
        )
        self.btn_copy.pack(side=tk.LEFT, padx=(8, 0))

        # 调试日志
        log_frame = ttk.LabelFrame(main, text="调试日志", padding=4)
        log_frame.pack(fill=tk.BOTH, expand=False, pady=(8, 0))
        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=6, state=tk.DISABLED, font=("Consolas", 9),
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # 状态栏
        self.var_status = tk.StringVar(value="就绪")
        ttk.Label(
            main, textvariable=self.var_status, relief=tk.SUNKEN, anchor=tk.W,
        ).pack(fill=tk.X, pady=(4, 0))

    def _build_config(self, parent: ttk.Widget) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X)

        ttk.Label(row, text="网格等级:").pack(side=tk.LEFT)
        self.var_level = tk.StringVar(value="15")
        ttk.OptionMenu(
            row, self.var_level, "15",
            "auto", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "10", "11", "12", "13", "14", "15",
        ).pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(row, text="颜色档:").pack(side=tk.LEFT)
        self.var_colors = tk.StringVar(value="8")
        ttk.OptionMenu(
            row, self.var_colors, "8", "auto", "2", "4", "8",
        ).pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(row, text="图形档:").pack(side=tk.LEFT)
        self.var_shapes = tk.StringVar(value="4")
        ttk.OptionMenu(
            row, self.var_shapes, "4", "auto", "2", "4",
        ).pack(side=tk.LEFT, padx=(4, 16))


    def _build_controls(self, parent: ttk.Widget) -> None:
        self.btn_select = ttk.Button(
            parent, text="选择区域", command=self._select_region,
        )
        self.btn_select.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_start = ttk.Button(
            parent, text="▶ 开始捕获", command=self._start_capture,
            state=tk.DISABLED,
        )
        self.btn_start.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_stop = ttk.Button(
            parent, text="⏹ 停止", command=self._stop_capture,
            state=tk.DISABLED,
        )
        self.btn_stop.pack(side=tk.LEFT, padx=(0, 16))

        self.btn_reset = ttk.Button(
            parent, text="重置解码", command=self._reset_decoder,
        )
        self.btn_reset.pack(side=tk.LEFT)

    # ---- 区域选择 ----

    def _select_region(self) -> None:
        self.root.iconify()
        self.root.after(
            300, lambda: RegionSelector(self.root, self._on_region_selected),
        )

    def _on_region_selected(self, region: tuple) -> None:
        self.region = region
        self.root.deiconify()
        self.btn_start.config(state=tk.NORMAL)
        x1, y1, x2, y2 = region
        self.var_status.set(
            f"已选区域: ({x1},{y1})-({x2},{y2})  {x2 - x1}×{y2 - y1}",
        )

    # ---- 捕获控制 ----

    def _start_capture(self) -> None:
        if self.region is None:
            messagebox.showerror("错误", "请先选择捕获区域")
            return

        self.capturing = True
        self._stop_event.clear()
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_select.config(state=tk.DISABLED)

        self._capture_thread = threading.Thread(
            target=self._capture_worker, daemon=True,
        )
        self._capture_thread.start()
        self.var_status.set("捕获中...")

    def _stop_capture(self) -> None:
        self.capturing = False
        self._stop_event.set()
        if self._capture_thread:
            self._capture_thread.join(timeout=3.0)
            self._capture_thread = None
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_select.config(state=tk.NORMAL)
        self.var_status.set("已停止")

    def _reset_decoder(self) -> None:
        self._lt_decoder = None
        self._detected_params = None
        self._frame_count = 0
        self._decoded_count = 0
        self._last_msg = None
        self._rate_bytes = 0
        self._rate_window_start = 0.0
        self._rate_kbps = 0.0
        self._transfer_start = 0.0
        self._total_valid_bytes = 0
        self.var_rate.set("0.0 kB/s")
        self.progress["value"] = 0
        self.var_progress.set("")
        self.text_result.config(state=tk.NORMAL)
        self.text_result.delete("1.0", tk.END)
        self.text_result.config(state=tk.DISABLED)
        self.btn_save_result.config(state=tk.DISABLED)
        self.btn_copy.config(state=tk.DISABLED)
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.var_status.set("解码器已重置")

    def _on_close(self) -> None:
        self._stop_event.set()
        self.capturing = False
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        if self._sct:
            self._sct.close()
        self.root.destroy()

    # ---- 捕获 + 解码线程 ----

    def _capture_worker(self) -> None:
        """后台线程：全速捕获屏幕区域并尝试解码。"""
        region = self.region
        capture_idx = 0

        while not self._stop_event.is_set():
            capture_idx += 1
            t0 = time.time()

            # 1. 捕获屏幕区域（双重捕获防撕裂）
            try:
                img = self._capture_region(region)
                if img is not None:
                    # 立即再抓一次，比较两次是否一致
                    img2 = self._capture_region(region)
                    if img2 is not None:
                        arr1 = np.asarray(img)
                        arr2 = np.asarray(img2)
                        if not np.array_equal(arr1, arr2):
                            # 两次不同 = 正在换帧（撕裂），跳过
                            continue
            except Exception as e:
                self._ui_queue.put(("log", f"[!] 捕获异常: {e}"))
                continue
            if img is None:
                continue

            # 2. 限速预览（~10fps 预览足够，不影响捕获速度）
            now = time.time()
            if now - self._last_preview_ts > 0.1:
                self._last_preview_ts = now
                self._ui_queue.put(("preview", img))

            # 3. 尝试解码
            try:
                frame_bytes, params, fail_reason = self._decode_frame(img, capture_idx)
            except Exception as e:
                tb = traceback.format_exc()
                fail_reason = f"解码异常: {e}"
                self._ui_queue.put(("log", f"[!] #{capture_idx} {fail_reason}\n{tb}"))
                frame_bytes, params = None, None

            # 3b. 解码失败：保存原始帧图像
            if frame_bytes is None and fail_reason is not None:
                self._save_failed_frame(img, capture_idx, fail_reason)

            # 4. 处理解码结果
            if frame_bytes is not None:
                self._frame_count += 1
                tl, seq = struct.unpack(">II", frame_bytes[:8])

                # 速率统计：记录有效帧字节数
                if self._rate_window_start == 0.0:
                    self._rate_window_start = time.time()
                if self._transfer_start == 0.0:
                    self._transfer_start = time.time()
                self._rate_bytes += len(frame_bytes)
                self._total_valid_bytes += len(frame_bytes)
                self._ui_queue.put(("log",
                    f"[OK] 帧 #{capture_idx} 解码成功: seq={seq} total_length={tl} "
                    f"({len(frame_bytes)}B) 耗时={time.time()-t0:.1f}ms"))

                # 首帧：初始化 LT 解码器
                if self._lt_decoder is None and params is not None:
                    total_length = tl
                    if total_length <= 0:
                        self._ui_queue.put(("log", f"[!] total_length={total_length} 无效"))
                        continue
                    L_max = params["L_max"]
                    chunk_size = L_max - 8
                    K = max(1, (total_length + chunk_size - 1) // chunk_size)
                    self._lt_decoder = LTDecoder(total_length, K, chunk_size)
                    self._detected_params = params
                    self._ui_queue.put(("params", params, K, total_length))
                    self._ui_queue.put(("log",
                        f"[*] LT 解码器初始化: K={K} chunk_size={chunk_size} "
                        f"L_max={L_max} total_length={total_length}"))

                # 添加帧到 LT 解码器
                if self._lt_decoder is not None:
                    is_new = self._lt_decoder.add_frame(frame_bytes)
                    if is_new:
                        self._decoded_count += 1
                        decoded_count = self._lt_decoder.decoded_count
                        K = self._lt_decoder.K

                        # 计算速率：每秒更新一次
                        now = time.time()
                        elapsed = now - self._rate_window_start
                        if elapsed >= 1.0:
                            kbps = self._rate_bytes / elapsed / 1024
                            self._rate_kbps = kbps
                            self._rate_bytes = 0
                            self._rate_window_start = now
                            self._ui_queue.put(("rate", kbps))

                        self._ui_queue.put((
                            "progress", decoded_count, K, self._decoded_count,
                        ))

                        # 每 5 帧或接近完成时尝试完整解码
                        if self._decoded_count % 5 == 0 or decoded_count >= K:
                            content = self._lt_decoder.decode()
                            if content is not None:
                                transfer_time = time.time() - self._transfer_start
                                self._ui_queue.put(("log",
                                    f"[✓] 喷泉码解码完成! {len(content)}B "
                                    f"(有效帧 {self._decoded_count}, "
                                    f"耗时 {transfer_time:.1f}s)"))
                                try:
                                    msg = app_decode(content)
                                    self._last_msg = msg
                                    self._ui_queue.put(("decoded", msg))
                                except Exception as e:
                                    tb = traceback.format_exc()
                                    self._ui_queue.put(("log",
                                        f"[!] 应用层解码失败: {e}\n{tb}"))
                                # 解码完成，停止捕获
                                self._stop_event.set()
                                break
                            else:
                                self._ui_queue.put(("log",
                                    f"[..] peeling: {decoded_count}/{K} 源块 "
                                    f"(有效帧 {self._decoded_count})"))
                    else:
                        self._ui_queue.put(("log",
                            f"[dup] 帧 #{capture_idx} seq={seq} 重复或无效，跳过"))

        self._ui_queue.put(("stopped",))

    def _capture_region(self, region: tuple) -> Optional[Image.Image]:
        """捕获屏幕指定区域为 PIL Image。"""
        x1, y1, x2, y2 = region
        w, h = x2 - x1, y2 - y1
        if w < 10 or h < 10:
            return None

        if _HAS_MSS and self._sct:
            monitor = {"top": y1, "left": x1, "width": w, "height": h}
            shot = self._sct.grab(monitor)
            return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        else:
            return ImageGrab.grab(bbox=(x1, y1, x2, y2))

    # ---- 解码逻辑 ----

    def _decode_frame(
        self, img: Image.Image, capture_idx: int = 0,
    ) -> Tuple[Optional[bytes], Optional[Dict], Optional[str]]:
        """尝试从图像中解码一帧 Cimbar。返回 (frame_bytes, params, fail_reason)。

        成功时 fail_reason 为 None；失败时为描述字符串。
        """
        # 已检测到参数：直接使用
        if self._detected_params is not None:
            fb, p, reason = self._decode_with_params(img, self._detected_params, capture_idx)
            if fb is not None:
                return fb, p, None
            return None, None, reason or "已知参数解码失败"

        # 解析用户配置
        level = self.var_level.get()
        colors = self.var_colors.get()
        shapes = self.var_shapes.get()

        # 全部手动指定
        if level != "auto" and colors != "auto" and shapes != "auto":
            params = {
                "level": int(level),
                "n_colors": int(colors),
                "n_shapes": int(shapes),
            }
            fb, p, reason = self._decode_with_params(img, params, capture_idx)
            if fb is not None:
                return fb, p, None
            return None, None, reason or "手动参数解码失败"

        # 自动检测：遍历候选组合
        tried = 0
        fail_reasons = []
        for lv, nc, ns in self._ALL_COMBOS:
            if level != "auto" and int(level) != lv:
                continue
            if colors != "auto" and int(colors) != nc:
                continue
            if shapes != "auto" and int(shapes) != ns:
                continue

            tried += 1
            params = {"level": lv, "n_colors": nc, "n_shapes": ns}
            result, p, reason = self._decode_with_params(img, params, capture_idx, quiet=True)
            if result is not None:
                self._ui_queue.put(("log",
                    f"[det] #{capture_idx} 自动检测命中: L{lv} C{nc} S{ns} "
                    f"(尝试了 {tried} 种组合)"))
                return result, p, None
            if reason:
                fail_reasons.append(f"{reason}")

        # 汇总失败原因
        reason_summary = f"自动检测失败({tried}种组合)"
        if fail_reasons:
            # 取最常见的失败原因
            from collections import Counter
            common = Counter(fail_reasons).most_common(3)
            reason_summary += ": " + "; ".join(f"{r}({c}x)" for r, c in common)

        if capture_idx % 10 == 0:
            self._ui_queue.put(("log",
                f"[--] #{capture_idx} {reason_summary}"))
        return None, None, reason_summary

    def _decode_with_params(
        self, img: Image.Image, params: Dict,
        capture_idx: int = 0, quiet: bool = False,
    ) -> Tuple[Optional[bytes], Optional[Dict], Optional[str]]:
        """用指定参数尝试解码一帧。返回 (frame_bytes, result_params, fail_reason)。"""
        lv = params["level"]
        nc = params["n_colors"]
        ns = params["n_shapes"]
        tag = f"L{lv}C{nc}S{ns}"

        try:
            with _capture_stdout() as buf:
                g = Grid(lv)
                pal = ColorPalette(nc)
                shp = ShapeSet(ns)

                sampler = OpenCVSampler(img, g, pal, shp)
                dec = SymbolDecoder(sampler, g)
                decoded = dec.decode()

            # OpenCVSampler 的 print 输出
            cv_log = buf.getvalue().strip()
            if cv_log and not quiet:
                self._ui_queue.put(("log", f"[cv] {tag}: {cv_log[:200]}"))

            # 验证 Format Info 与参数一致
            fi = decoded.format_info
            if fi.color_level_code != pal.code:
                reason = f"{tag} 颜色码不匹配"
                if not quiet:
                    self._ui_queue.put(("log",
                        f"[fi] {tag}: 颜色码不匹配 got={fi.color_level_code:#x} "
                        f"expect={pal.code:#x}"))
                return None, None, reason
            if fi.shape_level_code != shp.code:
                reason = f"{tag} 图形码不匹配"
                if not quiet:
                    self._ui_queue.put(("log",
                        f"[fi] {tag}: 图形码不匹配 got={fi.shape_level_code:#x} "
                        f"expect={shp.code:#x}"))
                return None, None, reason

            S = g.S
            M = len(pal.pairs) * ns
            data_codec = DataCodec(S, M)

            # 统计 erasure 数量
            n_erasure = sum(1 for e in decoded.erasure_flags if e)
            avg_conf = sum(decoded.confidences) / len(decoded.confidences) if decoded.confidences else 0

            frame_bytes = data_codec.decode(
                decoded.symbol_block, decoded.erasure_flags,
            )

            if frame_bytes is not None:
                result_params = {
                    "level": lv, "n_colors": nc, "n_shapes": ns,
                    "S": S, "M": M, "L_max": data_codec.L_max,
                }
                if not quiet:
                    self._ui_queue.put(("log",
                        f"[rs] {tag}: RS+CRC 通过 erasure={n_erasure}/{S} "
                        f"avg_conf={avg_conf:.2f}"))
                return frame_bytes, result_params, None
            else:
                reason = f"{tag} RS/CRC失败 erasure={n_erasure}/{S}"
                if not quiet:
                    self._ui_queue.put(("log",
                        f"[rs] {tag}: RS 或 CRC 失败 erasure={n_erasure}/{S} "
                        f"avg_conf={avg_conf:.2f}"))
                return None, None, reason

        except Exception as e:
            reason = f"{tag} 异常: {e}"
            if not quiet:
                self._ui_queue.put(("log", f"[err] {tag}: {e}"))
            return None, None, reason

    # ---- UI 队列轮询 ----

    FAILED_FRAME_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "debug", "failed_frames",
    )

    def _save_failed_frame(self, img: Image.Image, capture_idx: int, reason: str) -> None:
        """保存解码失败的原始帧图像到 debug/failed_frames/。"""
        os.makedirs(self.FAILED_FRAME_DIR, exist_ok=True)
        ts = time.strftime("%H%M%S")
        # 文件名中替换不合法字符
        safe_reason = reason.replace("/", "_").replace("\\", "_").replace(":", "_")[:40]
        fname = f"fail_{capture_idx:05d}_{ts}_{safe_reason}.png"
        path = os.path.join(self.FAILED_FRAME_DIR, fname)
        try:
            img.save(path)
            self._ui_queue.put(("log",
                f"[save] 失败帧已保存: {path}"))
        except Exception as e:
            self._ui_queue.put(("log",
                f"[save] 失败帧保存失败: {e}"))

    def _poll_ui_queue(self) -> None:
        try:
            while True:
                msg = self._ui_queue.get_nowait()
                self._handle_ui_msg(msg)
        except queue.Empty:
            pass
        self.root.after(50, self._poll_ui_queue)

    def _handle_ui_msg(self, msg: tuple) -> None:
        tag = msg[0]

        if tag == "preview":
            self._show_preview(msg[1])

        elif tag == "params":
            params, K, total_length = msg[1], msg[2], msg[3]
            self.var_status.set(
                f"已检测: L{params['level']} C{params['n_colors']} "
                f"S{params['n_shapes']} | K={K} total={total_length}B "
                f"L_max={params['L_max']}",
            )

        elif tag == "progress":
            decoded_count, K, total_decoded = msg[1], msg[2], msg[3]
            pct = decoded_count / K * 100 if K > 0 else 0
            self.progress["value"] = min(pct, 100)
            self.var_progress.set(
                f"已解码 {decoded_count}/{K} 源块 | "
                f"有效帧 {total_decoded} | 总帧 {self._frame_count}",
            )

        elif tag == "rate":
            kbps = msg[1]
            self.var_rate.set(f"{kbps:.1f} kB/s")

        elif tag == "decoded":
            self._show_result(msg[1])

        elif tag == "error":
            self.var_status.set(f"错误: {msg[1]}")

        elif tag == "log":
            self._append_log(msg[1])

        elif tag == "stopped":
            self.capturing = False
            self.btn_start.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            self.btn_select.config(state=tk.NORMAL)
            if "解码成功" not in self.var_status.get():
                self.var_status.set("捕获已停止")

    def _append_log(self, text: str) -> None:
        """追加一行调试日志。"""
        self.log_text.config(state=tk.NORMAL)
        ts = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"{ts} {text}\n")
        # 保留最后 200 行
        lines = int(self.log_text.index("end-1c").split(".")[0])
        if lines > 200:
            self.log_text.delete("1.0", f"{lines - 200}.0")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _show_preview(self, img: Image.Image) -> None:
        max_sz = self.MAX_PREVIEW_SIZE
        if max(img.size) > max_sz:
            ratio = max_sz / max(img.size)
            img = img.resize(
                (int(img.width * ratio), int(img.height * ratio)),
                Image.NEAREST,
            )
        self.photo_image = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.photo_image, text="")

    def _show_result(self, msg) -> None:
        self.text_result.config(state=tk.NORMAL)
        self.text_result.delete("1.0", tk.END)

        content_type = "文本" if msg.content_type == CONTENT_TYPE_TEXT else "文件"

        # 传输统计
        transfer_time = time.time() - self._transfer_start if self._transfer_start > 0 else 0
        total_data = len(msg.data)
        avg_kbps = (total_data / transfer_time / 1024) if transfer_time > 0 else 0

        # 自动保存文件
        save_path = self._auto_save(msg)

        lines = [
            f"类型: {content_type}",
            f"文件名: {msg.filename or '(无)'}",
            f"数据长度: {total_data} bytes",
            f"保存路径: {save_path}",
            "",
            f"── 传输统计 ──",
            f"传输耗时: {transfer_time:.1f} s",
            f"平均速率: {avg_kbps:.1f} kB/s",
            f"有效帧数: {self._decoded_count}",
            f"总捕获帧: {self._frame_count}",
            "",
        ]

        if msg.content_type == CONTENT_TYPE_TEXT:
            try:
                text = msg.data.decode("utf-8")
                lines.append(text)
            except UnicodeDecodeError:
                lines.append("(无法解码为 UTF-8 文本)")
        else:
            lines.append(f"(二进制文件内容 {total_data} bytes)")

        self.text_result.insert("1.0", "\n".join(lines))
        self.text_result.config(state=tk.DISABLED)

        self.btn_save_result.config(state=tk.NORMAL)
        if msg.content_type == CONTENT_TYPE_TEXT:
            self.btn_copy.config(state=tk.NORMAL)

        self.var_status.set("✓ 解码成功！")
        self.var_progress.set(
            f"解码完成 | {total_data / 1024:.1f} kB | "
            f"{transfer_time:.1f}s | {avg_kbps:.1f} kB/s | "
            f"有效帧 {self._decoded_count}/{self._frame_count}",
        )

    # ---- 保存 / 复制 ----

    RECV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "received")

    def _auto_save(self, msg) -> str:
        """自动保存解码内容到 received/ 目录，返回保存路径。"""
        os.makedirs(self.RECV_DIR, exist_ok=True)

        if msg.filename:
            # 用原文件名，冲突时加序号
            base, ext = os.path.splitext(msg.filename)
            path = os.path.join(self.RECV_DIR, msg.filename)
            i = 1
            while os.path.exists(path):
                path = os.path.join(self.RECV_DIR, f"{base}_{i}{ext}")
                i += 1
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            ext = ".txt" if msg.content_type == CONTENT_TYPE_TEXT else ".bin"
            path = os.path.join(self.RECV_DIR, f"received_{ts}{ext}")

        try:
            with open(path, "wb") as f:
                f.write(msg.data)
        except Exception:
            return "(保存失败)"

        return path

    def _save_result(self) -> None:
        if self._last_msg is None:
            return
        msg = self._last_msg

        if msg.content_type == CONTENT_TYPE_TEXT:
            path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")],
                initialfile=msg.filename or "decoded.txt",
            )
        else:
            path = filedialog.asksaveasfilename(
                defaultextension="",
                filetypes=[("所有文件", "*.*")],
                initialfile=msg.filename or "decoded_file",
            )

        if path:
            try:
                with open(path, "wb") as f:
                    f.write(msg.data)
                self.var_status.set(f"已保存: {path}")
            except Exception as e:
                messagebox.showerror("保存失败", str(e))

    def _copy_text(self) -> None:
        if self._last_msg is None:
            return
        msg = self._last_msg
        if msg.content_type == CONTENT_TYPE_TEXT:
            try:
                text = msg.data.decode("utf-8")
                self.root.clipboard_clear()
                self.root.clipboard_append(text)
                self.var_status.set("已复制到剪贴板")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    root = tk.Tk()
    CimbarDecoderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
