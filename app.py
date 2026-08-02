"""
QRTC Cimbar 播放器 GUI

功能：
- 配置颜色档、图形档、网格等级、模块大小、FPS
- 输入文本或选择文件作为内容
- 按指定 FPS 播放生成的 cimbar 帧动画
- 完整编码管线：应用层 → 传输层 → 数据层 → 符号层 → 像素渲染

用法：
    python app.py
"""

from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional

import numpy as np
from PIL import Image, ImageTk

from application_layer import (
    CONTENT_TYPE_FILE,
    CONTENT_TYPE_TEXT,
    app_encode,
)
from data_layer import DataCodec
from symbol_layer import (
    ColorPalette,
    Grid,
    ShapeSet,
    SymbolEncoder,
)
from transport_layer import TransportCodec
from transport_layer.lt_code import lt_encode_frame, split_blocks


# ---------------------------------------------------------------------------
# numpy 加速渲染器
# ---------------------------------------------------------------------------

def fast_render(
    frame,
    module_size: int = 12,
    quiet_zone_size: int = 4,
) -> Image.Image:
    """将 RenderedFrame 渲染为 PIL Image（numpy 向量化加速）。

    相比 FrameRenderer 的逐像素 draw.point，使用 numpy 数组操作快约 50-100x。
    """
    N = frame.N
    qz = quiet_zone_size
    ms = module_size
    img_size = (N + 2 * qz) * ms

    arr = np.full((img_size, img_size, 3), 255, dtype=np.uint8)

    palette = frame.palette
    shape_set = frame.shape_set

    # 预计算单个模块内的归一化坐标网格
    ys, xs = np.meshgrid(np.arange(ms), np.arange(ms), indexing="ij")
    x_norm = (xs + 0.5) / ms
    y_norm = (ys + 0.5) / ms

    for row in range(N):
        for col in range(N):
            mod = frame.modules[row][col]
            if mod is None:
                continue
            px = (col + qz) * ms
            py = (row + qz) * ms

            if mod.is_auxiliary:
                if mod.color_a == 0:  # 黑
                    arr[py:py + ms, px:px + ms] = (0, 0, 0)
                # 白色是默认值，无需操作
            else:
                rgb_a = palette.get_rgb(mod.color_a)
                rgb_b = palette.get_rgb(mod.color_b)
                shape = shape_set.get_shape(mod.shape_index)

                # shape.region_a 支持 numpy 数组输入（内部仅用 +, <, >=）
                mask_a = shape.region_a(x_norm, y_norm)
                block = np.full((ms, ms, 3), rgb_b, dtype=np.uint8)
                block[mask_a] = rgb_a
                arr[py:py + ms, px:px + ms] = block

    return Image.fromarray(arr, "RGB")


# ---------------------------------------------------------------------------
# GUI 应用
# ---------------------------------------------------------------------------

class CimbarPlayerApp:
    """Cimbar 帧播放器主窗口。"""

    BUFFER_SIZE = 60
    QUIET_ZONE = 4

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("QRTC Cimbar 播放器")
        self.root.minsize(640, 620)

        # --- 播放状态 ---
        self.playing = False
        self.paused = False
        self.playback_job: Optional[str] = None
        self.photo_image: Optional[ImageTk.PhotoImage] = None

        # --- 编码管线对象（生成时构建）---
        self._enc: Optional[SymbolEncoder] = None
        self._data_codec: Optional[DataCodec] = None
        self._blocks = None
        self._total_length = 0
        self._chunk_size = 0
        self._L_max = 0
        self._module_size = 12
        self._seq = 0
        self._frame_count = 0

        # --- 后台生成线程 ---
        self._frame_queue: queue.Queue = queue.Queue(maxsize=self.BUFFER_SIZE)
        self._gen_stop = threading.Event()
        self._gen_thread: Optional[threading.Thread] = None
        self._current_img: Optional[Image.Image] = None

        self.frame_info: Dict = {}

        self._build_ui()

    # ---- UI 构建 ----

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # 配置面板
        self._config_frame = ttk.LabelFrame(main, text="配置", padding=10)
        self._config_frame.pack(fill=tk.X, pady=(0, 8))
        self._build_config_panel(self._config_frame)

        # 内容面板
        self._content_frame = ttk.LabelFrame(main, text="内容", padding=10)
        self._content_frame.pack(fill=tk.X, pady=(0, 8))
        self._build_content_panel(self._content_frame)

        # 控制按钮
        self._ctrl_frame = ttk.Frame(main)
        self._ctrl_frame.pack(fill=tk.X, pady=(0, 8))
        self._build_control_panel(self._ctrl_frame)

        # 进度条
        self.progress = ttk.Progressbar(main, mode="determinate")
        self.progress.pack(fill=tk.X, pady=(0, 4))
        self.var_progress_label = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.var_progress_label).pack(anchor=tk.W)

        # 帧预览
        display = ttk.LabelFrame(main, text="帧预览", padding=10)
        display.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.image_label = ttk.Label(display, text="尚未生成", anchor=tk.CENTER,
                                     font=("", 14))
        self.image_label.pack(fill=tk.BOTH, expand=True)
        self.image_label.bind("<Configure>", self._on_display_resize)
        self._display_size: Optional[tuple] = None  # (w, h) of display area

        # 状态栏
        self.var_status = tk.StringVar(value="就绪")
        ttk.Label(main, textvariable=self.var_status,
                  relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X, pady=(8, 0))

    def _build_config_panel(self, parent: ttk.Widget) -> None:
        # 第一行：等级 / 颜色 / 图形
        row1 = ttk.Frame(parent)
        row1.pack(fill=tk.X, pady=2)

        ttk.Label(row1, text="网格等级:").pack(side=tk.LEFT)
        self.var_level = tk.StringVar(value="15 (133×133)")
        ttk.OptionMenu(
            row1, self.var_level, "15 (133×133)",
            "1 (21×21)", "2 (29×29)", "3 (37×37)", "4 (45×45)", "5 (53×53)",
            "6 (61×61)", "7 (69×69)", "8 (77×77)", "9 (85×85)",
            "10 (93×93)", "11 (101×101)", "12 (109×109)",
            "13 (117×117)", "14 (125×125)", "15 (133×133)",
        ).pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(row1, text="颜色档:").pack(side=tk.LEFT)
        self.var_colors = tk.StringVar(value="8 色")
        ttk.OptionMenu(
            row1, self.var_colors, "8 色", "2 色", "4 色", "8 色",
        ).pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(row1, text="图形档:").pack(side=tk.LEFT)
        self.var_shapes = tk.StringVar(value="4")
        ttk.OptionMenu(
            row1, self.var_shapes, "4", "2 图形", "4 图形",
        ).pack(side=tk.LEFT, padx=(4, 0))

        # 第二行：模块大小 / FPS
        row2 = ttk.Frame(parent)
        row2.pack(fill=tk.X, pady=(6, 0))

        ttk.Label(row2, text="模块大小:").pack(side=tk.LEFT)
        self.var_module_size = tk.StringVar(value="auto")
        ttk.OptionMenu(
            row2, self.var_module_size, "auto",
            "auto", "8", "10", "12", "16", "20",
        ).pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(row2, text="FPS:").pack(side=tk.LEFT)
        self.var_fps = tk.StringVar(value="10")
        ttk.Spinbox(
            row2, from_=1, to=60, textvariable=self.var_fps, width=5,
        ).pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(row2, text="(喷泉码无限生成帧)").pack(side=tk.LEFT, padx=(4, 0))

    def _build_content_panel(self, parent: ttk.Widget) -> None:
        # 内容类型选择
        type_row = ttk.Frame(parent)
        type_row.pack(fill=tk.X)
        self.var_content_type = tk.StringVar(value="text")
        ttk.Radiobutton(
            type_row, text="文本", variable=self.var_content_type,
            value="text", command=self._on_type_change,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            type_row, text="文件", variable=self.var_content_type,
            value="file", command=self._on_type_change,
        ).pack(side=tk.LEFT, padx=(8, 16))

        # 文本输入
        self.text_frame = ttk.Frame(parent)
        self.text_frame.pack(fill=tk.X, pady=(8, 0))
        self.text_input = tk.Text(self.text_frame, height=4, width=50)
        self.text_input.pack(fill=tk.X)
        self.text_input.insert("1.0", "Hello, Cimbar!")

        # 文件选择（初始隐藏）
        self.file_frame = ttk.Frame(parent)
        ttk.Label(self.file_frame, text="文件:").pack(side=tk.LEFT)
        self.var_file_path = tk.StringVar()
        ttk.Entry(
            self.file_frame, textvariable=self.var_file_path, width=42,
        ).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Button(
            self.file_frame, text="选择...", command=self._select_file,
        ).pack(side=tk.LEFT)

        # 文件名
        fn_row = ttk.Frame(parent)
        fn_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(fn_row, text="文件名:").pack(side=tk.LEFT)
        self.var_filename = tk.StringVar()
        ttk.Entry(
            fn_row, textvariable=self.var_filename, width=30,
        ).pack(side=tk.LEFT, padx=(4, 0))

    def _build_control_panel(self, parent: ttk.Widget) -> None:
        self.btn_generate = ttk.Button(parent, text="生成并播放", command=self._generate)
        self.btn_generate.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_play = ttk.Button(
            parent, text="▶ 播放", command=self._play, state=tk.DISABLED,
        )
        self.btn_play.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_pause = ttk.Button(
            parent, text="⏸ 暂停", command=self._pause, state=tk.DISABLED,
        )
        self.btn_pause.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_stop = ttk.Button(
            parent, text="⏹ 停止", command=self._stop, state=tk.DISABLED,
        )
        self.btn_stop.pack(side=tk.LEFT, padx=(0, 16))

        self.btn_save = ttk.Button(
            parent, text="保存当前帧", command=self._save_current_frame, state=tk.DISABLED,
        )
        self.btn_save.pack(side=tk.LEFT)

    # ---- 事件处理 ----

    def _on_type_change(self) -> None:
        if self.var_content_type.get() == "text":
            self.file_frame.pack_forget()
            self.text_frame.pack(fill=tk.X, pady=(8, 0))
        else:
            self.text_frame.pack_forget()
            self.file_frame.pack(fill=tk.X, pady=(8, 0))

    def _select_file(self) -> None:
        path = filedialog.askopenfilename()
        if path:
            self.var_file_path.set(path)
            if not self.var_filename.get():
                self.var_filename.set(os.path.basename(path))

    def _stop_gen_thread(self) -> None:
        """停止后台生成线程。"""
        self._gen_stop.set()
        if self._gen_thread is not None:
            self._gen_thread.join(timeout=2.0)
            self._gen_thread = None
        # 清空队列
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

    def _stop_playback(self) -> None:
        """停止播放（不重置 seq）。"""
        self.playing = False
        self.paused = False
        if self.playback_job is not None:
            self.root.after_cancel(self.playback_job)
            self.playback_job = None
        self.btn_play.config(state=tk.NORMAL, text="▶ 播放")
        self.btn_pause.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.DISABLED)
        # 恢复配置和内容面板
        if not self._config_frame.winfo_ismapped():
            self._config_frame.pack(fill=tk.X, pady=(0, 8), before=self._ctrl_frame)
        if not self._content_frame.winfo_ismapped():
            self._content_frame.pack(fill=tk.X, pady=(0, 8), before=self._ctrl_frame)

    # ---- 生成 ----

    def _generate(self) -> None:
        # 停止当前播放和生成线程
        self._stop_playback()
        self._stop_gen_thread()

        # 解析参数
        try:
            level_str = self.var_level.get().split(" ")[0]
            level = int(level_str)
            n_colors = int(self.var_colors.get().split(" ")[0])
            n_shapes = int(self.var_shapes.get().split(" ")[0])
            module_size = self.var_module_size.get()  # "auto" 或数字字符串
            fps = int(self.var_fps.get())
        except ValueError:
            messagebox.showerror("错误", "参数无效，请检查配置")
            return

        if fps < 1 or fps > 60:
            messagebox.showerror("错误", "FPS 范围 1-60")
            return

        # 获取内容
        if self.var_content_type.get() == "text":
            text = self.text_input.get("1.0", tk.END).rstrip("\n")
            if not text:
                messagebox.showerror("错误", "请输入文本内容")
                return
            data = text.encode("utf-8")
            content_type = CONTENT_TYPE_TEXT
            filename = self.var_filename.get()
        else:
            file_path = self.var_file_path.get()
            if not file_path or not os.path.exists(file_path):
                messagebox.showerror("错误", "请选择有效文件")
                return
            try:
                with open(file_path, "rb") as f:
                    data = f.read()
            except Exception as e:
                messagebox.showerror("错误", f"读取文件失败: {e}")
                return
            content_type = CONTENT_TYPE_FILE
            filename = self.var_filename.get() or os.path.basename(file_path)

        # 构建编码管线
        try:
            g = Grid(level)
            pal = ColorPalette(n_colors)
            shp = ShapeSet(n_shapes)
            enc = SymbolEncoder(pal, shp, g)
            S = enc.S
            M = enc.M

            data_codec = DataCodec(S, M)
            L_max = data_codec.L_max

            content = app_encode(data, content_type, filename)
            total_length = len(content)

            transport = TransportCodec(L_max)
            K, chunk_size = transport.get_params(total_length)
            blocks = split_blocks(content, chunk_size)
        except Exception as e:
            messagebox.showerror("编码初始化失败", str(e))
            return

        # 存储编码管线对象
        self._enc = enc
        self._data_codec = data_codec
        self._blocks = blocks
        self._total_length = total_length
        self._chunk_size = chunk_size
        self._L_max = L_max
        self._grid_n = g.N  # 网格尺寸，用于计算 auto module_size
        self._module_size = self._calc_module_size(module_size)
        self._seq = 0
        self._frame_count = 0

        self.frame_info = {
            "level": level, "n_colors": n_colors, "n_shapes": n_shapes,
            "module_size": module_size, "fps": fps,
            "S": S, "M": M, "L_max": L_max, "K": K,
            "total_length": total_length,
        }

        # 启动后台生成线程
        self._gen_stop.clear()
        self._gen_thread = threading.Thread(
            target=self._gen_worker, daemon=True,
        )
        self._gen_thread.start()

        # 禁用按钮，等待第一帧
        self.btn_generate.config(state=tk.DISABLED)
        self.btn_play.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED)
        self.var_progress_label.set("正在生成帧缓冲...")
        self.progress["value"] = 0

        # 轮询等待第一帧
        self.root.after(100, self._wait_first_frame)

    def _gen_worker(self) -> None:
        """后台线程：持续生成帧到队列，seq 无限递增。"""
        while not self._gen_stop.is_set():
            try:
                frame_bytes = lt_encode_frame(
                    self._blocks, self._seq, self._total_length,
                    self._chunk_size, self._L_max,
                )
                symbols = self._data_codec.encode(frame_bytes)
                rendered = self._enc.encode(symbols)
                # 每帧读取最新 module_size（支持运行时自动调整）
                ms = self._module_size
                img = fast_render(rendered, ms, self.QUIET_ZONE)

                # put 会阻塞当队列满，自然实现背压
                self._frame_queue.put(img, timeout=0.5)
                self._seq += 1
            except queue.Full:
                continue
            except Exception:
                import traceback
                msg = traceback.format_exc()
                self.root.after(0, self._on_generate_error, msg)
                break

    def _wait_first_frame(self) -> None:
        """等待第一帧生成完毕。"""
        if self._gen_stop.is_set() and self._frame_queue.empty():
            return  # 生成已停止

        if not self._frame_queue.empty():
            img = self._frame_queue.get_nowait()
            self._current_img = img
            self._frame_count = 1
            self._display_img(img)

            self.btn_generate.config(state=tk.NORMAL)
            self.btn_save.config(state=tk.NORMAL)
            self.var_progress_label.set(
                f"就绪 | L_max={self.frame_info['L_max']} K={self.frame_info['K']} "
                f"M={self.frame_info['M']} S={self.frame_info['S']} | "
                f"内容 {self.frame_info['total_length']} bytes",
            )
            self._update_buffer_progress()
            # 生成完毕后自动开始播放
            self._play()
        else:
            self.root.after(100, self._wait_first_frame)

    def _update_buffer_progress(self) -> None:
        """更新缓冲区填充进度条。"""
        pct = self._frame_queue.qsize() / self.BUFFER_SIZE * 100
        self.progress["value"] = pct

    def _on_generate_error(self, msg: str) -> None:
        self.btn_generate.config(state=tk.NORMAL)
        self.var_progress_label.set("")
        self.progress["value"] = 0
        messagebox.showerror("生成失败", msg)

    # ---- 自动模块大小 ----

    def _calc_module_size(self, setting) -> int:
        """根据设置计算 module_size。

        - "auto": 根据显示区域大小自动计算，填满区域
        - 数字字符串: 直接返回固定值
        """
        if isinstance(setting, str) and setting == "auto":
            if self._display_size is None or not hasattr(self, "_grid_n"):
                return 12  # 默认值
            avail_w, avail_h = self._display_size
            total_modules = self._grid_n + 2 * self.QUIET_ZONE
            # 取宽高较小者，确保不超出显示区域
            ms = min(avail_w, avail_h) // total_modules
            return max(ms, 2)  # 至少 2px
        else:
            try:
                return int(setting)
            except (ValueError, TypeError):
                return 12

    def _on_display_resize(self, event) -> None:
        """显示区域大小变化时，自动调整 module_size。"""
        new_size = (event.width, event.height)
        if new_size == self._display_size:
            return
        self._display_size = new_size

        if self._enc is None:
            return  # 尚未生成

        old_ms = self._module_size
        new_ms = self._calc_module_size(self.var_module_size.get())
        if new_ms != old_ms and new_ms > 0:
            self._module_size = new_ms
            # 清空缓冲队列，让新帧用新尺寸
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    break
            self._update_buffer_progress()

    # ---- 帧显示 ----

    def _display_img(self, img: Image.Image) -> None:
        """显示一帧 PIL Image（1:1 像素，不缩放）。"""
        self.photo_image = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.photo_image, text="")

        status = "播放中" if (self.playing and not self.paused) else \
                 "已暂停" if self.paused else "就绪"
        self.var_status.set(
            f"帧 #{self._frame_count} (seq={self._seq - 1}) | "
            f"缓冲 {self._frame_queue.qsize()}/{self.BUFFER_SIZE} | "
            f"K={self.frame_info.get('K', '?')} M={self.frame_info.get('M', '?')} | "
            f"{status}",
        )

    # ---- 播放控制 ----

    def _play(self) -> None:
        if self._enc is None:
            return

        # 从暂停恢复
        if self.paused:
            self.paused = False
            self.btn_play.config(state=tk.DISABLED, text="▶ 播放")
            self.btn_pause.config(state=tk.NORMAL)
            self._schedule_next_frame()
            return

        # 全新开始
        self.playing = True
        self.paused = False
        self.btn_play.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.NORMAL)
        # 隐藏配置和内容面板，放大显示区域
        self._config_frame.pack_forget()
        self._content_frame.pack_forget()
        self._schedule_next_frame()

    def _schedule_next_frame(self) -> None:
        if not self.playing or self.paused:
            return

        try:
            img = self._frame_queue.get_nowait()
            self._current_img = img
            self._frame_count += 1
            self._display_img(img)
            self._update_buffer_progress()
        except queue.Empty:
            pass

        try:
            fps = int(self.var_fps.get())
        except ValueError:
            fps = 10
        delay = max(1, int(1000 / fps))
        self.playback_job = self.root.after(delay, self._schedule_next_frame)

    def _pause(self) -> None:
        if not self.playing:
            return
        self.paused = True
        if self.playback_job is not None:
            self.root.after_cancel(self.playback_job)
            self.playback_job = None
        self.btn_pause.config(state=tk.DISABLED)
        self.btn_play.config(state=tk.NORMAL, text="▶ 继续")
        self.var_status.set(
            f"帧 #{self._frame_count} (seq={self._seq - 1}) | 已暂停",
        )

    def _stop(self) -> None:
        self._stop_playback()

    # ---- 保存 ----

    def _save_current_frame(self) -> None:
        if self._current_img is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
            initialfile=f"cimbar_frame_{self._frame_count:04d}.png",
        )
        if path:
            self._current_img.save(path)
            self.var_progress_label.set(f"已保存: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    root = tk.Tk()
    CimbarPlayerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
