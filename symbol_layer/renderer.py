"""
像素级渲染器：将 RenderedFrame 渲染为真实图像（PNG）。

每个模块渲染为 module_size × module_size 像素的正方形区域。
数据模块按图形定义绘制两种颜色的三角形分布。
辅助模块填充纯黑或纯白。
网格四周留有 quiet_zone_size 模块宽的静区。
"""

from __future__ import annotations

from typing import Optional

from PIL import Image, ImageDraw, ImageFilter
import random as _random

import cv2
import numpy as np
import math

from .colors import ColorPalette
from .encoder import RenderedFrame
from .grid import Grid
from .module import ModuleCategory, RenderedModule
from .shapes import ShapeSet


class FrameRenderer:
    """
    将 RenderedFrame 渲染为 PIL Image。

    Args:
        module_size: 每个模块的像素边长
        quiet_zone_size: 静区宽度（模块数），默认 4
    """

    def __init__(self, module_size: int = 10, quiet_zone_size: int = 4) -> None:
        if module_size < 1:
            raise ValueError(f"module_size 必须 >= 1, 得到 {module_size}")
        self.module_size = module_size
        self.quiet_zone_size = quiet_zone_size

    def render(self, frame: RenderedFrame) -> Image.Image:
        """
        渲染帧为 PIL Image（RGB 模式）。

        图像总尺寸 = (N + 2*quiet_zone) * module_size
        """
        N = frame.N
        qz = self.quiet_zone_size
        ms = self.module_size
        img_size = (N + 2 * qz) * ms

        img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        palette = frame.palette
        shape_set = frame.shape_set

        for row in range(N):
            for col in range(N):
                mod = frame.modules[row][col]
                if mod is None:
                    continue
                px = (col + qz) * ms
                py = (row + qz) * ms

                if mod.is_auxiliary:
                    self._draw_auxiliary(draw, px, py, ms, mod)
                else:
                    self._draw_data(draw, px, py, ms, mod, palette, shape_set)

        return img

    def render_to_file(self, frame: RenderedFrame, path: str) -> None:
        """渲染帧并保存为图像文件。"""
        img = self.render(frame)
        img.save(path)

    def _draw_auxiliary(
        self, draw: ImageDraw.ImageDraw,
        px: int, py: int, ms: int,
        mod: RenderedModule,
    ) -> None:
        """绘制辅助模块（纯黑或纯白）。"""
        color_id = mod.color_a
        if color_id == 0:
            rgb = (0, 0, 0)
        else:
            rgb = (255, 255, 255)
        draw.rectangle([px, py, px + ms - 1, py + ms - 1], fill=rgb)

    def _draw_data(
        self, draw: ImageDraw.ImageDraw,
        px: int, py: int, ms: int,
        mod: RenderedModule,
        palette: ColorPalette,
        shape_set: ShapeSet,
    ) -> None:
        """绘制数据模块（两种颜色 + 图形）。"""
        rgb_a = palette.get_rgb(mod.color_a)
        rgb_b = palette.get_rgb(mod.color_b)
        shape = shape_set.get_shape(mod.shape_index)

        # 按像素遍历，根据图形判定函数决定颜色
        for dy in range(ms):
            for dx in range(ms):
                # 归一化坐标 [0, 1)
                x = (dx + 0.5) / ms
                y = (dy + 0.5) / ms
                if shape.region_a(x, y):
                    draw.point((px + dx, py + dy), fill=rgb_a)
                else:
                    draw.point((px + dx, py + dy), fill=rgb_b)

    @staticmethod
    def add_noise(img: Image.Image, intensity: int = 15, seed: int | None = None) -> Image.Image:
        """
        对图像添加随机高斯噪声。

        Args:
            img: 原始 PIL Image（RGB）
            intensity: 噪声幅度（0-255），每个通道随机 ±intensity
            seed: 随机种子，None 则不固定
        Returns:
            添加噪声后的新 Image
        """
        rng = _random.Random(seed)
        pixels = img.load()
        w, h = img.size
        result = img.copy()
        rp = result.load()
        for y in range(h):
            for x in range(w):
                r, g, b = pixels[x, y]
                nr = max(0, min(255, r + rng.randint(-intensity, intensity)))
                ng = max(0, min(255, g + rng.randint(-intensity, intensity)))
                nb = max(0, min(255, b + rng.randint(-intensity, intensity)))
                rp[x, y] = (nr, ng, nb)
        return result

    @staticmethod
    def add_blur(img: Image.Image, radius: float = 1.0) -> Image.Image:
        """
        对图像添加高斯模糊。

        Args:
            img: 原始 PIL Image
            radius: 模糊半径（像素），0 表示不模糊
        Returns:
            模糊后的新 Image
        """
        if radius <= 0:
            return img.copy()
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    @staticmethod
    def add_affine(
        img: Image.Image,
        rotation_deg: float = 2.0,
        scale: float = 0.98,
        translate_x: float = 0.0,
        translate_y: float = 0.0,
    ) -> Image.Image:
        """
        对图像施加轻微仿射变换（旋转 + 缩放 + 平移）。

        以图像中心为原点进行变换，输出图像尺寸不变，超出边界部分裁剪，
        空白区域填充白色。

        Args:
            img: 原始 PIL Image
            rotation_deg: 旋转角度（度），正值顺时针
            scale: 缩放因子，1.0 = 不变
            translate_x: 水平平移（像素）
            translate_y: 垂直平移（像素）
        Returns:
            变换后的新 Image
        """
        w, h = img.size
        cx, cy = w / 2.0, h / 2.0

        theta = math.radians(rotation_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # PIL AFFINE 变换矩阵：输出 → 输入的逆映射
        # 正变换：先缩放，再旋转，再平移
        # 逆变换矩阵（6 元素 a,b,c,d,e,f）：
        #   src = a*dst + b*dst_y + c
        #   src_y = d*dst_x + e*dst_y + f
        s = 1.0 / scale
        a = s * cos_t
        b = -s * sin_t
        d = s * sin_t
        e = s * cos_t
        c = cx - a * cx - b * cy - translate_x
        f = cy - d * cx - e * cy - translate_y

        return img.transform(
            (w, h),
            Image.AFFINE,
            (a, b, c, d, e, f),
            resample=Image.BILINEAR,
            fillcolor=(255, 255, 255),
        )

    @staticmethod
    def add_lens_distortion(
        img: Image.Image,
        k1: float = 0.02,
        k2: float = 0.0,
    ) -> Image.Image:
        """
        对图像施加桶形/枕形镜头畸变。

        使用 Brown-Conrady 模型：
            r_distorted = r * (1 + k1*r^2 + k2*r^4)
        k1 > 0: 桶形畸变（边缘向外膨胀）
        k1 < 0: 枕形畸变（边缘向内收缩）

        Args:
            img: 原始 PIL Image
            k1: 二次畸变系数
            k2: 四次畸变系数
        Returns:
            畸变后的新 Image
        """
        import numpy as np

        w, h = img.size
        arr = np.asarray(img, dtype=np.uint8).copy()

        # 生成坐标网格
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        cx, cy = w / 2.0, h / 2.0

        # 归一化坐标（以图像中心为原点，短边为 1.0）
        norm = min(w, h) / 2.0
        x_n = (xs - cx) / norm
        y_n = (ys - cy) / norm
        r2 = x_n ** 2 + y_n ** 2
        r4 = r2 ** 2

        # 畸变因子
        factor = 1.0 + k1 * r2 + k2 * r4

        # 畸变后的归一化坐标
        x_d = x_n * factor
        y_d = y_n * factor

        # 映射回像素坐标
        map_x = (x_d * norm + cx).astype(np.float32)
        map_y = (y_d * norm + cy).astype(np.float32)

        # 双线性插值采样
        x0 = np.clip(np.floor(map_x).astype(int), 0, w - 1)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y0 = np.clip(np.floor(map_y).astype(int), 0, h - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)

        fx = (map_x - x0).astype(np.float32)[..., None]
        fy = (map_y - y0).astype(np.float32)[..., None]

        top = arr[y0, x0] * (1 - fx) + arr[y0, x1] * fx
        bot = arr[y1, x0] * (1 - fx) + arr[y1, x1] * fx
        result = top * (1 - fy) + bot * fy

        # 越界区域填充白色
        out_of_bounds = (map_x < 0) | (map_x >= w) | (map_y < 0) | (map_y >= h)
        result[out_of_bounds] = 255

        return Image.fromarray(result.astype(np.uint8), "RGB")

    @staticmethod
    def add_perspective(
        img: Image.Image,
        tilt_x: float = 0.15,
        tilt_y: float = 0.0,
        pan: float = 0.0,
    ) -> Image.Image:
        """
        模拟俯拍透视畸变（相机倾斜拍摄平面码图）。

        通过对图像四角施加非对称位移来模拟透视投影：
        - tilt_x: 水平方向倾斜（绕 Y 轴旋转），上边缩小/下边放大
        - tilt_y: 垂直方向倾斜（绕 X 轢旋转），左边缩小/右边放大
        - pan: 整体平移比例

        Args:
            img: 原始 PIL Image
            tilt_x: 水平倾斜量 (0~0.3)，正值=顶部远
            tilt_y: 垂直倾斜量 (0~0.3)，正值=左侧远
            pan: 平移比例
        Returns:
            透视畸变后的新 Image
        """
        w, h = img.size

        # 原始四角
        src = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h],
        ])

        # 透视畸变后的四角
        # tilt_x > 0: 顶部向内收缩（远处变小）
        top_shrink = w * tilt_x * 0.5
        bottom_shrink = -w * tilt_x * 0.5
        # tilt_y > 0: 左侧向内收缩
        left_shrink = h * tilt_y * 0.5
        right_shrink = -h * tilt_y * 0.5

        pan_px = w * pan

        dst = np.float32([
            [top_shrink + pan_px, left_shrink],
            [w - top_shrink + pan_px, right_shrink],
            [w - bottom_shrink + pan_px, h - right_shrink],
            [bottom_shrink + pan_px, h - left_shrink],
        ])

        matrix = cv2.getPerspectiveTransform(src, dst)
        result = cv2.warpPerspective(
            np.array(img.convert("RGB")),
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        return Image.fromarray(result, "RGB")

    @staticmethod
    def add_offset(
        img: Image.Image,
        offset_x: float = 0.1,
        offset_y: float = 0.1,
        margin: float = 0.3,
    ) -> Image.Image:
        """
        将码图放置在更大的白色画布上，模拟码图偏离图像中心。

        Args:
            img: 原始 PIL Image
            offset_x: 水平偏移比例 (-1~1)，正值=偏右，负值=偏左
            offset_y: 垂直偏移比例 (-1~1)，正值=偏下，负值=偏上
            margin: 四周额外留白占原图边长的比例
        Returns:
            偏移后的新 Image
        """
        w, h = img.size
        new_w = int(w * (1 + 2 * margin))
        new_h = int(h * (1 + 2 * margin))

        canvas = Image.new("RGB", (new_w, new_h), (255, 255, 255))

        max_dx = (new_w - w) / 2.0
        max_dy = (new_h - h) / 2.0
        px = int(new_w / 2 - w / 2 + offset_x * max_dx)
        py = int(new_h / 2 - h / 2 + offset_y * max_dy)

        canvas.paste(img, (px, py))
        return canvas
