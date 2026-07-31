"""
符号层（Symbol Layer）

负责视觉到符号块的映射：定义屏幕上画什么、如何定位对齐、每个模块承载哪个符号、符号块如何排列。

上层：数据层
- 编码方向：接收符号块，渲染到屏幕
- 解码方向：交付符号块 + 置信度（erasure 标记）

下层：物理图像（屏幕/摄像头）

模块结构：
- colors: 颜色定义、颜色档位、颜色对
- shapes: 图形定义、图形档位
- symbol: 符号编码/解码（颜色对 + 图形 → 符号值）
- grid: 网格、结构区域（Finder/Separator/Timing/Format Info）
- format_info: BCH(14,4) 编解码
- module: 模块类型定义
- encoder: 符号块 → 帧模块描述矩阵
- decoder: 帧图像 → 符号块 + 置信度
"""

from .colors import ColorPalette, ColorPair, COLOR_LEVELS, COLOR_LEVEL_CODE
from .shapes import ShapeSet, Shape, ShapeLevel, SHAPE_LEVEL_CODE, register_shape_set
from .symbol import SymbolSpace, SymbolComponents
from .grid import Grid, GridLevel, GRID_SIZES, ModuleType, ModuleCoord, FINDER_PATTERN
from .format_info import FormatInfo
from .module import ModuleCategory, RenderedModule
from .encoder import SymbolEncoder, RenderedFrame
from .decoder import SymbolDecoder, DecodedFrame, DecodedSymbol, ModuleSampler
from .renderer import FrameRenderer
from .image_sampler import ImageSampler
from .opencv_sampler import OpenCVSampler

__all__ = [
    # colors
    "ColorPalette", "ColorPair", "COLOR_LEVELS", "COLOR_LEVEL_CODE",
    # shapes
    "ShapeSet", "Shape", "ShapeLevel", "SHAPE_LEVEL_CODE", "register_shape_set",
    # symbol
    "SymbolSpace", "SymbolComponents",
    # grid
    "Grid", "GridLevel", "GRID_SIZES", "ModuleType", "ModuleCoord", "FINDER_PATTERN",
    # format_info
    "FormatInfo",
    # module
    "ModuleCategory", "RenderedModule",
    # encoder
    "SymbolEncoder", "RenderedFrame",
    # decoder
    "SymbolDecoder", "DecodedFrame", "DecodedSymbol", "ModuleSampler",
    # renderer
    "FrameRenderer",
    # image_sampler
    "ImageSampler",
    # opencv_sampler
    "OpenCVSampler",
]
