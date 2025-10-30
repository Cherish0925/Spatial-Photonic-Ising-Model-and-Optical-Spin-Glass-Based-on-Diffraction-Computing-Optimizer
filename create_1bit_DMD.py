#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 1920×1080 1bit BMP 测试图（DMD 用）
author : Leslie
"""

import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt



def show_image(img_uint8, title="Preview"):
    """
    使用 matplotlib 可视化生成的 1bit 图像
    :param img_uint8: numpy 数组 (0/255)
    """
    plt.figure(figsize=(8, 5))
    plt.imshow(img_uint8, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")
    plt.show()

def make_1bit_shape(shape: str, size: int, angle: float, canvas_wh=(1920, 1080)):
    """
    生成 1920×1080 1bit 黑白图，居中绘制旋转后的白色正方形或圆形
    :param shape: 'square' 或 'circle'
    :param size:  边长（正方形）或直径（圆形）
    :param angle: 旋转角度（度，逆时针为正）
    :param canvas_wh: (宽, 高)
    :return: uint8 numpy 数组（0/255）
    """
    w, h = canvas_wh
    if size > min(w, h):
        raise ValueError(f"❌ size ({size}) 超过画布最小边长 ({min(w, h)})")

    # 透明通道画布（方便旋转）
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    cx, cy = w // 2, h // 2

    if shape == "square":
        half = size // 2
        # 定义未旋转顶点（以中心为原点）
        pts = np.array([[-half, -half],
                        [half, -half],
                        [half, half],
                        [-half, half]], np.float32)
        # 旋转矩阵（绕中心旋转）
        M = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
        pts_rot = cv2.transform(pts.reshape(1, -1, 2), M).reshape(-1, 2)
        pts_rot += np.array([cx, cy])  # 平移到画布中心
        cv2.fillPoly(overlay, [np.int32(pts_rot)], color=(255, 255, 255, 255))

    elif shape == "circle":
        radius = size // 2
        cv2.circle(overlay, (cx, cy), radius, color=(255, 255, 255, 255), thickness=-1)
        if angle % 360 != 0:  # 圆形旋转没意义，但可保持一致
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            overlay = cv2.warpAffine(overlay, M, (w, h), flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    else:
        raise ValueError("❌ shape 只能是 'square' 或 'circle'")

    # 转灰度（取一个通道即可）
    gray = overlay[:, :, 0]
    # 二值化（保证纯 0/255）
    _, bin_mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return bin_mask


def save_1bit_bmp(img_uint8, file_path):
    """
    保存为 1bit BMP 文件
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    pil_img = Image.fromarray(img_uint8).convert('1')  # 转 1bit
    pil_img.save(file_path, format='BMP')
    print(f"✅ 已保存 1bit BMP -> {file_path}")
    print(f"   尺寸: {pil_img.size}, 模式: {pil_img.mode}")


def create_pattern(output_path, shape="square", size=500, rotation=0, show=False):
    """
    生成并保存 1bit 测试图
    """
    img = make_1bit_shape(shape, size, rotation)
    save_1bit_bmp(img, output_path)
    if show:  # 增加一个开关参数，默认不显示，调试时可以显示
        show_image(img, title=f"{shape}, size={size}, rot={rotation}")

def make_1bit_tian(size: int, border_thickness: int, cross_thickness: int,
                   canvas_wh=(1920, 1080)):
    """
    生成 1920×1080 1bit 黑白图，居中绘制“田”字格
    :param size: 外框边长（像素）
    :param border_thickness: 外框厚度（像素）
    :param cross_thickness: 十字架厚度（像素）
    :param canvas_wh: (宽, 高)
    :return: uint8 numpy 数组（0/255）
    """
    w, h = canvas_wh
    if size > min(w, h):
        raise ValueError(f"❌ size ({size}) 超过画布最小边长 ({min(w, h)})")

    overlay = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    half = size // 2

    # ---- 外框 ----
    # 外正方形
    outer_rect = (cx - half, cy - half, cx + half, cy + half)
    cv2.rectangle(overlay, outer_rect[:2], outer_rect[2:], color=255, thickness=-1)

    # 内正方形（挖空）
    inner_half = half - border_thickness
    if inner_half > 0:
        inner_rect = (cx - inner_half, cy - inner_half, cx + inner_half, cy + inner_half)
        cv2.rectangle(overlay, inner_rect[:2], inner_rect[2:], color=0, thickness=-1)

    # ---- 十字架 ----
    # 竖条
    cv2.rectangle(overlay,
                  (cx - cross_thickness // 2, cy - half),
                  (cx + cross_thickness // 2, cy + half),
                  color=255, thickness=-1)
    # 横条
    cv2.rectangle(overlay,
                  (cx - half, cy - cross_thickness // 2),
                  (cx + half, cy + cross_thickness // 2),
                  color=255, thickness=-1)

    # 保证二值化
    _, bin_mask = cv2.threshold(overlay, 127, 255, cv2.THRESH_BINARY)
    return bin_mask

# -------------------- 示例 --------------------
if __name__ == "__main__":
    t_x=742
    t_y=20
    t_z=20
    output_file = rf"F:\F4321_HS公版\3-测试图像\tian_{t_x}_{t_y}_{t_z}.bmp"

    img = make_1bit_tian(size=t_x, border_thickness=t_y, cross_thickness=t_z)
    save_1bit_bmp(img, output_file)
    show_image(img, title="田字格")

    # 输出路径
    #shape = "square"
    #shape = "circle"
    #size = 741
    #rotation = 45
    #output_file = rf"F:\F4321_HS公版\3-测试图像\{shape}_{size}_{rotation}_1bit.bmp"

    # 示例 1: 800px 正方形，不旋转
    #create_pattern(output_file, shape=shape, size=size, rotation=rotation,show=True)

    # 示例 2: 700px 圆形，旋转 30 度
    # create_pattern(r"F:\F4321_HS公版\3-测试图像\circle_700_rot30.bmp",
    #                shape="circle", size=700, rotation=30)

    # 示例 3: 600px 正方形，旋转 45 度
    # create_pattern(r"F:\F4321_HS公版\3-测试图像\square_600_rot45.bmp",
    #                shape="square", size=600, rotation=45)
