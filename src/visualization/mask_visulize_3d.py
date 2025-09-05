#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import argparse
import numpy as np
import tifffile
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw, ImageFont
import PIL
from typing import Union, Optional


# VTK imports
import vtk
from vtkmodules.util import numpy_support

# 配色表
LABEL_COLORMAP = [
    (255,   0,   0),
    (  0, 255,   0),
    (  0,   0, 255),
    (255, 255,   0),
    (255,   0, 255),
    (  0, 255, 255),
]

def relabel_data(data: np.ndarray):
    """把所有非零 label 重映射到 1,2,3… 并返回新的 data"""
    unique = np.unique(data)
    unique = unique[unique != 0]
    mapping = {old: new for new, old in enumerate(unique, start=1)}
    out = np.zeros_like(data, dtype=np.uint8)
    for old, new in mapping.items():
        out[data == old] = new
    return out

def numpy_to_vtk_image(data: np.ndarray) -> vtk.vtkImageData:
    """将 3D numpy 数组转为 vtkImageData。"""
    data = np.ascontiguousarray(data)
    depth, height, width = data.shape
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(width, height, depth)
    vtk_array = numpy_support.numpy_to_vtk(
        num_array=data.ravel(order='C'),
        deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR
    )
    vtk_img.GetPointData().SetScalars(vtk_array)
    return vtk_img

def process_label(label: int, data: np.ndarray, smooth_iters: int) -> vtk.vtkActor:
    """对单个标签提取等值面并返回对应 Actor。"""
    if label == 0:
        return None
    # 构造二值 mask
    mask = np.where(data == label, label, 0).astype(np.uint8)
    vtk_img = numpy_to_vtk_image(mask)

    # Marching Cubes 提取等值面
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(vtk_img)
    mc.SetValue(0, label)
    mc.ComputeNormalsOn()
    mc.Update()

    # 可选平滑
    if smooth_iters > 0:
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(mc.GetOutputPort())
        smoother.SetNumberOfIterations(smooth_iters)
        smoother.SetRelaxationFactor(0.1)
        smoother.FeatureEdgeSmoothingOff()
        smoother.BoundarySmoothingOn()
        smoother.Update()
        poly = smoother.GetOutput()
    else:
        poly = mc.GetOutput()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    r, g, b = LABEL_COLORMAP[label % len(LABEL_COLORMAP)]
    actor.GetProperty().SetColor(r/255.0, g/255.0, b/255.0)
    actor.GetProperty().SetOpacity(1.0)
    return actor

def render_single_view(data: np.ndarray, smooth_iters: int,
                       az: float, el: float,
                       size=(800,600)) -> Image.Image:
    """
    离屏渲染一个视角的 3D mask，返回 PIL.Image。
    az: 水平旋转角度，el: 俯仰角度。
    """
    # VTK 渲染器和窗口
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1,1,1)
    renwin = vtk.vtkRenderWindow()
    renwin.OffScreenRenderingOn()
    renwin.AddRenderer(renderer)
    renwin.SetSize(*size)

    # 添加所有 label 的 Actor
    labels = np.unique(data)
    if len(labels) == 1:
        print("[WARN] 数据中只包含背景标签，无法渲染。")
        return Image.new('RGB', size, (255, 255, 255))
    with ThreadPoolExecutor() as exe:
        futures = {exe.submit(process_label, lab, data, smooth_iters): lab
                   for lab in labels if lab != 0}
        for f in futures:
            actor = f.result()
            if actor:
                renderer.AddActor(actor)

    if renderer.GetActors().GetNumberOfItems() == 0:
        raise RuntimeError("没有检测到可渲染的标签。")

    # 设置相机视角
    cam = renderer.GetActiveCamera()
    renderer.ResetCamera()
    cam.Azimuth(az)
    cam.Elevation(el)
    renderer.ResetCameraClippingRange()
    renwin.Render()

    # 抓取图像
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(renwin)
    w2if.SetInputBufferTypeToRGB()
    w2if.ReadFrontBufferOff()
    w2if.Update()
    vtk_img = w2if.GetOutput()

    # 转为 numpy 然后 PIL.Image
    width, height, _ = vtk_img.GetDimensions()
    arr = numpy_support.vtk_to_numpy(vtk_img.GetPointData().GetScalars())
    arr = arr.reshape(height, width, 3)
    return Image.fromarray(arr)

def combine_modalities(images, out_path: str):
    """把两张 PIL.Image 横向拼接并保存为 out_path。"""
    w, h = images[0].size
    grid = Image.new('RGB', (w*2, h), (255,255,255))
    for i, img in enumerate(images):
        grid.paste(img, (i*w, 0))
    grid.save(out_path)
    print(f"  → 合并对比图: {out_path}")

# 定义一个更通用的字体类型提示
AnyFont = Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]

def get_scalable_font(font_size: int) -> AnyFont:
    """
    尝试返回一个可缩放的 TTF 字体。
    它会按顺序在 Pillow 自带目录、常见 Linux 和 macOS/Windows 路径中查找。
    如果都找不到，则回退到 Pillow 的默认字体（并尝试设置大小）。
    """
    # 1) Pillow 自带字体
    try:
        # os.path.dirname(PIL.__file__) 在某些环境中可能不准确
        # Pillow >= 9.3.0 提供了更可靠的方式
        from PIL import features
        if features.check("raqm"):
             pil_fonts_path = os.path.join(os.path.dirname(features.get_supported_modules()["raqm"].__file__), "DejaVuSans.ttf")
             if os.path.exists(pil_fonts_path):
                 return ImageFont.truetype(pil_fonts_path, font_size)

        # 传统方式
        pil_fonts_dir = os.path.join(os.path.dirname(PIL.__file__), "fonts")
        candidate = os.path.join(pil_fonts_dir, "DejaVuSans.ttf")
        if os.path.exists(candidate):
            return ImageFont.truetype(candidate, font_size)
    except Exception:
        pass

    # 2) 常见 Linux/macOS 路径
    font_candidates = [
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    # 3) Windows 路径
    if os.name == 'nt':
        font_candidates.extend([
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
        ])

    for candidate in font_candidates:
        if os.path.exists(candidate):
            try:
                return ImageFont.truetype(candidate, font_size)
            except IOError:
                continue

    # 4) 最后回退
    print(f"[WARN] 找不到任何可缩放的 TTF 字体，将使用默认字体。")
    try:
        # Pillow >= 9.2.0 支持为默认字体设置大小
        return ImageFont.load_default(size=font_size)
    except TypeError:
        # 兼容旧版 Pillow
        if font_size > 12: # 默认字体很小，如果用户想要大字体，给个提示
             print("[WARN] 当前 Pillow 版本过低，无法设置默认字体大小。")
        return ImageFont.load_default()


def annotate_image(img: Image.Image,
                   label: str,
                   top_margin: int = 8,
                   font_path: Optional[str] = None,
                   font_size: int = 36,
                   text_color=(0, 0, 0),
                   bg_color=(255, 255, 255)) -> Image.Image:
    """
    在 img 的顶部中央绘制 label 文本（带背景），支持指定字体文件和大小。

    Args:
        img: 要处理的 PIL.Image.Image 对象。
        label: 要绘制的文本。
        top_margin: 文本距离图片顶部的距离。
        font_path: (可选) .ttf 或 .otf 字体文件的路径。如果提供，将优先使用此字体。
        font_size: 文本大小。
        text_color: 文本颜色。
        bg_color: 文本背景颜色。

    Returns:
        处理后的 PIL.Image.Image 对象。
    """
    draw = ImageDraw.Draw(img)
    font = None

    # 1) 加载字体：优先使用用户指定的 font_path
    if font_path and os.path.exists(font_path):
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            print(f"[WARN] 无法从 '{font_path}' 加载字体，将尝试自动查找。")

    # 如果用户未指定字体或加载失败，则自动查找
    if font is None:
        font = get_scalable_font(font_size)

    # 2) 计算文字尺寸 (兼容不同 Pillow 版本)
    try:
        # Pillow >= 10.0.0
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        # Pillow < 10.0.0 (getsize is deprecated)
        text_w, text_h = draw.textsize(label, font=font)

    # 3) 计算位置：水平居中，距离顶部 top_margin
    img_w, img_h = img.size
    x = (img_w - text_w) // 2
    y = top_margin

    # 4) 绘制背景和文字
    pad = max(4, font_size // 4)  # 内边距根据字体大小动态调整
    draw.rectangle(
        [(x - pad, y - pad), (x + text_w + pad, y + text_h + pad)],
        fill=bg_color
    )
    draw.text((x, y - bbox[1]), label, fill=text_color, font=font) # 使用bbox[1]微调垂直位置
    
    return img

def make_pdf_from_images(root_dir: str,
                                      pdf_path: str,
                                      font_size: int = 48,
                                      title_margin: int = 20):
    """
    将 root_dir 下每个 label_xxx 目录的 4 张 compare 图（front/side/top/iso）
    垂直堆叠成 4 行，并在最顶部加一行标题 “Label ID: xxx”，
    最后合成一个多页 PDF（每个 label 一页）。
    """
    pages = []
    for label_folder in sorted(os.listdir(root_dir)):
        folder = os.path.join(root_dir, label_folder)
        if not os.path.isdir(folder):
            continue

        # 读取四个视角的 PNG
        view_files = ['front_compare.png',
                      'side_compare.png',
                      'top_compare.png',
                      'iso_compare.png']
        imgs = []
        for vf in view_files:
            path = os.path.join(folder, vf)
            if not os.path.exists(path):
                raise FileNotFoundError(f"缺少 {path}")
            imgs.append(Image.open(path).convert('RGB'))

        # 四张图尺寸相同
        w, h = imgs[0].size

        # 准备字体和标题文本
        font = get_scalable_font(font_size)
        label_id = label_folder.split('_', 1)[-1]  # 从 'label_3' 得到 '3'
        title = f"Label ID: {label_id}"

        # 计算标题尺寸
        dummy = ImageDraw.Draw(imgs[0])
        try:
            bbox = dummy.textbbox((0, 0), title, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = font.getsize(title)

        # 标题区高度 = 文本高度 + 上下 margin
        title_h = text_h + title_margin * 2

        # 创建整页画布：宽 w，高 = title_h + 4*h
        canvas = Image.new('RGB', (w, title_h + 4*h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # 在标题区中央画文字
        x = (w - text_w) // 2
        y = title_margin
        draw.text((x, y), title, fill=(0,0,0), font=font)

        # 把四张图按行粘到标题区下方
        for i, im in enumerate(imgs):
            canvas.paste(im, (0, title_h + i * h))

        pages.append(canvas)

    if not pages:
        print("[WARN] 未发现任何 label 结果，PDF 为空。")
        return

    # 保存多页 PDF
    pages[0].save(
        pdf_path,
        save_all=True,
        append_images=pages[1:],
        resolution=300
    )
    print(f"[INFO] 已生成每页带标题的 PDF：{pdf_path}")
def main():
    p = argparse.ArgumentParser(
        description='对同一 label_id 的 gt, pred 两个 TIFF 在相同视角下渲染并拼成一张对比图。'
    )
    p.add_argument('--input_dir', '-i', required=True, type=str, dest='input_dir',
                   help='包含 fn_{id}_gt.tiff / fn_{id}_pred.tiff 的文件夹')
    p.add_argument('-o','--output_dir', default='outputs',
                   help='输出根目录（默认：./outputs）')
    p.add_argument('--error_type', '-e', type=str, default='fn', choices=('fn', 'fp',))
    p.add_argument('-s','--smooth', type=int, default=20,
                   help='平滑迭代次数（默认 20；设置 0 跳过平滑）')
    args = p.parse_args()

    # 匹配 fn_{id}_gt.tiff
    error_type = args.error_type
    pattern = re.compile(rf'{error_type}_(\d+)_gt\.tiff$', re.IGNORECASE)
    os.makedirs(args.output_dir, exist_ok=True)

    for fname in sorted(os.listdir(args.input_dir)):
        m = pattern.match(fname)
        if not m:
            continue
        label_id = m.group(1)
        base = f'{error_type}_{label_id}_'
        print(f"  → label_id: {label_id}")
        paths = {
            'gt':   os.path.join(args.input_dir, base + 'gt.tiff'),
            'pred': os.path.join(args.input_dir, base + 'pred.tiff'),
        }
        # 确保两文件都存在
        if not all(os.path.exists(p) for p in paths.values()):
            print(f"[WARN] label {label_id} 缺少 gt/pred 文件，已跳过。")
            continue

        # 读取并检查 3D 数据
        data_dict = {}
        for key, path in paths.items():
            try:
                data = tifffile.imread(path)
                if error_type == 'fp' and key == 'pred':
                    # 如果是 fp, 则过滤掉pred中的非 fp 标签
                    data[data != int(label_id)] = 0

                if error_type == 'fn' and key == 'gt':
                    # 如果是 fn, 则过滤掉 gt 中的非 fn 标签
                    data[data != int(label_id)] = 0
                data = relabel_data(data)  # 重映射标签
            except Exception as e:
                print(f"[ERROR] 读取 {path} 失败: {e}")
                continue
            if data.ndim != 3:
                print(f"[WARN] {path} 不是 3D 数据，已跳过。")
                continue
            if not np.issubdtype(data.dtype, np.integer):
                data = data.astype(np.uint8)
            data_dict[key] = data

        if error_type == 'fp':
            # 取出 pred≠0 的 gt 标签列表
            mask_pred0 = data_dict['pred'] == 0
            gt_labels = np.unique(data_dict['gt'][~mask_pred0])

            # 把 gt 中不在该列表里的全部置零
            mask_gt_invalid = ~np.isin(data_dict['gt'], gt_labels)
            data_dict['gt'][mask_gt_invalid] = 0
        elif error_type == 'fn':
            # 取出 gt≠0 的 pred 标签列表
            pred_labels = np.unique(data_dict['pred'][data_dict['gt'] != 0])
            # 把 pred 中不在该列表里的全部置零
            mask_pred_invalid = ~np.isin(data_dict['pred'], pred_labels)
            data_dict['pred'][mask_pred_invalid] = 0
        if len(data_dict) != 2:
            continue

        out_id_dir = os.path.join(args.output_dir, f'label_{label_id}')
        os.makedirs(out_id_dir, exist_ok=True)
        print(f"[INFO] 处理 label {label_id} …")

        # 四个视角
        views = {
            'front': (0, 0),
            'side':  (90, 0),
            'top':   (0, 90),
            'iso':   (45, 45),
        }

        for view_name, (az, el) in views.items():
            imgs = []
            # 对应的文字标签
            tags = ['GT', 'Pred']
            for key, tag in zip(('gt', 'pred'), tags):
                img = render_single_view(data_dict[key], args.smooth, az, el)
                # 在图上加文字
                img = annotate_image(img, tag)
                imgs.append(img)
            out_png = os.path.join(out_id_dir,
                                   f'{view_name}_compare.png')
            combine_modalities(imgs, out_png)
    pdf_file = os.path.join(args.output_dir, f'{error_type}_errors_all.pdf')
    make_pdf_from_images(args.output_dir, pdf_file,
                                      font_size=48,     # 可调整标题大小
                                      title_margin=20)
if __name__ == '__main__':
    main()
