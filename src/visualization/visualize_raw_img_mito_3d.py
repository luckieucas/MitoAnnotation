#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import tifffile
import colorsys
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import Union, List, Tuple, Dict

# VTK imports
import vtk
from vtkmodules.util import numpy_support

# --- Start: Utility and Core Rendering Functions ---

def generate_distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    """
    生成n个视觉上区分度高的颜色。
    通过在HSV色彩空间的色相(Hue)上均匀分布来生成。
    """
    if n == 0:
        return []
    
    hsv_colors = []
    # 使用黄金比例来获得更分散的颜色
    golden_ratio_conjugate = 0.61803398875
    hue = 0.5  # 从一个不刺眼的颜色开始
    for _ in range(n):
        hue += golden_ratio_conjugate
        hue %= 1
        # 保持较高的饱和度(S)和亮度(V)以确保颜色鲜明
        hsv_colors.append((hue, 0.85, 0.95))

    rgb_colors = []
    for h, s, v in hsv_colors:
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        rgb_colors.append((int(r * 255), int(g * 255), int(b * 255)))
        
    return rgb_colors


def relabel_data(data: np.ndarray):
    """把所有非零 label 重映射到 1,2,3… 并返回新的 data"""
    unique = np.unique(data)
    unique = unique[unique != 0]
    mapping = {old: new for new, old in enumerate(unique, start=1)}
    out = np.zeros_like(data, dtype=np.uint16) # Use uint16 for more labels
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
        array_type=vtk.VTK_UNSIGNED_SHORT if data.dtype == np.uint16 else vtk.VTK_UNSIGNED_CHAR
    )
    vtk_img.GetPointData().SetScalars(vtk_array)
    return vtk_img

def process_label(label: int, data: np.ndarray, smooth_iters: int, color_map: Dict[int, Tuple[int, int, int]]) -> vtk.vtkActor:
    """对单个标签提取等值面并返回对应 Actor。"""
    if label == 0:
        return None
    mask = np.where(data == label, label, 0).astype(data.dtype)
    vtk_img = numpy_to_vtk_image(mask)

    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(vtk_img)
    mc.SetValue(0, label)
    mc.ComputeNormalsOn()
    mc.Update()

    poly_data = mc.GetOutput()
    if smooth_iters > 0 and poly_data.GetNumberOfPoints() > 0:
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(mc.GetOutputPort())
        smoother.SetNumberOfIterations(smooth_iters)
        smoother.SetRelaxationFactor(0.1)
        smoother.Update()
        poly_data = smoother.GetOutput()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_data)
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    r, g, b = color_map.get(label, (255, 255, 255))  # Default to white if label not in map
    actor.GetProperty().SetColor(r/255.0, g/255.0, b/255.0)
    actor.GetProperty().SetOpacity(1.0)
    return actor

def render_single_view(data: np.ndarray, smooth_iters: int, color_map: Dict,
                       az: float, el: float, size=(800,800)) -> Image.Image:
    """从指定视角 (azimuth, elevation) 离屏渲染一个 3D mask。"""
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1,1,1)
    renwin = vtk.vtkRenderWindow()
    renwin.OffScreenRenderingOn()
    renwin.AddRenderer(renderer)
    renwin.SetSize(*size)

    labels = [l for l in np.unique(data) if l != 0]
    if not labels:
        print("[WARN] 数据中无有效标签，返回空白3D渲染图。")
        return Image.new('RGB', size, (255, 255, 255))
    
    with ThreadPoolExecutor() as exe:
        futures = [exe.submit(process_label, lab, data, smooth_iters, color_map) for lab in labels]
        for f in futures:
            actor = f.result()
            if actor:
                renderer.AddActor(actor)

    if renderer.GetActors().GetNumberOfItems() == 0:
        print("[WARN] 未能生成任何可渲染的 Actor，返回空白图像。")
        return Image.new('RGB', size, (255, 255, 255))

    cam = renderer.GetActiveCamera()
    renderer.ResetCamera()
    cam.Azimuth(az)
    cam.Elevation(el)
    renderer.ResetCameraClippingRange()
    renwin.Render()

    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(renwin)
    w2if.SetInputBufferTypeToRGB()
    w2if.ReadFrontBufferOff()
    w2if.Update()
    vtk_img = w2if.GetOutput()

    width, height, _ = vtk_img.GetDimensions()
    arr = numpy_support.vtk_to_numpy(vtk_img.GetPointData().GetScalars())
    arr = arr.reshape(height, width, 3)
    return Image.fromarray(np.flipud(arr))

def get_font(font_size: int) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
    """获取一个可用的字体。"""
    font_paths = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]
    for path in font_paths:
        if os.path.exists(path):
            try: return ImageFont.truetype(path, font_size)
            except IOError: continue
    return ImageFont.load_default()

def annotate_image(img: Image.Image, label: str, font_size: int = 30) -> Image.Image:
    """在图像顶部添加标题。"""
    draw = ImageDraw.Draw(img)
    font = get_font(font_size)
    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        text_w, text_h = draw.textsize(label, font=font)
        
    img_w, _ = img.size
    position = ((img_w - text_w) // 2, 10)
    draw.rectangle((position[0] - 5, position[1] - 5, position[0] + text_w + 5, position[1] + text_h + 5), fill='white')
    draw.text(position, label, fill='black', font=font)
    return img

# --- Start: New and Modified Functions for This Version ---

def create_overlay_slice(image_3d: np.ndarray, mask_3d: np.ndarray, plane: str, 
                         color_map: Dict, alpha: float) -> Image.Image:
    """
    生成带透明蒙版叠加的中央正交切片。
    """
    if image_3d.ndim != 3 or mask_3d.ndim != 3:
        raise ValueError("输入数据必须是三维的。")
        
    depth, height, width = image_3d.shape
    
    if plane == 'xy':
        img_slice = image_3d[depth // 2, :, :]
        mask_slice = mask_3d[depth // 2, :, :]
    elif plane == 'xz':
        img_slice = image_3d[:, height // 2, :]
        mask_slice = mask_3d[:, height // 2, :]
    elif plane == 'yz':
        img_slice = image_3d[:, :, width // 2]
        mask_slice = mask_3d[:, :, width // 2]
    else:
        raise ValueError("平面参数必须是 'xy', 'xz', 或 'yz'。")

    # 1. 创建基础灰度图像
    if img_slice.max() > 0:
        img_slice = (img_slice / img_slice.max() * 255).astype(np.uint8)
    base_image = Image.fromarray(img_slice).convert('RGB')

    # 2. 创建彩色透明蒙版层
    overlay = Image.new('RGBA', base_image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    alpha_int = int(alpha * 255)
    unique_labels = [l for l in np.unique(mask_slice) if l != 0]

    for label_val in unique_labels:
        color = color_map.get(label_val, (0,0,0))
        rgba_color = color + (alpha_int,)
        
        # 创建一个临时位图，其中label所在位置为255，其余为0
        label_bitmap = Image.fromarray(((mask_slice == label_val) * 255).astype(np.uint8))
        # 在这个位图上填充颜色
        overlay_draw.bitmap((0,0), label_bitmap, fill=rgba_color)
    
    # 3. 合并图像
    combined_image = Image.alpha_composite(base_image.convert('RGBA'), overlay).convert('RGB')
    return combined_image


def create_page_layout(images: List[Image.Image], titles: List[str], base_name: str) -> Image.Image:
    """将6张图和标题组合成一个3x2的网格页。"""
    if len(images) != 6 or len(titles) != 6:
        raise ValueError("需要6张图片和6个标题。")

    base_size = images[0].size
    resized_images = [img.resize(base_size, Image.Resampling.LANCZOS) for img in images]
    annotated = [annotate_image(img, title, font_size=32) for img, title in zip(resized_images, titles)]

    w, h = base_size
    grid_img = Image.new('RGB', (w * 2, h * 3 + 80), (255, 255, 255))
    
    draw = ImageDraw.Draw(grid_img)
    main_font = get_font(48)
    main_title = f"Sample: {base_name}"
    try:
        bbox = draw.textbbox((0, 0), main_title, font=main_font)
        title_w = bbox[2] - bbox[0]
    except AttributeError:
        title_w, _ = draw.textsize(main_title, font=main_font)
    draw.text(((w * 2 - title_w) // 2, 20), main_title, fill='black', font=main_font)

    # 粘贴6个子图
    grid_img.paste(annotated[0], (0, 80))        # Row 1, Col 1
    grid_img.paste(annotated[1], (w, 80))        # Row 1, Col 2
    grid_img.paste(annotated[2], (0, 80 + h))    # Row 2, Col 1
    grid_img.paste(annotated[3], (w, 80 + h))    # Row 2, Col 2
    grid_img.paste(annotated[4], (0, 80 + h*2))  # Row 3, Col 1
    grid_img.paste(annotated[5], (w, 80 + h*2))  # Row 3, Col 2

    return grid_img

def save_pages_to_pdf(pages: List[Image.Image], pdf_path: str):
    """将一组PIL图像保存为多页PDF。"""
    if not pages:
        print("[WARN] 没有可生成的页面，PDF未创建。")
        return
    pages[0].save(pdf_path, save_all=True, append_images=pages[1:], resolution=150, quality=95)
    print(f"\n[SUCCESS] PDF 已成功生成: {pdf_path}")

# --- Main execution logic ---

def main():
    p = argparse.ArgumentParser(
        description='生成包含图像三视图(带蒙版叠加)和蒙版多角度3D渲染的PDF。'
    )
    p.add_argument('-i', '--input_dir', required=True, type=str, help='包含 TIFF 文件的文件夹。')
    p.add_argument('-o', '--output_pdf', default='output_visualization.pdf', help='输出的PDF文件名。')
    p.add_argument('-s', '--smooth', type=int, default=15, help='3D渲染的平滑迭代次数 (默认 15)')
    p.add_argument('--render_size', type=int, default=600, help='每个视图的像素尺寸 (默认: 600)')
    p.add_argument('--alpha', type=float, default=0.4, help='叠加蒙版的透明度 (0.0 to 1.0, 默认: 0.4)')
    args = p.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        print("[ERROR] alpha值必须在0.0和1.0之间。", file=sys.stderr)
        sys.exit(1)

    file_pairs = []
    for fname in sorted(os.listdir(args.input_dir)):
        if fname.endswith('_im.tiff'):
            base_name = fname.removesuffix('_im.tiff')
            mask_path = os.path.join(args.input_dir, f"{base_name}_mito.tiff")
            if os.path.exists(mask_path):
                file_pairs.append((base_name, os.path.join(args.input_dir, fname), mask_path))
            else:
                print(f"[WARN] 找到 '{fname}' 但缺少蒙版 '{os.path.basename(mask_path)}'，已跳过。")
    
    if not file_pairs:
        print("[ERROR] 未找到任何匹配的 `_im.tiff` / `_mito.tiff` 文件对。", file=sys.stderr)
        return

    pdf_pages = []
    img_size = (args.render_size, args.render_size)

    for base_name, im_path, mask_path in file_pairs:
        print(f"--- Processing: {base_name} ---")
        try:
            print("  → Reading data...")
            image_data = tifffile.imread(im_path)
            mask_data = tifffile.imread(mask_path)
            
            print("  → Relabeling mask and generating colors...")
            relabeled_mask = relabel_data(mask_data)
            unique_labels = [l for l in np.unique(relabeled_mask) if l != 0]
            colors = generate_distinct_colors(len(unique_labels))
            color_map = {label: color for label, color in zip(unique_labels, colors)}
            
            print("  → Generating views...")
            # 1. 生成带蒙版叠加的2D切片
            slice_xy = create_overlay_slice(image_data, relabeled_mask, 'xy', color_map, args.alpha)
            slice_xz = create_overlay_slice(image_data, relabeled_mask, 'xz', color_map, args.alpha)
            slice_yz = create_overlay_slice(image_data, relabeled_mask, 'yz', color_map, args.alpha)
            
            # 2. 生成三个不同角度的3D渲染
            render_iso = render_single_view(relabeled_mask, args.smooth, color_map, az=45, el=30, size=img_size)
            render_top = render_single_view(relabeled_mask, args.smooth, color_map, az=0, el=90, size=img_size)
            render_side = render_single_view(relabeled_mask, args.smooth, color_map, az=90, el=0, size=img_size)

            # 3. 组合成单页
            images = [slice_xy, slice_xz, slice_yz, render_iso, render_top, render_side]
            titles = [
                "Image + Mask (XY Plane)", "Image + Mask (XZ Plane)", "Image + Mask (YZ Plane)",
                "3D Render (View 1)", "3D Render (Top View)", "3D Render (Side View)"
            ]
            
            page = create_page_layout(images, titles, base_name)
            pdf_pages.append(page)
            print(f"  → Page for '{base_name}' created.")

        except Exception as e:
            print(f"[ERROR] 处理文件 {base_name} 时发生错误: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            continue
            
    save_pages_to_pdf(pdf_pages, args.output_pdf)

if __name__ == '__main__':
    main()
