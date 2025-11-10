#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont
import PIL
from typing import Union, Optional
from tqdm import tqdm
# VTK imports
import vtk
from vtkmodules.util import numpy_support

# Color map for different labels
LABEL_COLORMAP = [
    (255,   0,   0),  # Red
    (  0, 255,   0),  # Green
    (  0,   0, 255),  # Blue
    (255, 255,   0),  # Yellow
    (255,   0, 255),  # Magenta
    (  0, 255, 255),  # Cyan
    (255, 165,   0),  # Orange
    (128,   0, 128),  # Purple
    (255, 192, 203),  # Pink
    (  0, 128,   0),  # Dark Green
    (128, 128, 128),  # Gray
    (255, 215,   0),  # Gold
]

def relabel_data(data: np.ndarray):
    """Remap all non-zero labels to 1,2,3... and return new data"""
    unique = np.unique(data)
    unique = unique[unique != 0]
    mapping = {old: new for new, old in enumerate(unique, start=1)}
    out = np.zeros_like(data, dtype=np.uint8)
    for old, new in mapping.items():
        out[data == old] = new
    return out

def numpy_to_vtk_image(data: np.ndarray, spacing: tuple = (1.0, 1.0, 1.0)) -> vtk.vtkImageData:
    """Convert 3D numpy array to vtkImageData with specified spacing."""
    data = np.ascontiguousarray(data)
    depth, height, width = data.shape
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(width, height, depth)
    vtk_img.SetSpacing(spacing[2], spacing[1], spacing[0])  # VTK uses (x, y, z) order
    vtk_array = numpy_support.numpy_to_vtk(
        num_array=data.ravel(order='C'),
        deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR
    )
    vtk_img.GetPointData().SetScalars(vtk_array)
    return vtk_img

def get_all_label_bounding_boxes(data: np.ndarray) -> dict:
    """Get bounding box coordinates for all labels in one pass."""
    bboxes = {}
    unique_labels = np.unique(data)
    
    for label in unique_labels:
        if label == 0:
            continue
            
        # Find all positions where the label exists
        positions = np.where(data == label)
        if len(positions[0]) == 0:
            continue
        
        # Get min and max coordinates
        z_min, z_max = positions[0].min(), positions[0].max()
        y_min, y_max = positions[1].min(), positions[1].max()
        x_min, x_max = positions[2].min(), positions[2].max()
        
        bboxes[label] = (z_min, z_max, y_min, y_max, x_min, x_max)
    
    return bboxes

def get_label_bounding_box(data: np.ndarray, label: int) -> tuple:
    """Get bounding box coordinates for a specific label."""
    if label == 0:
        return None
    
    # Find all positions where the label exists
    positions = np.where(data == label)
    if len(positions[0]) == 0:
        return None
    
    # Get min and max coordinates
    z_min, z_max = positions[0].min(), positions[0].max()
    y_min, y_max = positions[1].min(), positions[1].max()
    x_min, x_max = positions[2].min(), positions[2].max()
    
    return (z_min, z_max, y_min, y_max, x_min, x_max)

def create_bounding_box_actor(bbox: tuple, label: int, spacing: tuple = (1.0, 1.0, 1.0)) -> vtk.vtkActor:
    """Create a wireframe bounding box actor for a label."""
    if bbox is None:
        return None
    
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    
    # Convert to physical coordinates using spacing
    x_min_phys = x_min * spacing[2]
    x_max_phys = (x_max + 1) * spacing[2]
    y_min_phys = y_min * spacing[1]
    y_max_phys = (y_max + 1) * spacing[1]
    z_min_phys = z_min * spacing[0]
    z_max_phys = (z_max + 1) * spacing[0]
    
    # Create a cube source
    cube = vtk.vtkCubeSource()
    cube.SetXLength(x_max_phys - x_min_phys)
    cube.SetYLength(y_max_phys - y_min_phys)
    cube.SetZLength(z_max_phys - z_min_phys)
    cube.SetCenter(
        (x_min_phys + x_max_phys) / 2,
        (y_min_phys + y_max_phys) / 2,
        (z_min_phys + z_max_phys) / 2
    )
    cube.Update()
    
    # Create mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(cube.GetOutputPort())
    
    # Create actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    # Set color and properties
    r, g, b = LABEL_COLORMAP[label % len(LABEL_COLORMAP)]
    actor.GetProperty().SetColor(r/255.0, g/255.0, b/255.0)
    actor.GetProperty().SetOpacity(0.3)  # Semi-transparent
    actor.GetProperty().SetRepresentationToWireframe()  # Wireframe only
    actor.GetProperty().SetLineWidth(2.0)  # Thicker lines
    
    return actor


def render_single_label_view(bbox: tuple, label: int, az: float, el: float,
                            size=(800, 600), spacing: tuple = (1.0, 1.0, 1.0)) -> Image.Image:
    """
    Off-screen render 3D bounding box for a single label from one viewpoint, return PIL.Image.
    bbox: tuple of (z_min, z_max, y_min, y_max, x_min, x_max)
    label: label ID
    az: horizontal rotation angle, el: elevation angle.
    spacing: voxel spacing in (z, y, x) order (depth, height, width).
    """
    if bbox is None:
        print(f"[WARN] No bounding box for label {label}, cannot render.")
        return Image.new('RGB', size, (255, 255, 255))
    
    # VTK renderer and window
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)
    renwin = vtk.vtkRenderWindow()
    renwin.OffScreenRenderingOn()
    renwin.AddRenderer(renderer)
    renwin.SetSize(*size)

    # Add actor for single label
    actor = create_bounding_box_actor(bbox, label, spacing)
    if actor:
        renderer.AddActor(actor)
    else:
        raise RuntimeError(f"No renderable actor for label {label}.")

    # Set camera viewpoint
    cam = renderer.GetActiveCamera()
    renderer.ResetCamera()
    cam.Azimuth(az)
    cam.Elevation(el)
    renderer.ResetCameraClippingRange()
    renwin.Render()

    # Capture image
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(renwin)
    w2if.SetInputBufferTypeToRGB()
    w2if.ReadFrontBufferOff()
    w2if.Update()
    vtk_img = w2if.GetOutput()

    # Convert to numpy then PIL.Image
    width, height, _ = vtk_img.GetDimensions()
    arr = numpy_support.vtk_to_numpy(vtk_img.GetPointData().GetScalars())
    arr = arr.reshape(height, width, 3)
    return Image.fromarray(arr)


def get_scalable_font(font_size: int) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
    """
    Try to return a scalable TTF font.
    """
    # 1) Pillow built-in fonts
    try:
        from PIL import features
        if features.check("raqm"):
             pil_fonts_path = os.path.join(os.path.dirname(features.get_supported_modules()["raqm"].__file__), "DejaVuSans.ttf")
             if os.path.exists(pil_fonts_path):
                 return ImageFont.truetype(pil_fonts_path, font_size)

        # Traditional method
        pil_fonts_dir = os.path.join(os.path.dirname(PIL.__file__), "fonts")
        candidate = os.path.join(pil_fonts_dir, "DejaVuSans.ttf")
        if os.path.exists(candidate):
            return ImageFont.truetype(candidate, font_size)
    except Exception:
        pass

    # 2) Common Linux/macOS paths
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    
    # 3) Windows paths
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

    # 4) Final fallback
    print(f"[WARN] Cannot find any scalable TTF fonts, will use default font.")
    try:
        return ImageFont.load_default(size=font_size)
    except TypeError:
        if font_size > 12:
             print("[WARN] Current Pillow version is too old, cannot set default font size.")
        return ImageFont.load_default()

def annotate_image(img: Image.Image,
                   label: str,
                   top_margin: int = 8,
                   font_path: Optional[str] = None,
                   font_size: int = 36,
                   text_color=(0, 0, 0),
                   bg_color=(255, 255, 255)) -> Image.Image:
    """
    Draw label text (with background) at the top center of img.
    """
    draw = ImageDraw.Draw(img)
    font = None

    # Load font
    if font_path and os.path.exists(font_path):
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            print(f"[WARN] Cannot load font from '{font_path}', will try automatic search.")

    if font is None:
        font = get_scalable_font(font_size)

    # Calculate text dimensions
    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        text_w, text_h = draw.textsize(label, font=font)

    # Calculate position
    img_w, img_h = img.size
    x = (img_w - text_w) // 2
    y = top_margin

    # Draw background and text
    pad = max(4, font_size // 4)
    draw.rectangle(
        [(x - pad, y - pad), (x + text_w + pad, y + text_h + pad)],
        fill=bg_color
    )
    draw.text((x, y - bbox[1]), label, fill=text_color, font=font)
    
    return img

def generate_viewpoints(num_views: int = 8):
    """
    Generate multiple viewpoints for 3D visualization.
    Returns a list of (azimuth, elevation) tuples.
    """
    viewpoints = []
    
    if num_views == 4:
        # Standard 4 views
        viewpoints = [
            (0, 0),      # Front
            (90, 0),     # Side
            (0, 90),     # Top
            (45, 45),    # Isometric
        ]
    elif num_views == 8:
        # 8 views around the object
        for i in range(8):
            az = i * 45  # 0, 45, 90, 135, 180, 225, 270, 315
            el = 30 if i % 2 == 0 else 60  # Alternate between 30 and 60 degrees
            viewpoints.append((az, el))
    elif num_views == 12:
        # 12 views for more comprehensive coverage
        for i in range(12):
            az = i * 30  # Every 30 degrees
            el = 20 + (i % 3) * 20  # 20, 40, 60 degrees in cycles
            viewpoints.append((az, el))
    else:
        # Custom number of views
        for i in range(num_views):
            az = i * (360 / num_views)
            el = 30 + (i % 4) * 15  # Vary elevation
            viewpoints.append((az, el))
    
    return viewpoints


def make_pdf_one_page_per_label(all_label_images: dict, output_path: str,
                               font_size: int = 48, title_margin: int = 20):
    """
    Create a PDF where each label gets one page with all its viewpoints.
    all_label_images: dict of {label: [list of images]}
    """
    if not all_label_images:
        print("[WARN] No label images provided, PDF is empty.")
        return

    pages = []
    
    for label, images in all_label_images.items():
        if not images:
            print(f"[WARN] No images for label {label}, skipping.")
            continue
            
        print(f"[INFO] Creating page for label {label} with {len(images)} views")
        
        # Create one page for this label
        title = f"Label {label} - 3D Visualization"
        
        # Arrange images in a grid on one page
        if len(images) == 1:
            # Single image - use it directly
            canvas = images[0].copy()
        else:
            # Multiple images - arrange in grid
            img_w, img_h = images[0].size
            
            # Calculate grid dimensions
            if len(images) <= 4:
                grid_w, grid_h = 2, 2
            elif len(images) <= 6:
                grid_w, grid_h = 3, 2
            elif len(images) <= 9:
                grid_w, grid_h = 3, 3
            else:
                grid_w, grid_h = 4, 3
            
            # Create page canvas
            page_w = img_w * grid_w
            page_h = img_h * grid_h + 100  # Extra space for title
            
            canvas = Image.new('RGB', (page_w, page_h), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)
            
            # Add title
            font = get_scalable_font(font_size)
            try:
                bbox = draw.textbbox((0, 0), title, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            except AttributeError:
                text_w, text_h = font.getsize(title)
            
            x = (page_w - text_w) // 2
            y = title_margin
            draw.text((x, y), title, fill=(0, 0, 0), font=font)
            
            # Arrange images in grid
            for i, img in enumerate(images):
                if i >= grid_w * grid_h:  # Limit to grid size
                    break
                row = i // grid_w
                col = i % grid_w
                x_pos = col * img_w
                y_pos = 100 + row * img_h  # Start after title area
                canvas.paste(img, (x_pos, y_pos))
        
        pages.append(canvas)

    if not pages:
        print("[ERROR] No pages created for PDF")
        return

    # Save PDF
    pages[0].save(
        output_path,
        save_all=True,
        append_images=pages[1:],
        resolution=300
    )
    print(f"[INFO] Generated PDF with {len(pages)} pages (one per label): {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='3D bounding box visualization of mask labels - each label gets one page in a single PDF.'
    )
    parser.add_argument('--input', '-i', required=True, type=str,
                       help='Input mask file (TIFF format)')
    parser.add_argument('--output', '-o', required=True, type=str,
                       help='Output PDF file path')
    parser.add_argument('--spacing', nargs=3, type=float, default=[1.0, 1.0, 1.0],
                       help='Voxel spacing in z, y, x order (default: 1.0 1.0 1.0)')
    parser.add_argument('--views', '-v', type=int, default=8,
                       help='Number of viewpoints to render (default: 8)')
    parser.add_argument('--size', nargs=2, type=int, default=[800, 600],
                       help='Image size width height (default: 800 600)')
    parser.add_argument('--title', '-t', type=str, default="3D Visualization",
                       help='Title for the PDF (default: "3D Visualization")')
    parser.add_argument('--max-labels', type=int, default=None,
                       help='Maximum number of labels to process (for testing)')
    
    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    spacing = tuple(args.spacing)
    print(f"[INFO] Using voxel spacing: {spacing} (z, y, x)")
    print(f"[INFO] Rendering {args.views} viewpoints")
    print(f"[INFO] Image size: {args.size[0]}x{args.size[1]}")

    # Load mask data
    try:
        data = tifffile.imread(args.input)
        print(f"[INFO] Loaded mask data: {data.shape}, dtype: {data.dtype}")
    except Exception as e:
        print(f"[ERROR] Failed to load mask file: {e}")
        sys.exit(1)

    # Validate data
    if data.ndim != 3:
        print(f"[ERROR] Input data must be 3D, got {data.ndim}D")
        sys.exit(1)

    if not np.issubdtype(data.dtype, np.integer):
        print(f"[WARN] Converting data from {data.dtype} to uint8")
        data = data.astype(np.uint8)

    # Pre-process data and compute bounding boxes once
    print("[INFO] Pre-processing data and computing bounding boxes...")
    data = relabel_data(data)
    bboxes = get_all_label_bounding_boxes(data)
    
    if not bboxes:
        print("[ERROR] No non-zero labels found in the mask")
        sys.exit(1)
    
    # Limit number of labels if specified
    if args.max_labels and len(bboxes) > args.max_labels:
        print(f"[INFO] Limiting to first {args.max_labels} labels (out of {len(bboxes)})")
        bboxes = dict(list(bboxes.items())[:args.max_labels])
    
    print(f"[INFO] Found {len(bboxes)} labels with bounding boxes: {list(bboxes.keys())}")

    # Generate viewpoints
    viewpoints = generate_viewpoints(args.views)
    print(f"[INFO] Generated {len(viewpoints)} viewpoints")

    # Generate single PDF with one page per label
    print(f"[INFO] Generating single PDF with one page per label")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Collect images for all labels
    all_label_images = {}
    
    for label_idx, (label, bbox) in tqdm(enumerate(bboxes.items()), total=len(bboxes)):
        print(f"\n[INFO] Processing label {label} ({label_idx+1}/{len(bboxes)})")
        
        # Render images for this label
        images = []
        for i, (az, el) in enumerate(viewpoints):
            print(f"  Rendering view {i+1}/{len(viewpoints)}: az={az:.1f}°, el={el:.1f}°")
            try:
                img = render_single_label_view(bbox, label, az, el, 
                                             size=tuple(args.size), spacing=spacing)
                
                # Add viewpoint annotation
                view_label = f"View {i+1}\nAz: {az:.0f}° El: {el:.0f}°"
                img = annotate_image(img, view_label, font_size=24)
                images.append(img)
            except Exception as e:
                print(f"  [ERROR] Failed to render view {i+1} for label {label}: {e}")
                continue
        
        if images:
            all_label_images[label] = images
            print(f"  [INFO] Collected {len(images)} images for label {label}")
        else:
            print(f"  [WARN] No images generated for label {label}")
    
    if all_label_images:
        # Generate PDF with one page per label
        print(f"\n[INFO] Creating PDF with {len(all_label_images)} pages (one per label)...")
        make_pdf_one_page_per_label(all_label_images, args.output)
        print(f"[INFO] Done! PDF saved to: {args.output}")
    else:
        print("[ERROR] No images were successfully rendered for any label")
        sys.exit(1)

if __name__ == '__main__':
    main()
