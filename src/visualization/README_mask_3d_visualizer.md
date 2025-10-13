# Mask 3D Bounding Box Visualizer

这个脚本用于对3D mask文件中的每个标签进行3D边界框可视化，支持多视角采样并导出为PDF。每个标签生成一页PDF，所有标签合并为一个大的PDF文件。

## 功能特点

- 支持3D TIFF格式的mask文件
- 自动识别mask中的所有非零标签
- 为每个标签生成3D边界框（wireframe）
- 支持多视角渲染（4、8、12个视角或自定义）
- 可调节体素间距
- 每个标签作为PDF的一页，所有标签合并为一个大PDF
- 半透明线框显示，便于观察重叠区域

## 使用方法

### 基本用法

```bash
# 基本使用：每个标签一页，合并为一个大PDF
python mask_3d_visualizer.py -i input_mask.tiff -o output.pdf
```

### 完整参数

```bash
python mask_3d_visualizer.py \
    --input input_mask.tiff \
    --output output.pdf \
    --spacing 1.0 1.0 1.0 \
    --views 8 \
    --size 800 600 \
    --title "My 3D Visualization"
```

## 参数说明

- `--input, -i`: 输入的3D mask文件路径（TIFF格式）
- `--output, -o`: 输出的PDF文件路径
- `--spacing`: 体素间距，按z, y, x顺序（默认1.0 1.0 1.0）
- `--views, -v`: 渲染视角数量（默认8）
- `--size`: 图像尺寸，宽度 高度（默认800 600）
- `--title, -t`: PDF标题（默认"3D Visualization"）
- `--max-labels`: 限制处理的标签数量（用于测试）

## 视角配置

脚本支持不同的视角配置：

- **4个视角**: 前视图、侧视图、顶视图、等轴视图
- **8个视角**: 围绕物体每45度一个视角，高度角在30°和60°之间交替
- **12个视角**: 每30度一个视角，高度角在20°、40°、60°之间循环
- **自定义**: 任意数量的视角，均匀分布

## 输出格式

- 每个标签生成一页PDF，包含该标签的所有视角
- 边界框以半透明线框形式显示，便于观察
- 图像上标注视角信息（方位角和高度角）
- 所有标签的页面合并为一个大的PDF文件
- 每个页面自动调整网格布局（2x2, 3x2, 3x3, 4x3等）

## 依赖要求

- Python 3.6+
- VTK
- PIL (Pillow)
- numpy
- tifffile

## 示例

```bash
# 基本使用
python mask_3d_visualizer.py -i segmentation.tiff -o result.pdf

# 高分辨率，12个视角
python mask_3d_visualizer.py -i segmentation.tiff -o result.pdf --views 12 --size 1200 900

# 自定义体素间距和平滑
python mask_3d_visualizer.py -i segmentation.tiff -o result.pdf --spacing 0.5 0.5 1.0 --smooth 50
```

## 注意事项

1. 输入文件必须是3D的TIFF格式
2. 标签值必须是非零整数
3. 如果标签数量超过颜色映射数量，会循环使用颜色
4. 建议使用合适的体素间距以获得最佳视觉效果
5. 边界框可视化比等值面可视化更快，适合快速预览
6. 半透明线框设计便于观察多个标签的空间关系
