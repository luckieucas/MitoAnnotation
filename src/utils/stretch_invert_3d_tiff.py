#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import tifffile as tiff

def stretch_invert_small(in_path, out_path, p_low=1.0, p_high=99.0, compression="zlib", use_stretch=True):
    # 读入整卷 (Z, Y, X)
    vol = tiff.imread(in_path)
    if vol.ndim != 3:
        raise ValueError(f"Expect 3D (Z,Y,X), got {vol.shape}")
    orig_dtype = vol.dtype

    # 保存背景掩码（原始值为0的位置）
    background_mask = (vol == 0)

    if use_stretch:
        # 全局分位点 + 对比度拉伸到 [0,1]
        p1, p99 = np.percentile(vol, [p_low, p_high])
        eps = max(p99 - p1, 1e-6)
        vol_f = vol.astype(np.float32, copy=False)
        vol_f = np.clip((vol_f - p1) / eps, 0.0, 1.0)
    else:
        # 不做stretch，直接归一化到 [0,1]
        vol_f = vol.astype(np.float32, copy=False)
        if np.issubdtype(orig_dtype, np.integer):
            maxv = np.iinfo(orig_dtype).max
            vol_f = vol_f / maxv
        else:
            minv, maxv = vol_f.min(), vol_f.max()
            eps = max(maxv - minv, 1e-6)
            vol_f = (vol_f - minv) / eps
        p1, p99 = None, None  # 不使用stretch时返回None

    # 反相并回写 dtype（整数保持位宽；浮点输出 float32）
    if np.issubdtype(orig_dtype, np.integer):
        maxv = np.iinfo(orig_dtype).max
        out = (maxv - np.rint(vol_f * maxv)).astype(orig_dtype, copy=False)
    else:
        out = (1.0 - vol_f).astype(np.float32, copy=False)
    
    # 将背景（原始值为0的位置）设置回0
    out[background_mask] = 0

    # 一次性写为 ImageJ 栈（常见查看器能识别为 3D）
    tiff.imwrite(
        out_path, out,
        imagej=True, photometric="minisblack",
        metadata={"axes": "ZYX"},
        bigtiff=True,
        compression=None if (compression is None or str(compression).lower() == "none") else compression
    )
    return p1, p99

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="3D TIFF: global p1–p99 stretch + invert (small data).")
    ap.add_argument("--input", required=True, help="输入 3D TIFF 路径（形状 Z,Y,X）")
    ap.add_argument("--output", required=True, help="输出 3D TIFF 路径")
    ap.add_argument("--p_low", type=float, default=1.0, help="下分位点，默认1")
    ap.add_argument("--p_high", type=float, default=99.0, help="上分位点，默认99")
    ap.add_argument("--compression", type=str, default="zlib", help="压缩：zlib/lzw/none")
    ap.add_argument("--no_stretch", action="store_true", help="不使用对比度拉伸，直接反相")
    args = ap.parse_args()

    p1, p99 = stretch_invert_small(args.input, args.output, args.p_low, args.p_high, args.compression, use_stretch=not args.no_stretch)
    if p1 is not None and p99 is not None:
        print(f"Estimated percentiles: p{args.p_low}={p1:.6g}, p{args.p_high}={p99:.6g}")
    else:
        print("No stretch applied (direct invert).")
    print(f"Saved -> {args.output}")
