#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Debug performance issues in mask_3d_visualizer.py
"""

import time
import numpy as np
import tifffile

def analyze_mask_file(file_path):
    """Analyze the mask file to understand performance characteristics."""
    print(f"Analyzing mask file: {file_path}")
    print("-" * 50)
    
    try:
        # Load data
        start_time = time.time()
        data = tifffile.imread(file_path)
        load_time = time.time() - start_time
        
        print(f"File loaded in {load_time:.2f} seconds")
        print(f"Data shape: {data.shape}")
        print(f"Data dtype: {data.dtype}")
        print(f"Data size in memory: {data.nbytes / (1024*1024):.1f} MB")
        
        # Analyze labels
        start_time = time.time()
        unique_labels = np.unique(data)
        unique_time = time.time() - start_time
        
        non_zero_labels = unique_labels[unique_labels != 0]
        print(f"Unique labels found in {unique_time:.3f} seconds")
        print(f"Total unique labels: {len(unique_labels)}")
        print(f"Non-zero labels: {len(non_zero_labels)}")
        
        if len(non_zero_labels) > 0:
            print(f"Label range: {non_zero_labels.min()} to {non_zero_labels.max()}")
            
            # Analyze label distribution
            label_counts = {}
            for label in non_zero_labels:
                count = np.sum(data == label)
                label_counts[label] = count
            
            print(f"Label distribution:")
            for label, count in sorted(label_counts.items()):
                print(f"  Label {label}: {count:,} voxels ({count/data.size*100:.2f}%)")
        
        # Test bounding box computation
        print(f"\nTesting bounding box computation...")
        start_time = time.time()
        
        bboxes = {}
        for label in non_zero_labels[:10]:  # Test first 10 labels
            positions = np.where(data == label)
            if len(positions[0]) > 0:
                z_min, z_max = positions[0].min(), positions[0].max()
                y_min, y_max = positions[1].min(), positions[1].max()
                x_min, x_max = positions[2].min(), positions[2].max()
                bboxes[label] = (z_min, z_max, y_min, y_max, x_min, x_max)
        
        bbox_time = time.time() - start_time
        print(f"Computed bounding boxes for {len(bboxes)} labels in {bbox_time:.3f} seconds")
        
        # Estimate total time for all labels
        if len(non_zero_labels) > 10:
            estimated_time = bbox_time * len(non_zero_labels) / 10
            print(f"Estimated time for all {len(non_zero_labels)} labels: {estimated_time:.2f} seconds")
        
        return {
            'shape': data.shape,
            'dtype': data.dtype,
            'memory_mb': data.nbytes / (1024*1024),
            'num_labels': len(non_zero_labels),
            'load_time': load_time,
            'unique_time': unique_time,
            'bbox_time': bbox_time
        }
        
    except Exception as e:
        print(f"Error analyzing file: {e}")
        return None

def main():
    file_path = "/projects/weilab/liupeng/dataset/mito/han24/pos0_mito_refined.tiff"
    
    print("=" * 60)
    print("Mask File Performance Analysis")
    print("=" * 60)
    
    analysis = analyze_mask_file(file_path)
    
    if analysis:
        print(f"\nSummary:")
        print(f"  Data size: {analysis['memory_mb']:.1f} MB")
        print(f"  Number of labels: {analysis['num_labels']}")
        print(f"  Load time: {analysis['load_time']:.2f}s")
        print(f"  Unique computation: {analysis['unique_time']:.3f}s")
        print(f"  Bounding box computation: {analysis['bbox_time']:.3f}s")
        
        # Performance recommendations
        print(f"\nPerformance Recommendations:")
        if analysis['num_labels'] > 1000:
            print("  ⚠️  High number of labels detected - consider filtering small labels")
        if analysis['memory_mb'] > 100:
            print("  ⚠️  Large data size - consider downsampling for preview")
        if analysis['unique_time'] > 1.0:
            print("  ⚠️  Slow unique computation - data might be fragmented")

if __name__ == "__main__":
    main()
