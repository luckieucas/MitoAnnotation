#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Performance test script for mask_3d_visualizer.py
Tests the optimized bounding box visualization performance.
"""

import time
import subprocess
import sys
import os

def test_performance():
    """Test the performance of the optimized visualizer."""
    
    # Test parameters
    input_file = "/projects/weilab/liupeng/dataset/mito/han24/pos0_mito_refined.tiff"
    output_file = "/tmp/test_performance.pdf"
    
    # Test with different view counts
    test_cases = [
        {"views": 4, "size": "400 300", "desc": "4 views, small size"},
        {"views": 8, "size": "600 400", "desc": "8 views, medium size"},
        {"views": 12, "size": "800 600", "desc": "12 views, large size"},
    ]
    
    print("=" * 60)
    print("Performance Test for Mask 3D Bounding Box Visualizer")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['desc']}")
        print("-" * 40)
        
        cmd = [
            "python", "mask_3d_visualizer.py",
            "--input", input_file,
            "--output", output_file,
            "--views", str(test_case["views"]),
            "--size", test_case["size"],
            "--spacing", "30", "8", "8",
            "--title", f"Performance Test {i}"
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        # Measure execution time
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd=os.path.dirname(__file__), timeout=300)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            if result.returncode == 0:
                print(f"✅ SUCCESS: Completed in {execution_time:.2f} seconds")
                print(f"   Average per view: {execution_time/test_case['views']:.2f} seconds")
                
                # Check output file size
                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file)
                    print(f"   Output PDF size: {file_size/1024:.1f} KB")
                else:
                    print("   ⚠️  Output file not found")
            else:
                print(f"❌ FAILED: Return code {result.returncode}")
                print(f"   Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("❌ TIMEOUT: Process took longer than 5 minutes")
        except Exception as e:
            print(f"❌ ERROR: {e}")
        
        # Clean up output file
        if os.path.exists(output_file):
            os.remove(output_file)
    
    print("\n" + "=" * 60)
    print("Performance Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    test_performance()
