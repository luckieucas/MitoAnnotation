import tifffile as tiff
import numpy as np

img = tiff.imread("/projects/weilab/liupeng/dataset/mito/wei20/mitoEM-H_test_mito.tiff")

img_0 = img[:500]

tiff.imwrite("/projects/weilab/liupeng/dataset/mito/wei20/mitoEM-H_test_mito_0.tiff", img_0, bigtiff="IF_NEEDED", compression="zlib")