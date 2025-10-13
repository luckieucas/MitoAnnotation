import tifffile as tiff
import numpy as np

img = tiff.imread("/projects/weilab/liupeng/dataset/mito/wei20/mitoEM-H_den_im.tiff")

img_0 = img[:500]

tiff.imwrite("/projects/weilab/liupeng/dataset/mito/wei20/mitoEM-H_den_im_0.tiff", img_0, bigtiff="IF_NEEDED", compression="zlib")