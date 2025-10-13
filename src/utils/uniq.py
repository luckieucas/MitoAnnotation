import numpy as np
def unique_uint_bitmap(memmap_or_ndarray):
    a = memmap_or_ndarray
    kind = a.dtype
    if kind == np.uint8:
        presence = np.zeros(256, dtype=bool)
    elif kind == np.uint16:
        presence = np.zeros(65536, dtype=bool)
    else:
        raise ValueError("Use other methods for this dtype")

    flat = a.ravel(order='K')
    # 分块设置存在位
    step = 64_000_000
    for i in range(0, flat.size, step):
        presence[flat[i:i+step]] = True
    return np.flatnonzero(presence)
