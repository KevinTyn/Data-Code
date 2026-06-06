import numpy as np
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling as RioResampling

def reproject_to_union(src_path, ref_crs, ref_transform, ref_shape, resampling=RioResampling.bilinear):
    with rasterio.open(src_path) as src:
        dst = np.full(ref_shape, np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=resampling,
            src_nodata=src.nodata,
            dst_nodata=np.nan
        )
        return np.ma.masked_invalid(dst)

def make_overlap_mask(arr1, arr2):
    mask1 = ~arr1.mask if np.ma.isMaskedArray(arr1) else ~np.isnan(arr1)
    mask2 = ~arr2.mask if np.ma.isMaskedArray(arr2) else ~np.isnan(arr2)
    return mask1.astype(bool), mask2.astype(bool), (mask1 & mask2)

def extract_patches(arr, mask, patch_size=128, stride=128):
    H, W = arr.shape
    patches = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            m = mask[y:y+patch_size, x:x+patch_size]
            if m.sum() == patch_size * patch_size:
                p = arr[y:y+patch_size, x:x+patch_size]
                patches.append(p.astype(np.float32))
    return patches

def distribute_with_unet(arr1, arr2, delta_pred, mask1, mask2, alpha=0.5):
    arr1c, arr2c = arr1.copy(), arr2.copy()
    overlap = mask1 & mask2
    arr1c[overlap] = arr1[overlap] + alpha * delta_pred[overlap]
    arr2c[overlap] = arr2[overlap] - (1 - alpha) * delta_pred[overlap]
    return arr1c, arr2c

def blend_overlap(arr1, arr2c, mask1, mask2, method="weighted", nodata_val=-9999.0):
    H, W = arr1.shape
    out = np.full((H, W), nodata_val, dtype=np.float32)
    overlap = mask1 & mask2
    out[mask1 & ~mask2] = arr1[mask1 & ~mask2]
    out[mask2 & ~mask1] = arr2c[mask2 & ~mask1]
    if method == "ref":
        out[overlap] = arr1[overlap]
    elif method == "mean":
        out[overlap] = 0.5 * (arr1[overlap] + arr2c[overlap])
    else:  # weighted
        d1, d2 = mask1.astype(np.float32), mask2.astype(np.float32)
        s = d1 + d2
        s[s == 0] = 1.0
        alpha = d1 / s
        out[overlap] = alpha[overlap] * arr1[overlap] + (1.0 - alpha[overlap]) * arr2c[overlap]
    return out
