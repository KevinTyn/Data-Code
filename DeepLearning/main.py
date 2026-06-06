import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.transform import Affine
from torch.utils.data import DataLoader
import torch

from models import UNet
from utils import reproject_to_union, make_overlap_mask, extract_patches, distribute_with_unet, blend_overlap
from dataset import InSARDataset
from train import train_model, sliding_predict_residual

# ========== 路径设定 ==========
tif1 = r"H:\23_113_111\vel_23_113_111.tif"
tif2 = r"H:\23_11_111\vel_23_11_111.tif"
out1_corrected = r"H:\vel1_corrected_union.tif"
out2_corrected = r"H:\vel2_corrected_union.tif"
out_mosaic     = r"H:\mosaic_vel_union.tif"

# ========== 计算并集网格 ==========
with rasterio.open(tif1) as s1, rasterio.open(tif2) as s2:
    crs1 = s1.crs
    transform1 = s1.transform
    bounds1 = s1.bounds
    profile1 = s1.profile
    px = transform1.a
    py = -transform1.e
    bounds2_in_ref = transform_bounds(s2.crs, crs1, *s2.bounds, densify_pts=21)

union_left   = min(bounds1.left,  bounds2_in_ref[0])
union_bottom = min(bounds1.bottom,bounds2_in_ref[1])
union_right  = max(bounds1.right, bounds2_in_ref[2])
union_top    = max(bounds1.top,   bounds2_in_ref[3])

union_width  = int(np.ceil((union_right - union_left) / px))
union_height = int(np.ceil((union_top - union_bottom) / py))
union_transform = Affine(px, 0, union_left, 0, -py, union_top)

print("并集网格大小:", union_height, union_width)

# ========== 重投影 ==========
arr1 = reproject_to_union(tif1, crs1, union_transform, (union_height, union_width))
arr2 = reproject_to_union(tif2, crs1, union_transform, (union_height, union_width))

mask1, mask2, overlap_mask = make_overlap_mask(arr1, arr2)

# ========== 构造训练数据 ==========
patch_size, stride = 16, 8
vel1_patches = extract_patches(arr1.filled(np.nan), overlap_mask, patch_size, stride)
vel2_patches = extract_patches(arr2.filled(np.nan), overlap_mask, patch_size, stride)

pairs = [(a, b) for a, b in zip(vel1_patches, vel2_patches) if np.isfinite(a).all() and np.isfinite(b).all()]
vel1_patches, vel2_patches = zip(*pairs) if pairs else ([], [])

print(f"训练样本数: {len(vel1_patches)}")

dataset = InSARDataset(vel1_patches, vel2_patches, normalize=True)
loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=False)

# ========== 训练 U-Net ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(in_channels=1, out_channels=1, features=(32, 64, 128)).to(device)
model = train_model(model, loader, epochs=10, lr=1e-3, device=device, lambda_smooth=0.01)

# ========== 推理残差 ==========
mean, std = dataset.mean, dataset.std
arr2_filled = arr2.filled(np.nan).astype(np.float32)
nan_mask = ~np.isfinite(arr2_filled)
fill_value = np.nanmean(arr2_filled); fill_value = 0.0 if np.isnan(fill_value) else fill_value
arr2_filled[nan_mask] = fill_value

delta_pred = sliding_predict_residual(model, arr2_filled, tile=256, overlap=64,
                                      device=device, mean=mean, std=std)

# ========== 双向分配 ==========
alpha = 0.5
arr1_corrected, arr2_corrected = distribute_with_unet(
    arr1.filled(np.nan).astype(np.float32),
    arr2_filled,
    delta_pred,
    mask1, mask2,
    alpha=alpha
)

arr1_corrected[~mask1] = np.nan
arr2_corrected[~mask2] = np.nan

# ========== 拼接 ==========
mosaic = blend_overlap(arr1_corrected, arr2_corrected, mask1, mask2, method="weighted")
nodata_out = -9999.0
mosaic_out = np.where(np.isfinite(mosaic), mosaic, nodata_out).astype(np.float32)
arr1c_out  = np.where(np.isfinite(arr1_corrected), arr1_corrected, nodata_out).astype(np.float32)
arr2c_out  = np.where(np.isfinite(arr2_corrected), arr2_corrected, nodata_out).astype(np.float32)

# ========== 写出 ==========
profile_out = profile1.copy()
profile_out.update(
    dtype=rasterio.float32,
    count=1,
    nodata=nodata_out,
    crs=crs1,
    transform=union_transform,
    width=union_width,
    height=union_height
)

with rasterio.open(out1_corrected, "w", **profile_out) as dst:
    dst.write(arr1c_out, 1)
with rasterio.open(out2_corrected, "w", **profile_out) as dst:
    dst.write(arr2c_out, 1)
with rasterio.open(out_mosaic, "w", **profile_out) as dst:
    dst.write(mosaic_out, 1)

print("✅ 输出完成：")
print("校正影像1：", out1_corrected)
print("校正影像2：", out2_corrected)
print("拼接影像：", out_mosaic)
