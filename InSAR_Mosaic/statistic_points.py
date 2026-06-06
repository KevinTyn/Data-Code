import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ===== 函数：重投影到目标格网 =====
def reproject_to_grid(src, profile):
    """重投影到目标格网并返回掩膜数组"""
    arr = np.empty((profile["height"], profile["width"]), dtype=src.dtypes[0])
    reproject(
        source=rasterio.band(src, 1),
        destination=arr,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=profile["transform"],
        dst_crs=profile["crs"],
        resampling=Resampling.nearest
    )
    return np.ma.masked_equal(arr, src.nodata)

# 输入文件


tif1 = r"E:\shanxi\18\18_11_121\vel_18_11_121.tif"
tif2 = r"E:\shanxi\18\18_113_121\vel_18_113_121.tif"
coh1 = r"E:\shanxi\18\18_11_121\coh_18_11_121.tif"
coh2 = r"E:\shanxi\18\18_113_121\coh_18_113_121.tif"



src1, src2 = rasterio.open(tif1), rasterio.open(tif2)
src_coh1, src_coh2 = rasterio.open(coh1), rasterio.open(coh2)

# ===== 交集范围 =====
xmin = max(src1.bounds.left, src2.bounds.left)
ymin = max(src1.bounds.bottom, src2.bounds.bottom)
xmax = min(src1.bounds.right, src2.bounds.right)
ymax = min(src1.bounds.top, src2.bounds.top)

# 以第一个tif的分辨率/投影为基准
res, crs = src1.res, src1.crs
width, height = int((xmax - xmin) / res[0]), int((ymax - ymin) / res[1])
transform = from_origin(xmin, ymax, *res)

profile = src1.profile.copy()
profile.update({"height": height, "width": width, "transform": transform, "crs": crs})

# ===== 重投影到交集格网 =====
arr1 = reproject_to_grid(src1, profile)
arr2 = reproject_to_grid(src2, profile)
coh_arr1 = reproject_to_grid(src_coh1, profile)
coh_arr2 = reproject_to_grid(src_coh2, profile)

# ===== 掩膜：只保留共同有效区域 =====
mask = (~arr1.mask) & (~arr2.mask)
arr1 = np.where(mask, arr1, profile["nodata"])
arr2 = np.where(mask, arr2, profile["nodata"])
coh_arr1 = np.where(mask, coh_arr1, profile["nodata"])
coh_arr2 = np.where(mask, coh_arr2, profile["nodata"])

# ===== 粗差剔除（仅对速度场） =====
diff = np.ma.masked_equal(arr1, profile["nodata"]) - np.ma.masked_equal(arr2, profile["nodata"])
mu, sigma = diff.mean(), diff.std()
k = 3  # 阈值倍数，可调
mask_outlier = np.abs(diff - mu) > k * sigma

arr1 = np.where(mask_outlier, profile["nodata"], arr1)
arr2 = np.where(mask_outlier, profile["nodata"], arr2)




# ===== 相干性条件与速度差 =====
coh_min = np.minimum(coh_arr1, coh_arr2)
valid_mask = (arr1 != profile["nodata"]) & (arr2 != profile["nodata"]) & (coh_min != profile["nodata"])
vel_diff = np.where(valid_mask, arr1 - arr2, np.nan)

# ===== 四个相干性区间统计 =====
bins = [(0,0.25), (0.25,0.5), (0.5,0.75), (0.75,1.01)]
stats = {}

for lo, hi in bins:
    mask_bin = valid_mask & (coh_min >= lo) & (coh_min < hi)
    vals = vel_diff[mask_bin]
    stats[(lo,hi)] = {
        "mean": np.nanmean(vals),
        "std": np.nanstd(vals),
        "count": np.sum(mask_bin)
    }

# 打印结果
for k,v in stats.items():
    print(f"相干性 {k[0]}–{k[1]} 区间: "
          f"均值={v['mean']:.3f}, 标准差={v['std']:.3f}, 样本数={v['count']}")

# ===== 可视化对比 =====
plt.figure(figsize=(8,6))
labels, means, stds = [], [], []
for k,v in stats.items():
    labels.append(f"{k[0]}–{k[1]}")
    means.append(v["mean"])
    stds.append(v["std"])

plt.bar(labels, means, yerr=stds, capsize=5, color=["#d73027","#fc8d59","#fee090","#91bfdb"])
plt.axhline(0, color="k", linestyle="--")
plt.ylabel("速度差 (arr1 - arr2)")
plt.title("不同相干性区间的速度差统计")
plt.show()


# 打印结果
print("=== 各相干性区间速度差统计 ===")
for k,v in stats.items():
    print(f"相干性 {k[0]}–{k[1]} 区间: "
          f"有效值数={v['count']}, "
          f"均值={v['mean']:.3f}, "
          f"标准差={v['std']:.3f}")

# 粗差剔除后有效点总数
total_valid = np.sum(valid_mask)

# 四个区间的总和
total_bins = sum(v["count"] for v in stats.values())

print("粗差剔除后有效点总数:", total_valid)
print("四个区间有效点总和:", total_bins)
print("是否一致:", total_valid == total_bins)
