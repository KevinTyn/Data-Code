import numpy as np
import rasterio
from rasterio.warp import reproject, transform_bounds
from rasterio.enums import Resampling as RioResampling
from rasterio.transform import Affine
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scipy.ndimage as ndi

# ================== UNet ==================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, features=(32,64,128)):
        super().__init__()
        self.downs, self.pools = nn.ModuleList(), nn.ModuleList()
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            self.pools.append(nn.MaxPool2d(2))
            ch = f
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        rev = list(reversed(features))
        self.ups, self.upconvs = nn.ModuleList(), nn.ModuleList()
        ch = features[-1]*2
        for f in rev:
            self.upconvs.append(nn.ConvTranspose2d(ch, f, 2, 2))
            self.ups.append(DoubleConv(ch, f))
            ch = f
        self.final = nn.Conv2d(rev[-1], out_channels, 1)

    def forward(self, x):
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x); skips.append(x); x = pool(x)
        x = self.bottleneck(x)
        for upconv, up, skip in zip(self.upconvs, self.ups, reversed(skips)):
            x = upconv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = up(torch.cat([skip, x], dim=1))
        return self.final(x)

# ================== IO utils ==================
def reproject_to_union(src_path, ref_crs, ref_transform, ref_shape, resampling=RioResampling.bilinear, as_masked=True):
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
        return np.ma.masked_invalid(dst) if as_masked else dst

def make_overlap_mask(arr1, arr2):
    mask1 = ~arr1.mask if np.ma.isMaskedArray(arr1) else ~np.isnan(arr1)
    mask2 = ~arr2.mask if np.ma.isMaskedArray(arr2) else ~np.isnan(arr2)
    return mask1.astype(bool), mask2.astype(bool), (mask1 & mask2)

def union_grid(profile_ref, bounds_ref, crs_ref, other_path):
    with rasterio.open(other_path) as s2:
        bounds2_in_ref = transform_bounds(s2.crs, crs_ref, *s2.bounds, densify_pts=21)
    px = profile_ref['transform'].a
    py = -profile_ref['transform'].e
    union_left   = min(bounds_ref.left,  bounds2_in_ref[0])
    union_bottom = min(bounds_ref.bottom,bounds2_in_ref[1])
    union_right  = max(bounds_ref.right, bounds2_in_ref[2])
    union_top    = max(bounds_ref.top,   bounds2_in_ref[3])
    union_width  = int(np.ceil((union_right - union_left) / px))
    union_height = int(np.ceil((union_top - union_bottom) / py))
    union_transform = Affine(px, 0, union_left, 0, -py, union_top)
    return union_transform, union_height, union_width

# ================== alpha maps ==================
def distance_weight_alpha(mask1, mask2):
    overlap = mask1 & mask2
    d1 = ndi.distance_transform_edt(mask1)
    d2 = ndi.distance_transform_edt(mask2)
    s = d1 + d2
    s[s == 0] = 1.0
    alpha_d = d2 / s
    return alpha_d.astype(np.float32), overlap

def adaptive_alpha(mask1, mask2, coh1, coh2, beta_alpha=0.5):
    alpha_d, overlap = distance_weight_alpha(mask1, mask2)
    # coherence-based reliability split: higher coherence side moves less
    denom = (coh1 + coh2 + 1e-6)
    alpha_r = coh2 / denom
    alpha = np.clip(beta_alpha * alpha_d + (1 - beta_alpha) * alpha_r, 0.1, 0.9).astype(np.float32)
    return alpha, overlap

# ================== patching ==================
def extract_patches(arr1, arr2, coh1, coh2, mask, alpha, patch_size=128, stride=128):
    H, W = arr1.shape
    v1_p, v2_p, c1_p, c2_p, m_p, a_p = [], [], [], [], [], []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            m = mask[y:y+patch_size, x:x+patch_size]
            if m.sum() == patch_size * patch_size:
                v1_p.append(arr1[y:y+patch_size, x:x+patch_size].astype(np.float32))
                v2_p.append(arr2[y:y+patch_size, x:x+patch_size].astype(np.float32))
                c1_p.append(coh1[y:y+patch_size, x:x+patch_size].astype(np.float32))
                c2_p.append(coh2[y:y+patch_size, x:x+patch_size].astype(np.float32))
                m_p.append(m.astype(np.float32))
                a_p.append(alpha[y:y+patch_size, x:x+patch_size].astype(np.float32))
    return v1_p, v2_p, c1_p, c2_p, m_p, a_p

# ================== dataset ==================
class InSARDataset(Dataset):
    def __init__(self, v1_patches, v2_patches, coh1_patches, coh2_patches, mask_patches, alpha_patches, normalize=True):
        self.v1, self.v2 = v1_patches, v2_patches
        self.c1, self.c2 = coh1_patches, coh2_patches
        self.mask, self.alpha = mask_patches, alpha_patches
        self.normalize = normalize
        if normalize and len(self.v1) > 0:
            cat = np.concatenate([p.flatten() for p in self.v1 + self.v2])
            self.mean, self.std = float(np.mean(cat)), float(np.std(cat) + 1e-6)
        else:
            self.mean, self.std = 0.0, 1.0

    def __len__(self): return len(self.v1)

    def __getitem__(self, idx):
        v1 = (self.v1[idx] - self.mean) / self.std if self.normalize else self.v1[idx]
        v2 = (self.v2[idx] - self.mean) / self.std if self.normalize else self.v2[idx]
        c1 = self.c1[idx]
        c2 = self.c2[idx]
        m  = self.mask[idx]
        a  = self.alpha[idx]
        x  = torch.tensor(np.stack([v1, v2, c1, c2], axis=0), dtype=torch.float32)  # [4,H,W]
        m  = torch.tensor(m, dtype=torch.float32).unsqueeze(0)
        a  = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
        return x, m, a

# ================== training ==================
def charbonnier(x, eps=1e-3):
    return torch.sqrt(x**2 + eps**2)

def train_model_symmetric_coh(model, loader, epochs=30, lr=1e-3, device="cuda",
                              lambda_smooth=0.02, beta_alpha=0.5):
    opt = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        running = 0.0
        for x, m, a_d in loader:
            x, m, a_d = x.to(device), m.to(device), a_d.to(device)
            v1, v2, c1, c2 = x[:,0:1], x[:,1:2], torch.clamp(x[:,2:3],0,1), torch.clamp(x[:,3:4],0,1)

            # coherence-weight for loss (overlap only)
            w = torch.sqrt(c1) + torch.sqrt(c2)
            w = w * m

            # alpha: distance + coherence
            a_r = c2 / (c1 + c2 + 1e-6)
            a = torch.clamp(beta_alpha * a_d + (1 - beta_alpha) * a_r, 0.1, 0.9)

            # predict correction
            c = model(x)
            v1c = v1 + a * c
            v2c = v2 - (1.0 - a) * c

            # overlap consistency (robust)
            diff = (v2c - v1c)
            loss_cons = (w * charbonnier(diff)).sum() / (w.sum() + 1e-6)

            # global smoothness (TV)
            grad_y = c[:,:,1:,:] - c[:,:,:-1,:]
            grad_x = c[:,:,:,1:] - c[:,:,:,:-1]
            loss_smooth = grad_x.abs().mean() + grad_y.abs().mean()

            loss = loss_cons + lambda_smooth * loss_smooth
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * x.size(0)
        print(f"Epoch {ep+1}/{epochs}  Loss: {running/len(loader.dataset):.6f}")
    return model

# ================== sliding inference ==================
def sliding_predict_c(model, arr1, arr2, coh1, coh2, tile=256, overlap=64, device="cuda", mean=0.0, std=1.0):
    H, W = arr1.shape
    out = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)
    step = tile - overlap
    ys = list(range(0, max(H - tile, 0) + 1, step)) or [0]
    xs = list(range(0, max(W - tile, 0) + 1, step)) or [0]
    model.eval()
    with torch.no_grad():
        for y in ys:
            for x in xs:
                hh, ww = min(tile, H - y), min(tile, W - x)
                p1 = arr1[y:y+hh, x:x+ww].astype(np.float32)
                p2 = arr2[y:y+hh, x:x+ww].astype(np.float32)
                q1 = coh1[y:y+hh, x:x+ww].astype(np.float32)
                q2 = coh2[y:y+hh, x:x+ww].astype(np.float32)
                pad = lambda: np.zeros((tile, tile), dtype=np.float32)
                pad1, pad2, padc1, padc2 = pad(), pad(), pad(), pad()
                pad1[:hh,:ww] = (p1 - mean) / std
                pad2[:hh,:ww] = (p2 - mean) / std
                padc1[:hh,:ww] = q1
                padc2[:hh,:ww] = q2
                inp = torch.tensor(np.stack([pad1, pad2, padc1, padc2], axis=0)).unsqueeze(0).to(device)
                pred = model(inp).squeeze().cpu().numpy()[:hh,:ww]
                out[y:y+hh, x:x+ww] += pred
                weight[y:y+hh, x:x+ww] += 1.0
    weight[weight == 0] = 1.0
    return out / weight

# ================== blending ==================
def blend_overlap_distance(arr1, arr2c, mask1, mask2, nodata_val=-9999.0):
    H, W = arr1.shape
    out = np.full((H, W), nodata_val, dtype=np.float32)
    alpha_map, overlap = distance_weight_alpha(mask1, mask2)
    out[mask1 & ~mask2] = arr1[mask1 & ~mask2]
    out[mask2 & ~mask1] = arr2c[mask2 & ~mask1]
    out[overlap] = alpha_map[overlap] * arr1[overlap] + (1.0 - alpha_map[overlap]) * arr2c[overlap]
    return out

# ================== main ==================
tif1 = r"H:\19_113_121\mintpy\geo\vel_19_113_121.tif"
tif2 = r"H:\19_113_116\mintpy\geo\vel_19_113_116.tif"
coh1_path = r"H:\19_113_121\coh\19_113_121.tif"
coh2_path = r"H:\19_113_116\coh\19_113_116.tif"

out1_corrected = r"H:\vel1_corrected_union.tif"
out2_corrected = r"H:\vel2_corrected_union.tif"
out_mosaic     = r"H:\mosaic_vel_union.tif"

# 1. union grid
with rasterio.open(tif1) as s1:
    crs1 = s1.crs
    transform1 = s1.transform
    bounds1 = s1.bounds
    profile1 = s1.profile
union_transform, union_height, union_width = union_grid(profile1, bounds1, crs1, tif2)
print("并集网格大小:", union_height, union_width)

# 2. reproject velocities & coherence to union
arr1 = reproject_to_union(tif1, crs1, union_transform, (union_height, union_width))
arr2 = reproject_to_union(tif2, crs1, union_transform, (union_height, union_width))
coh1 = reproject_to_union(coh1_path, crs1, union_transform, (union_height, union_width), resampling=RioResampling.nearest)
coh2 = reproject_to_union(coh2_path, crs1, union_transform, (union_height, union_width), resampling=RioResampling.nearest)

# sanitize coherence to [0,1]
coh1 = np.clip(coh1.filled(np.nan), 0.0, 1.0)
coh2 = np.clip(coh2.filled(np.nan), 0.0, 1.0)

mask1, mask2, overlap_mask = make_overlap_mask(arr1, arr2)
alpha_map, overlap = adaptive_alpha(mask1, mask2, coh1, coh2, beta_alpha=0.7)

# 3. patches from overlap
patch_size, stride = 16, 8
v1_p, v2_p, c1_p, c2_p, m_p, a_p = extract_patches(arr1.filled(np.nan), arr2.filled(np.nan),
                                                   coh1, coh2, overlap_mask, alpha_map,
                                                   patch_size, stride)
dataset = InSARDataset(v1_p, v2_p, c1_p, c2_p, m_p, a_p, normalize=True)
loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=False)
print(f"训练样本数: {len(dataset)}")

# 4. train
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(in_channels=4, out_channels=1, features=(32, 64, 128)).to(device)
model = train_model_symmetric_coh(model, loader, epochs=20, lr=1e-3, device=device,
                                  lambda_smooth=0.02, beta_alpha=0.5)

# 5. inference correction
mean, std = dataset.mean, dataset.std
arr1_filled = arr1.filled(np.nan).astype(np.float32)
arr2_filled = arr2.filled(np.nan).astype(np.float32)
arr1_filled[~np.isfinite(arr1_filled)] = 0.0
arr2_filled[~np.isfinite(arr2_filled)] = 0.0

c_pred = sliding_predict_c(model, arr1_filled, arr2_filled, coh1, coh2,
                           tile=256, overlap=16, device=device, mean=mean, std=std)

# ================== 6. 应用修正（带三层保护） ==================
arr1_corrected = arr1_filled.copy()
arr2_corrected = arr2_filled.copy()

# 修正只在 overlap 内应用
arr1_corrected[overlap] = arr1_corrected[overlap] + alpha_map[overlap] * c_pred[overlap]
arr2_corrected[overlap] = arr2_corrected[overlap] - (1.0 - alpha_map[overlap]) * c_pred[overlap]

# 保护一：绝对不要再清 NaN —— 删除下面两行（如果它们还在你的代码里，请一定删除）
# arr1_corrected[~mask1] = np.nan
# arr2_corrected[~mask2] = np.nan

# 保护二：校正后在 overlap 内若仍出现 NaN，则回退到原始填充值
bad1 = overlap & ~np.isfinite(arr1_corrected)
bad2 = overlap & ~np.isfinite(arr2_corrected)
arr1_corrected[bad1] = arr1_filled[bad1]
arr2_corrected[bad2] = arr2_filled[bad2]

# 保护三：非有效区回退原值，避免 NaN 进入融合
arr1_corrected = np.where(mask1, arr1_corrected, arr1_filled)
arr2_corrected = np.where(mask2, arr2_corrected, arr2_filled)

# ================== 7. 融合（只用各自的有效掩膜） ==================
def blend_overlap_distance_safe(arr1, arr2, mask1, mask2, nodata_val=-9999.0):
    H, W = arr1.shape
    out = np.full((H, W), nodata_val, dtype=np.float32)
    alpha_map, _ = distance_weight_alpha(mask1, mask2)

    only1 = mask1 & ~mask2
    only2 = mask2 & ~mask1
    both  = mask1 & mask2

    # 单边有效
    out[only1] = arr1[only1]
    out[only2] = arr2[only2]

    # 双边有效（再做一次 NaN 抑制以防边界漏网）
    v1b = np.where(np.isfinite(arr1), arr1, 0.0)
    v2b = np.where(np.isfinite(arr2), arr2, 0.0)
    out[both] = alpha_map[both] * v1b[both] + (1.0 - alpha_map[both]) * v2b[both]

    return out

mosaic = blend_overlap_distance_safe(arr1_corrected, arr2_corrected, mask1, mask2)


nodata_out = -9999.0
mosaic_out = np.where(np.isfinite(mosaic), mosaic, nodata_out).astype(np.float32)
arr1c_out  = np.where(np.isfinite(arr1_corrected), arr1_corrected, nodata_out).astype(np.float32)
arr2c_out  = np.where(np.isfinite(arr2_corrected), arr2_corrected, nodata_out).astype(np.float32)

# 8. write
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




# 1. 掩膜统计
print("mask1 有效像元:", mask1.sum())
print("mask2 有效像元:", mask2.sum())
print("overlap 有效像元:", overlap.sum())

# 2. overlap 区域 NaN 比例
nan_ratio1 = np.isnan(arr1.filled(np.nan)[overlap]).mean()
nan_ratio2 = np.isnan(arr2.filled(np.nan)[overlap]).mean()
print(f"arr1 overlap NaN比例: {nan_ratio1:.3f}")
print(f"arr2 overlap NaN比例: {nan_ratio2:.3f}")

# 3. 残差预测分布
print("c_pred 均值:", np.nanmean(c_pred))
print("c_pred 标准差:", np.nanstd(c_pred))
print("c_pred 在 overlap 内为零的比例:", (c_pred[overlap]==0).mean())

# 4. 校正后检查
nan_corr1 = np.isnan(arr1_corrected[overlap]).mean()
nan_corr2 = np.isnan(arr2_corrected[overlap]).mean()
print(f"arr1_corrected overlap NaN比例: {nan_corr1:.3f}")
print(f"arr2_corrected overlap NaN比例: {nan_corr2:.3f}")

# 5. 融合结果有效比例
valid_ratio = np.isfinite(mosaic).mean()
print(f"mosaic 有效像元比例: {valid_ratio:.3f}")
