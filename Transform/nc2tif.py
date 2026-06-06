import os
import xarray as xr
import rasterio
from rasterio.transform import from_origin
import numpy as np

input_folder = r"E:\fhly\Annual_Aridity_index_1km(1985_2024)"
tif_folder = os.path.join(input_folder, "tif")
os.makedirs(tif_folder, exist_ok=True)

def find_coord(ds, candidates):
    for name in candidates:
        if name in ds.coords:
            return name
        if name in ds:  # 某些写在变量里
            return name
    raise ValueError(f"找不到坐标：{candidates}")

for file in os.listdir(input_folder):
    if not file.endswith(".nc"):
        continue

    nc_path = os.path.join(input_folder, file)
    print(f"\n=== 处理: {nc_path} ===")

    # 打开 nc（尝试两个引擎）
    try:
        ds = xr.open_dataset(nc_path, engine="netcdf4")
    except Exception:
        ds = xr.open_dataset(nc_path, engine="h5netcdf")

    var_name = list(ds.data_vars)[0]
    data = ds[var_name]

    # 自动识别坐标名
    lon_name = find_coord(ds, ["lon", "longitude", "x"])
    lat_name = find_coord(ds, ["lat", "latitude", "y"])
    lon = ds[lon_name].values
    lat = ds[lat_name].values

    # 诊断信息
    print(f"- 变量: {var_name}")
    print(f"- dims: {data.dims}, shape: {data.shape}")
    print(f"- lon: first={lon[0]}, last={lon[-1]} (递增? {lon[0] < lon[-1]})")
    print(f"- lat: first={lat[0]}, last={lat[-1]} (递增? {lat[0] < lat[-1]})")

    # 确保 (lat, lon) 维度顺序
    if data.dims != (lat_name, lon_name):
        print(f"- 重排维度到({lat_name}, {lon_name})")
        data = data.transpose(lat_name, lon_name)

    band_data = data.values.astype("float32")

    # 分辨率（正数）
    xres = abs(lon[1] - lon[0])
    yres = abs(lat[1] - lat[0])

    # 计算上左角（upper-left）
    # - 上（北）用纬度的最大值
    # - 左（西）用经度的最小值
    ul_y = np.max(lat)
    ul_x = np.min(lon)

    # 根据经纬度方向翻转数据，使矩阵行列与上左角一致
    flipped_vertical = False
    flipped_horizontal = False

    # 行方向（北→南）：如果数组是南→北递增（lat[0] < lat[-1]），需要上下翻转
    if lat[0] < lat[-1]:
        band_data = band_data[::-1, :]
        flipped_vertical = True

    # 列方向（西→东）：如果数组是东→西递减（lon[0] > lon[-1]），需要左右翻转
    if lon[0] > lon[-1]:
        band_data = band_data[:, ::-1]
        flipped_horizontal = True

    print(f"- 翻转: vertical={flipped_vertical}, horizontal={flipped_horizontal}")
    print(f"- upper-left: x={ul_x}, y={ul_y}, xres={xres}, yres={yres}")

    # 使用正像元大小和上左角构造 transform
    transform = from_origin(ul_x, ul_y, xres, yres)

    # 输出
    nc_name = os.path.splitext(file)[0]
    tif_path = os.path.join(tif_folder, f"{nc_name}.tif")

    with rasterio.open(
        tif_path,
        'w',
        driver='GTiff',
        height=band_data.shape[0],
        width=band_data.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",  # 仅在确认为经纬度
        transform=transform,
    ) as dst:
        dst.write(band_data, 1)

    print(f"- 已保存: {tif_path}")
    ds.close()
