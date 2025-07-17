import numpy as np
from osgeo import gdal

# 读取TIF图像
def read_tif(file_path):
    dataset = gdal.Open(file_path)
    band = dataset.GetRasterBand(1)
    return band.ReadAsArray()

# 筛选0-1之间的数值
def filter_values(images):
    return np.where((images >= 0) & (images <= 1), images, np.nan)

# 计算变异系数
def calculate_cv(images):
    mean = np.nanmean(images, axis=0)
    std_dev = np.nanstd(images, axis=0)
    cv = std_dev / mean
    return cv

# 归一化变异系数
def normalize_cv(cv):
    cv_min = np.nanmin(cv)
    cv_max = np.nanmax(cv)
    normalized_cv = (cv - cv_min) / (cv_max - cv_min)
    return normalized_cv

# 保存结果为TIF图像
def save_tif(data, output_path, reference_tif):
    driver = gdal.GetDriverByName('GTiff')
    dataset = gdal.Open(reference_tif)
    out_dataset = driver.Create(output_path, dataset.RasterXSize, dataset.RasterYSize, 1, gdal.GDT_Float32)
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(data)
    out_dataset.SetGeoTransform(dataset.GetGeoTransform())
    out_dataset.SetProjection(dataset.GetProjection())
    out_band.FlushCache()

# 主程序
# tif_files = [r"C:\Users\Kevin\Desktop\1\PVA_2015.tif", r"C:\Users\Kevin\Desktop\1\PVA_2017.tif", r"C:\Users\Kevin\Desktop\1\PVA_2019.tif", r"C:\Users\Kevin\Desktop\1\PVA_2021.tif"]
tif_files = [r"F:\new\analysis\tu\PVA_2011.tif",r"F:\new\analysis\tu\PVA_2013.tif", r"F:\new\analysis\tu\PVA_2015.tif",r"F:\new\analysis\tu\PVA_2017.tif",r"F:\new\analysis\tu\PVA_2019.tif",r"F:\new\analysis\tu\PVA_2021.tif",r"F:\new\analysis\tu\PVA_2023.tif"]
images = [read_tif(file) for file in tif_files]
filtered_images = filter_values(np.array(images))
cv = calculate_cv(filtered_images)
normalized_cv = normalize_cv(cv)
save_tif(normalized_cv, 'normalized_cv_1123.tif', tif_files[0])


