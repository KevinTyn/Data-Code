import numpy as np
from osgeo import gdal, osr
import matplotlib.pyplot as plt

def read_tif(file_path):
    try:
        dataset = gdal.Open(file_path)
        if dataset is None:
            raise FileNotFoundError(f"File {file_path} not found.")
        band = dataset.GetRasterBand(1)
        array = band.ReadAsArray()
        return array, dataset
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

def reshape_image(image):
    if image is not None and image.ndim == 1:
        size = int(np.sqrt(image.size))
        return image.reshape((size, size))
    return image

def filter_data(image):
    return np.where((image >= 0) & (image <= 1), image, 0)

def calculate_cva(image1, image2):
    diff = image2 - image1
    magnitude = np.sqrt(diff**2)
    direction = np.arctan2(diff, diff)
    return magnitude, direction

def save_tif(output_path, array, reference_dataset):
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(output_path, reference_dataset.RasterXSize, reference_dataset.RasterYSize, 1, gdal.GDT_Byte)
    out_dataset.SetGeoTransform(reference_dataset.GetGeoTransform())
    out_dataset.SetProjection(reference_dataset.GetProjection())
    out_band = out_dataset.GetRasterBand(1)
    array_scaled = np.clip(array * 255, 0, 255).astype(np.uint8)
    out_band.WriteArray(array_scaled)
    out_band.FlushCache()
    out_dataset = None

def apply_threshold(magnitude, threshold):
    change_map = np.where(magnitude > threshold, 1, 0)
    return change_map

def calculate_threshold(magnitude):
    valid_values = magnitude[(magnitude >= 0) & (magnitude <= 1)]
    threshold = np.mean(valid_values) + np.std(valid_values)
    return threshold

# 读取四年的图像（假设数据已经归一化）
image1, dataset1 = read_tif(r"F:\new\analysis\tu\PVA_2011.tif")
image2, dataset2 = read_tif(r"F:\new\analysis\tu\PVA_2013.tif")
image3, dataset3 = read_tif(r"F:\new\analysis\tu\PVA_2015.tif")
image4, dataset4 = read_tif(r"F:\new\analysis\tu\PVA_2017.tif")
image5, dataset5 = read_tif(r"F:\new\analysis\tu\PVA_2019.tif")
image6, dataset6 = read_tif(r"F:\new\analysis\tu\PVA_2021.tif")
image7, dataset7 = read_tif(r"F:\new\analysis\tu\PVA_2023.tif")

# 确保数据是二维的
image1 = reshape_image(image1)
image2 = reshape_image(image2)
image3 = reshape_image(image3)
image4 = reshape_image(image4)
image5 = reshape_image(image5)
image6 = reshape_image(image6)
image7 = reshape_image(image7)

# 确保图像不为空
if all(image is not None for image in (image1, image2, image3, image4, image5, image6, image7)):
    # 筛选数据，只保留在0到1之间的值
    image1 = filter_data(image1)
    image2 = filter_data(image2)
    image3 = filter_data(image3)
    image4 = filter_data(image4)
    image5 = filter_data(image5)
    image6 = filter_data(image6)
    image7 = filter_data(image7)

    # 计算变化向量
    magnitude12, direction12 = calculate_cva(image1, image2)
    magnitude23, direction23 = calculate_cva(image2, image3)
    magnitude34, direction34 = calculate_cva(image3, image4)
    magnitude45, direction45 = calculate_cva(image4, image5)
    magnitude56, direction56 = calculate_cva(image5, image6)
    magnitude67, direction67 = calculate_cva(image6, image7)

    # 设置阈值（只计算0到1之间的数值）
    threshold12 = calculate_threshold(magnitude12)
    threshold23 = calculate_threshold(magnitude23)
    threshold34 = calculate_threshold(magnitude34)
    threshold45 = calculate_threshold(magnitude45)
    threshold56 = calculate_threshold(magnitude56)
    threshold67 = calculate_threshold(magnitude67)


    # 应用阈值，生成变化区域图
    change_map12 = apply_threshold(magnitude12, threshold12)
    change_map23 = apply_threshold(magnitude23, threshold23)
    change_map34 = apply_threshold(magnitude34, threshold34)
    change_map45 = apply_threshold(magnitude45, threshold45)
    change_map56 = apply_threshold(magnitude56, threshold56)
    change_map67 = apply_threshold(magnitude67, threshold67)

    # # 保存变化图像为 TIF 文件
    # save_tif(r"E:\矿大\呼包鄂榆\数据集\1\analysis\CVA\magnitude12.tif", magnitude12, dataset1)
    # save_tif(r"E:\矿大\呼包鄂榆\数据集\1\analysis\CVA\magnitude23.tif", magnitude23, dataset2)
    # save_tif(r"E:\矿大\呼包鄂榆\数据集\1\analysis\CVA\magnitude34.tif", magnitude34, dataset3)

    # 保存变化区域图像为 TIF 文件
    save_tif(r"F:\new\analysis\cva\change_map12.tif", change_map12, dataset1)
    save_tif(r"F:\new\analysis\cva\change_map23.tif", change_map23, dataset2)
    save_tif(r"F:\new\analysis\cva\change_map34.tif", change_map34, dataset3)
    save_tif(r"F:\new\analysis\cva\change_map45.tif", change_map45, dataset4)
    save_tif(r"F:\new\analysis\cva\change_map56.tif", change_map56, dataset5)
    save_tif(r"F:\new\analysis\cva\change_map67.tif", change_map67, dataset6)

    # 可视化变化向量的强度和变化区域
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.title('Year 1 to Year 2 Magnitude')
    plt.imshow(magnitude12, cmap='hot')
    plt.colorbar()

    plt.subplot(2, 3, 2)
    plt.title('Year 2 to Year 3 Magnitude')
    plt.imshow(magnitude23, cmap='hot')
    plt.colorbar()

    plt.subplot(2, 3, 3)
    plt.title('Year 3 to Year 4 Magnitude')
    plt.imshow(magnitude34, cmap='hot')
    plt.colorbar()

    plt.subplot(2, 3, 4)
    plt.title('Year 1 to Year 2 Change Map')
    plt.imshow(change_map12, cmap='gray')
    plt.colorbar()

    plt.subplot(2, 3, 5)
    plt.title('Year 2 to Year 3 Change Map')
    plt.imshow(change_map23, cmap='gray')
    plt.colorbar()

    plt.subplot(2, 3, 6)
    plt.title('Year 3 to Year 4 Change Map')
    plt.imshow(change_map34, cmap='gray')
    plt.colorbar()

    plt.show()
else:
    print("One or more images could not be read.")
