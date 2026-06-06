import h5py
from osgeo import gdal, osr
import numpy as np

# 打开 HDF5 文件
file_path = r"D:\BaiduSyncdisk\Code\matlab\2023.h5"
with h5py.File(file_path, 'r') as file:
    # 选择一个组
    group_key = list(file.keys())[0]
    data = file[group_key][:]

    # 获取投影信息
    projection_info = 'PROJCS["AMSRE_Coordinate_System",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Cylindrical_Equal_Area"],PARAMETER["standard_parallel_1",30],PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'

    # 创建 GeoTIFF 文件
    driver = gdal.GetDriverByName('GTiff')
    output_file = 'output_file.tif'
    rows, cols = data.shape
    dataset = driver.Create(output_file, cols, rows, 1, gdal.GDT_Float32)

    # 设置投影和地理变换
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection_info)
    dataset.SetProjection(srs.ExportToWkt())
    geotransform = [7087254.893403063, 926.6258333333334, 0.0, 5896446.849020582, 0.0, -926.6254166666668]
    dataset.SetGeoTransform(geotransform)

    # 写入数据
    dataset.GetRasterBand(1).WriteArray(data)
    dataset.FlushCache()

    print("GeoTIFF 文件已成功创建。")
