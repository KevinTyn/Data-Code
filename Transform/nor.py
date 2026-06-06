import os
import numpy as np
import rasterio


def normalize_geotiffs(input_folder, output_folder):
    # 1. 自动创建输出的归一化文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已创建输出文件夹: {output_folder}")

    # 2. 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 仅处理 .tif 或 .tiff 结尾的文件
        if filename.lower().endswith(('.tif', '.tiff')):
            input_path = os.path.join(input_folder, filename)

            # 生成带 _nor 后缀的输出文件名
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_nor{ext}"
            output_path = os.path.join(output_folder, output_filename)

            print(f"正在处理: {filename} ...")

            # 3. 使用 rasterio 打开原始图像
            with rasterio.open(input_path) as src:
                # 获取图像的所有空间和元数据信息 (坐标系、仿射变换、大小等)
                profile = src.profile.copy()
                nodata_value = src.nodata

                # 读取所有波段的像素数据
                # 将数据类型转为 float32 以便进行浮点数的运算
                data = src.read().astype(np.float32)

                # 创建用于存储归一化结果的数组
                out_data = np.zeros_like(data, dtype=np.float32)

                # 获取波段数量
                num_bands = data.shape[0]

                # 4. 逐波段进行归一化 (Min-Max Normalization -> 0 到 1)
                for i in range(num_bands):
                    band = data[i]

                    # 如果图像有 NoData 值，需要将其掩码排除，防止它参与最大最小值计算
                    if nodata_value is not None:
                        valid_mask = (band != nodata_value) & (~np.isnan(band))
                    else:
                        valid_mask = ~np.isnan(band)

                    # 如果该波段有有效像素
                    if np.any(valid_mask):
                        b_min = np.min(band[valid_mask])
                        b_max = np.max(band[valid_mask])

                        # 避免除以 0 的情况（整张图颜色一样）
                        if b_max > b_min:
                            out_data[i][valid_mask] = (band[valid_mask] - b_min) / (b_max - b_min)
                        else:
                            out_data[i][valid_mask] = 0.0

                    # 处理原始的 NoData 区域（用 np.nan 填充）
                    out_data[i][~valid_mask] = np.nan

                # 5. 更新元数据
                # 因为归一化后变成了 0~1 的小数，所以必须把数据类型修改为 float32
                # 同时将新的 NoData 值更新为 nan
                profile.update(
                    dtype=rasterio.float32,
                    nodata=np.nan
                )

                # 6. 将处理后的数据连同原有的空间信息写入新文件
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(out_data)

            print(f"处理完成 -> {output_filename}")


if __name__ == "__main__":
    # 在这里修改为你的实际文件夹路径
    INPUT_DIR = r"D:\BaiduSyncdisk\课题首要\PV_selection\file\process\17_process"
    OUTPUT_DIR = r"D:\BaiduSyncdisk\课题首要\PV_selection\file\process\17_process\normalized_tiffs"

    normalize_geotiffs(INPUT_DIR, OUTPUT_DIR)
    print("所有文件归一化处理完毕！")