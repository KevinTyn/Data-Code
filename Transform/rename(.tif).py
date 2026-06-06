import os

folder = r"E:\fhly\Annual_NPP_500m(2000_2023)"  # 修改为你的文件夹路径

for filename in os.listdir(folder):
    if filename.endswith(".tif.tif"):
        new_name = filename[:-4]  # 去掉最后一个 .tif
        os.rename(os.path.join(folder, filename),
                  os.path.join(folder, new_name))
        print(f"重命名: {filename} -> {new_name}")
