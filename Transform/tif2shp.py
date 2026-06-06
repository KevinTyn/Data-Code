import os
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union
import geopandas as gpd
from pathlib import Path
from collections import defaultdict


def get_exact_exterior(tif_path):
    """核心：提取单个TIF的最外层精确边界"""
    with rasterio.open(tif_path) as src:
        mask = src.dataset_mask()
        crs = src.crs
        results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) in enumerate(shapes(mask, mask=mask, transform=src.transform))
        )
        geoms = [shape(feature['geometry']) for feature in results]
        if not geoms:
            return None, None

        merged_geom = unary_union(geoms)

        # 内部函数：去除空洞
        def remove_interiors(poly):
            if poly.is_empty: return poly
            return Polygon(poly.exterior)

        if merged_geom.geom_type == 'Polygon':
            final_geom = remove_interiors(merged_geom)
        elif merged_geom.geom_type == 'MultiPolygon':
            final_geom = MultiPolygon([remove_interiors(p) for p in merged_geom.geoms])
        else:
            final_geom = merged_geom

        return final_geom, crs


def save_grouped_results(data_list, output_root, crs):
    """独立函数：按组号合并并导出总文件"""
    if not data_list:
        return

    group_path = Path(output_root) / "grouped_results"
    group_path.mkdir(parents=True, exist_ok=True)

    # 1. 提取组号并分类
    grouped_dict = defaultdict(list)
    for item in data_list:
        grouped_dict[item['group_id']].append(item)

    # 2. 导出每个组的合并文件
    for g_id, items in grouped_dict.items():
        gdf = gpd.GeoDataFrame(items, crs=crs)
        out_name = f"group_{g_id}.shp"
        gdf.to_file(group_path / out_name, driver='ESRI Shapefile', encoding='utf-8')
        print(f"  --> 组合并完成: {out_name}")

    # 3. 顺便导出一个包含所有影像的总文件
    total_gdf = gpd.GeoDataFrame(data_list, crs=crs)
    total_gdf.to_file(Path(output_root) / "total_all_in_one.shp", driver='ESRI Shapefile', encoding='utf-8')
    print(f"  --> 总汇总文件导出完成: total_all_in_one.shp")


# --- 主控制流程 ---
def run_main_task(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tif_files = list(input_path.rglob("vel_*.tif"))
    results_list = []
    last_crs = None

    print(f"开始处理，发现 {len(tif_files)} 个文件...")

    for tif_file in tif_files:
        try:
            # 1. 提取边界
            geom, crs = get_exact_exterior(tif_file)
            if geom is None: continue
            last_crs = crs

            # 2. 识别组ID (vel_17_113_111 -> 113)
            parts = tif_file.stem.split('_')
            group_id = parts[2] if len(parts) > 2 else "unknown"

            # 3. 记录数据
            res_item = {
                'file_name': tif_file.name,
                'group_id': group_id,
                'geometry': geom
            }
            results_list.append(res_item)

            # 4. 导出单个文件的SHP
            single_gdf = gpd.GeoDataFrame([res_item], crs=crs)
            single_gdf.to_file(output_path / f"{tif_file.stem}.shp")
            print(f"已处理: {tif_file.name}")

        except Exception as e:
            print(f"处理 {tif_file.name} 时出错: {e}")

    # ========================================================
    # 这里是【按组合并】的开关，不需要时注释掉下面这一行即可
    save_grouped_results(results_list, output_path, last_crs)
    # ========================================================


if __name__ == "__main__":
    my_input = r'E:\shanxi\17'
    my_output = r'E:\shanxi\17\1'
    run_main_task(my_input, my_output)