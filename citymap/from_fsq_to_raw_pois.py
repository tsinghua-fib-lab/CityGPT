# ATTENTION: 从data2读取数据
import os
import pickle

import pandas as pd
from tqdm import tqdm

from map_config import all_bbox, FSQ_PATH, FSQ_PKL_PATH

if not os.path.exists(FSQ_PKL_PATH):
    os.makedirs(FSQ_PKL_PATH, exist_ok=True)

REQUIRED_COLUMNS = ["latitude", "longitude", "fsq_category_labels", "name"]


for ii, (city, bbox) in (enumerate(all_bbox.items())):
    print(f"{city} ({ii+1}/{len(all_bbox)})")
    total_count = 0
    lat_max, lat_min = bbox["max_lat"], bbox["min_lat"]
    lon_max, lon_min = bbox["max_lon"], bbox["min_lon"]

    # 遍历文件夹中的所有 Parquet 文件
    try:
        parquet_files = [f for f in os.listdir(FSQ_PATH) if f.endswith(".parquet")]

        if not parquet_files:
            print(f"文件夹 {FSQ_PATH} 中没有找到任何 parquet 文件。")
        else:
            print(
                f"在文件夹 {FSQ_PATH} 中找到 {len(parquet_files)} 个 parquet 文件，开始统计..."
            )

            target_df = pd.DataFrame()
            # 遍历每个 parquet 文件，统计总计
            for parquet_file in tqdm(parquet_files):
                file_path = os.path.join(FSQ_PATH, parquet_file)
                try:
                    df = pd.read_parquet(file_path)

                    missing_columns = [
                        col for col in REQUIRED_COLUMNS if col not in df.columns
                    ]
                    if missing_columns:
                        print(
                            f"文件 {parquet_file} 中缺少以下列，跳过文件处理：{', '.join(missing_columns)}"
                        )
                        continue

                    # 筛选经纬度在指定范围内的数据
                    in_range_df = df[
                        (df["latitude"] >= lat_min)
                        & (df["latitude"] <= lat_max)
                        & (df["longitude"] >= lon_min)
                        & (df["longitude"] <= lon_max)
                    ]
                    count_in_range = in_range_df.shape[0]
                    target_df = pd.concat(
                        [target_df, in_range_df], axis=0, ignore_index=True
                    )

                    # 累加统计结果
                    total_count += count_in_range

                    if "name" not in df.columns:
                        print(
                            f"文件 {parquet_file} 中不存在 'name' 列，跳过 name 统计..."
                        )
                    else:
                        # 累加 name 为空的条数
                        pass

                except Exception as e:
                    print(f"读取文件 {parquet_file} 时发生错误：{e}")

            # 输出总计结果
            print(
                f"文件夹 {FSQ_PATH} 下所有 parquet 文件中，经纬度在指定范围内的总条数：{total_count}"
            )
            # 输出
            # ATTENTION:格式 list["latitude", "longitude", "fsq_category_labels", "name"]
            output_list = target_df[REQUIRED_COLUMNS].values.tolist()
            pickle.dump(
                output_list,
                open(os.path.join(FSQ_PKL_PATH,f"{city}_raw_pois.pkl"), "wb"),
            )
    except FileNotFoundError:
        print(f"文件夹 {FSQ_PATH} 不存在，请检查路径。")
    except Exception as e:
        print(f"发生错误：{e}")
