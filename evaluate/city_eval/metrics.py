import os
import re
import pandas as pd
import numpy as np
import argparse

from config import EVAL_TASK_MAPPING_v1, EVAL_TASK_MAPPING_v2
def get_result(file_path, result_files, map_task, region):
    final_result = {}
    for result_file in result_files:
        match = re.match(r"city_eval_([^_]+)_([^_]+)_([^_]+)_", result_file)
        if not match:
            continue
        region_exp = match.group(1)
        if region_exp != region:
            continue
        version = match.group(2)
        model_name = match.group(3)
        result = pd.read_csv(os.path.join(file_path,result_file))
        # Check if the number of results is correct
        file_row_count = len(result)  
        # 20
        if version == "v1":
            if file_row_count != 20:
                print(f"Error: The number of results for model {model_name} under city {region_exp} is {file_row_count}, not 20.")
        elif version == "v2":
            if file_row_count != 39:
                print(f"Error: The number of results for model {model_name} under city {region_exp} is {file_row_count}, not 39.")
            
        for _, row in result.iterrows():
            task = row['task_name']
            acc = row['accuracy']
            if (model_name, region_exp) not in final_result:
                final_result[(model_name, region_exp)] = {
                    "city_image": [],
                    "urban_semantics": [],
                    "spatial_reasoning_route": [],
                    "spatial_reasoning_noroute": [],
                }
            for cat,tasks in map_task.items():
                if task in tasks:
                    final_result[(model_name, region_exp)][cat].append(acc)
                    # print("1st task:{}, 2nd task:{}".format(cat, task))
                    break
    
    data = []
    for (model_name, region_exp), categories in final_result.items():
        city_image_acc = round(sum(categories["city_image"]) / len(categories["city_image"]), 4) if categories["city_image"] else None
        urban_semantics_acc = round(sum(categories["urban_semantics"]) / len(categories["urban_semantics"]), 4) if categories["urban_semantics"] else None
        spatial_reasoning_route_acc = round(sum(categories["spatial_reasoning_route"]) / len(categories["spatial_reasoning_route"]), 4) if categories["spatial_reasoning_route"] else None
        spatial_reasoning_noroute_acc = round(sum(categories["spatial_reasoning_noroute"]) / len(categories["spatial_reasoning_noroute"]), 4) if categories["spatial_reasoning_noroute"] else None

        row = [model_name, region_exp, 
            city_image_acc,
            urban_semantics_acc,
            spatial_reasoning_route_acc,
            spatial_reasoning_noroute_acc]
        data.append(row)
    
    df = pd.DataFrame(data, columns=['Model_Name', 'Region', 'City_Image', 'Urban_Semantics', 'Spatial_Reasoning_Route', 'Spatial_Reasoning_NoRoute'])
    df['Spatial_Reasoning'] = df[['Spatial_Reasoning_Route', 'Spatial_Reasoning_NoRoute']].mean(axis=1).round(4)
    return df



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--region_exp", type=str, default="yuyuantan")
    parser.add_argument("--evaluate_version", type=str, default="v1")
    args = parser.parse_args()
    file_path = "evaluate/city_eval/results"
    result_files = os.listdir(file_path)
    
    # if args.region_exp == "wudaokou_small" or args.region_exp == "wudaokou_large" or args.region_exp == "yuyuantan" or args.region_exp == "dahongmen" or args.region_exp == "beijing" or args.region_exp == "wangjing" or args.region_exp == "Beijing":
    #     EVAL_TASK_MAPPING = EVAL_TASK_MAPPING_v2
    # else:
    #     if args.city_eval_version == "v1":
    #         EVAL_TASK_MAPPING = EVAL_TASK_MAPPING_v1
    #     elif args.city_eval_version == "v2":
    #         EVAL_TASK_MAPPING = EVAL_TASK_MAPPING_v2
    EVAL_TASK_MAPPING = EVAL_TASK_MAPPING_v2
    result = get_result(file_path, result_files, EVAL_TASK_MAPPING, args.region_exp)
    result.to_csv(os.path.join(file_path,"cityeval_benchmark_result.csv"), index=False)
    print("CityEval results have been saved!")

