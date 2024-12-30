import os
import re
import pandas as pd
import numpy as np
import argparse

from config import EVAL_TASK_MAPPING_v1, EVAL_TASK_MAPPING_v2
def get_result(file_path, result_files, map_task):
    final_result = {}
    for result_file in result_files:
        match = re.match(r"city_eval_([^_]+)_([^_]+)_([^_]+)_", result_file)
        if not match:
            continue
        region_exp = match.group(1)
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
                print(f"Error: The number of results for model {model_name} under city {region_exp} is {file_row_count}, not 49.")
            
        for _, row in result.iterrows():
            task = row['task_name']
            acc = row['accuracy']
            if (model_name, region_exp) not in final_result:
                final_result[(model_name, region_exp)] = {
                    "city_image": None,
                    "urban_semantics": None,
                    "spatial_reasoning_route": None,
                    "spatial_reasoning_noroute": None,
                }
            for cat,tasks in map_task.items():
                if task in tasks:
                    final_result[(model_name, region_exp)][cat] = acc
                    break
    
    data = []
    for (model_name, region_exp), categories in final_result.items():
        row = [model_name, region_exp, categories['city_image'], categories['urban_semantics'], categories['spatial_reasoning_route'], categories['spatial_reasoning_noroute']]
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
    
    if args.region_exp == "wudaokou_small" or args.region_exp == "wudaokou_large" or args.region_exp == "yuyuantan" or args.region_exp == "dahongmen" or args.region_exp == "beijing" or args.region_exp == "wangjing":
        EVAL_TASK_MAPPING = EVAL_TASK_MAPPING_v2
    else:
        if args.city_eval_version == "v1":
            EVAL_TASK_MAPPING = EVAL_TASK_MAPPING_v1
        elif args.city_eval_version == "v2":
            EVAL_TASK_MAPPING = EVAL_TASK_MAPPING_v2
    result = get_result(file_path, result_files, EVAL_TASK_MAPPING)
    result.to_csv(os.path.join(file_path,"cityeval_benchmark_result.csv"), index=False)
    print("CityEval results have been saved!")

