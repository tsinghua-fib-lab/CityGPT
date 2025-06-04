from modelscope.msdatasets import MsDataset
import json
import jsonlines
import random
import argparse

from templates import geoeta_addr_element_text, geoeag_addr_match_text

DATA_VERSION = "v9"

def construct_dialogues_geoeta(data, output_file, max_sample, task_counts):
    # GeoGLUE数据集GeoETA的数据构造
    tag_full_names = {
        'roadno': 'road_number', 
        'cellno': 'cell_number', 
        'floorno': 'floor_number',
        'prov': 'province',
        'houseno': 'house_number'
    }
    with jsonlines.open(output_file, mode='w') as writer:
        for index, item in enumerate(data):
            if index >= max_sample:  # 检查是否达到处理数据的限制
                break  
            tokens = item['tokens']
            tags = item['ner_tags']
            full_address = ''.join(tokens).replace('_', ' ')

            address_components = {}
            ordered_elements = []  
            current_tag = None
            buffer = []
            
            for token, tag in zip(tokens, tags):
                base_tag = tag[2:] if '-' in tag else tag  
                tag_type = tag_full_names.get(base_tag, base_tag) 
                if 'B-' in tag or 'I-' in tag:
                    if current_tag and tag.startswith('B-'):
                        if current_tag not in address_components:
                            ordered_elements.append(current_tag)
                        address_components[current_tag] = ''.join(buffer)
                        buffer = [token]
                        current_tag = tag_type
                    else:
                        buffer.append(token)
                        current_tag = tag_type
                elif 'E-' in tag:
                    buffer.append(token)
                    if current_tag not in address_components:
                        ordered_elements.append(current_tag)
                    address_components[current_tag] = ''.join(buffer)
                    current_tag = None
                    buffer = []
                elif 'O' in tag:
                    if buffer and current_tag:
                        if current_tag not in address_components:
                            ordered_elements.append(current_tag)
                        address_components[current_tag] = ''.join(buffer)
                        buffer = []

            elements_requested = ", ".join(ordered_elements)
            assistant_answer = json.dumps({key: address_components[key] for key in ordered_elements}, ensure_ascii=False)
            
            geoeta_addr_element_session = [
                {"role": "user", "content": geoeta_addr_element_text(elements_requested, full_address)},
                {"role": "assistant", "content": assistant_answer}
            ]
            
            dialogues = {
                "task": "GeoGLUE",
                "id": f"GeoETA-addr_element-{index}",
                "diag": geoeta_addr_element_session
            }
            
            writer.write(dialogues)
    task_counts['GeoETA'] = index


def construct_dialogues_geoeag(data, output_file, task_counts):
    # GeoGLUE数据集GeoEAG的数据构造
    with jsonlines.open(output_file, mode='a') as writer:
        count = 0  
        for index, data_item in enumerate(data):
            sentence1 = data_item["sentence1"]
            sentence2 = data_item["sentence2"]
            label = data_item["label"]
            # 筛选北京范围内exact_match的数据
            if ("北京" in sentence1 or "北京" in sentence2) and label == "exact_match":
                geoeag_addr_match_session = [
                    {"role": "user", "content": geoeag_addr_match_text(sentence1)},
                    {"role": "assistant", "content": sentence2}
                ]

                dialogue_info = {
                    "task": "GeoGLUE",
                    "id": f"GeoEAG-addr_match-{count}",
                    "diag": geoeag_addr_match_session
                }
                
                writer.write(dialogue_info)
                count += 1  
                
        task_counts['GeoEAG'] = count
        


def main(args):
    random.seed(42)
    task_counts = {}
    
    # GeoETA数据集的对话构造
    # GeoETA数据集采样个数
    max_sample = 1000
    train_datasets_geoeta = MsDataset.load('GeoGLUE', namespace='damo', subset_name='GeoETA', split='train')
    train_data_geoeta = train_datasets_geoeta['train']
    construct_dialogues_geoeta(train_data_geoeta,args.output_file, max_sample, task_counts)

    # GeoEAG数据集的对话构造
    train_datasets_geoeag = MsDataset.load('GeoGLUE', namespace='damo', subset_name='GeoEAG', split='train')
    train_data_geoeag = train_datasets_geoeag['train']
    construct_dialogues_geoeag(train_data_geoeag, args.output_file, task_counts)
    
    # 输出任务统计结果
    print("任务类型统计结果：")
    for task_type, count in task_counts.items():
        print(f"{task_type}: {count}")


if __name__ == "__main__":


    # v1.1- GeoGLUE 增加了GeoGLUE中的GeoETA数据集，针对门址地址的元素分解任务
    # v1.2- GeoGLUE 增加了GeoGLUE中的GeoEAG数据集，针对exact_match匹配任务
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", default="./examples/spatial-v9.jsonl")
    parser.add_argument("--data_version", type=str, default=DATA_VERSION)
    args = parser.parse_args()
    main(args)
