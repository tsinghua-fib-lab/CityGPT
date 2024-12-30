import json
import os
import random
import argparse
import jsonlines
import pandas as pd

from .predict_utils import *
random.seed(42)
MAX_VALID = 5
TEMP_INPUT_FILE = "evaluate/agent/prediction/input_diags.jsonl"
INPUT_TRAJ_FILE = 'evaluate/agent/prediction/trajectory_modified.csv'


def generate_inputs():
    candidate_type='all_pois'
    # 减少测试用例，每个用户最多只测试max_valid次
    max_valid = MAX_VALID

    data = pd.read_csv(INPUT_TRAJ_FILE)

    user_traj_id_mapping = data.groupby("user_id").agg(traj_list=("trajectory_id", set)).reset_index().set_index("user_id")["traj_list"].to_dict()
    for u in user_traj_id_mapping:
        cur_list = list(user_traj_id_mapping[u])
        sample_list = random.sample(cur_list, min(max_valid, len(cur_list)))
        user_traj_id_mapping[u] = sample_list
    
    df_grouped = data.groupby('user_id')['POI_name'].apply(set)
    choice_list = ['A', 'B', 'C', 'D','E','F','G','H','I','J','K','L','M','N']

    # 将groupby结果转化为字典
    user_pois = df_grouped.to_dict()

    # 增加全局地点字典作为候选，进一步提升问题难度，提升对环境信息的熟悉程度的重要性
    all_pois = data.groupby("POI_name").agg(traj_count=("trajectory_id", "count")).reset_index().set_index("POI_name")["traj_count"].to_dict()
    all_pois_set = set(all_pois.keys())

    data_by_traj_id = {}
    input_data_list = []
    for traj_id, group_df in data.groupby('trajectory_id'):
        trajectory_data = []

        for index, row in group_df.iterrows():
            # print(row)
            traject, user_id, start_time, end_time, poi_name, poi_cat_id, poi_cat_name, poi_id, longitude, latitude, intent, norm = row
            time_half_hour = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
            time_info = round_to_half_hour(time_half_hour).strftime('%Y-%m-%d %H:%M')
            day_type = check_weekend(start_time)
            # trajectory_points = [time_info, poi_name, poi_cat_name, intent, day_type]
            trajectory_points = [time_info, poi_name]
            trajectory_data.append(trajectory_points)

        if traj_id not in user_traj_id_mapping[user_id]:
            continue

        data_by_traj_id[traj_id] = trajectory_data[:-1]
        target = trajectory_data[-1]
        target_stay_time = target[0]
        if candidate_type == "self_history":
            candidate = user_pois.get(int(traj_id.split('_')[0]))
        elif candidate_type == "all_pois":
            candidate = all_pois_set

        cur_candi = random.sample(candidate, 5)
        if target[1] not in cur_candi:
            cur_candi.append(target[1])
        random.shuffle(cur_candi)

        my_dict = {choice_list[i]: item for i, item in enumerate(cur_candi)}
        my_dict_str = "\n".join([choice_list[i]+"."+item for i, item in enumerate(cur_candi)])        

        diags = [dict(role="system", content=get_prompts_new("system"))]
        examples = few_shot_exmaples()
        diags.extend(examples)
        diags.append(dict(role="user", content=get_prompts_new("user").format(data_by_traj_id[traj_id], target_stay_time, my_dict_str, choice_list)))
        input_data_list.append({"diags": diags, "target": target, "my_dict": my_dict, "choice_list": choice_list, "history": data_by_traj_id[traj_id], "candidate": cur_candi})

    with jsonlines.open(TEMP_INPUT_FILE, "w") as wid:
        wid.write_all(input_data_list)


def generate_answers(model_name, samples=None):
    save_model_name = model_name.replace("/", "-")

    input_data_list = []
    with jsonlines.open(TEMP_INPUT_FILE, "r") as fid:
        for obj in fid:
            input_data_list.append(obj)
    if samples == None:
        samples = len(input_data_list)
    correct_predictions = 0
    correct_predictions2 = 0
    # 总预测次数，即总样本数
    total_predictions = 0
    
    his_data = []
    try:
        with jsonlines.open('evaluate/agent/prediction/res/results_{}.json'.format(save_model_name), "r") as fid:
            for obj in fid:
                his_data.append(obj)
    except:
        print("没有历史数据积累")
        output_dir = 'evaluate/agent/prediction/res/'
        os.makedirs(output_dir, exist_ok=True)
    
    for i, data in enumerate(input_data_list):
        if i>samples:
            continue

        diags = data["diags"]
        target = data["target"]
        my_dict = data["my_dict"]
        choice_list = data["choice_list"]
        history = data["history"]
        candidate = data["candidate"]

        if i<len(his_data):
            print("该样例已测试，跳过")
            continue
        diags[1]["content"]=diags[1]["content"]+"Note that the user had 1 hour to move from the last POI in history to the predicted one. Directly predict the next POI from candidates."
        answer01 = get_chat_completion(session=diags, model_name=model_name, temperature=0, max_tokens=200)
        printQA(diags, answer01)
        
        pre_poi_name2 = process_action("ACTION:"+answer01, list(my_dict.values()))

        pre_poi = extract_choice(answer01, choice_list)
        pre_poi_name = my_dict[pre_poi]
        ground_truth_poi = target[1]
        total_predictions += 1
        print("ground truth:{} predict options:{} predict name from options:{} prediction name directly:{}".format(target[1], pre_poi, pre_poi_name, pre_poi_name2))
        # 如果预测正确，增加正确预测的数量
        # print(pre_poi_name == ground_truth_poi)
        if pre_poi_name == ground_truth_poi:
            correct_predictions += 1
        
        if pre_poi_name2 == ground_truth_poi:
            correct_predictions2 += 1

        result = {
            'history': history,
            'target': target,
            'candidate': candidate,
            'my_dict': my_dict,
            'ground_truth': ground_truth_poi,
            'prediction_number': pre_poi,
            'prediction_name': pre_poi_name,
            'prediction_name2': pre_poi_name2,
            "raw_answer": answer01
        }

        with open('evaluate/agent/prediction/res/results_{}.json'.format(save_model_name), 'a', encoding='utf-8') as json_file:
            json_str = json.dumps(result, ensure_ascii=False)
            json_file.write(json_str + "\n")  # 在这里添加了换行符

    # 计算预测准确率
    print('Prediction accuracy:', correct_predictions/total_predictions, correct_predictions2/total_predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="chatglm3-6B-v21.4:23131")
    parser.add_argument("--mode", type=str, default="gen_answer", choices=["gen_input", "gen_answer"])
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()

    if args.mode == "gen_input":
        generate_inputs()
    elif args.mode == "gen_answer":
        generate_answers(args.model, samples=args.samples)
