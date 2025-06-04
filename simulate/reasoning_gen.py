import json
import os
import random
import re
import time
import argparse
from shapely.geometry import Polygon

import numpy as np
import pandas as pd

from config import RESOURCE_PATH, REGION_EXP, STEP, MIN_ROAD_LENGTH, MAP_DICT, MAP_PORT_DICT, TRAIN_DATA_PATH, MAP_CACHE_PATH, ROUTING_PATH, DATA_VERSION, DIS2CORNER
from evaluate.city_eval.utils import dir_all_dis, compute_length, calcu_azimuth, angle2dir, angle2dir_4, secondary_dir_to_primary_dirs, primary_directions, secondary_directions, EW, NS, dir_map, dir_map2, get_landuse_dict, task_files_adaption
from simulate.player import category_mapping
from simulate.utils import load_map, compute_length_template
from simulate.address_system import get_aoi_address


#######POI2RD_DIS
def get_poi2rd_dis(map, diags, diag_routes, name_adr_poi):
    # shuffle the order of dialogs
    combined = list(zip(diags, diag_routes))
    random.shuffle(combined)
    diags, diag_routes = zip(*combined)
    diags = list(diags)
    diag_routes = list(diag_routes)

    poi2rd_dis = []
    index_dis = 0
    for cnt, diag in enumerate(diags):
        dialog = {'task': 'cityreasoning-{}'.format(REGION_EXP), 'id': 'poi2rd_dis' + str(index_dis), 'diag': []}
        content = json.loads(diag[1]["content"])
        start_poi_name = content['start_name']
        start_poi_addr = content['start_addr']
        start_poi = "{}{}".format(start_poi_name, start_poi_addr)
        end_poi_name = content['dest_name']
        end_poi_addr = content['dest_addr']
        end_poi = "{}{}".format(end_poi_name, end_poi_addr)
        routes = content['routes']
        dir_dis_fin = dir_all_dis(routes, secondary_directions, primary_directions,secondary_dir_to_primary_dirs)[0]
        segment = ','.join([str(item) for item in dir_dis_fin])
        st1 = "Step 1: Determine the direction of each segment of the journey in the form of '(The direction you are facing when walking along the road,distance)'. The distance moved in the secondary direction needs to be multiplied by 0.7 and decomposed into the primary direction.:\n" + segment + '\n'
        NSEW = {'NS': [], 'EW': []}
        for dir, dis in dir_dis_fin:
            if dir in NS:
                NSEW['NS'].append(dis)
            elif dir in EW:
                NSEW['EW'].append(dis)
            else:
                print(dir)
        NS_dis = []
        EW_dis = []
        for dir, dis in dir_dis_fin:
            if dir in NS:
                NS_dis.append((dir, dis))
            elif dir in EW:
                EW_dis.append((dir, dis))
        st2 = ("Step 2: Calculate distances you move on south-north and east-west.\
            Consider distances you move southward and northward,\
            you move a total of {}-{}={} meters south-north. Consider distances you move westward and eastward,\
            you move a total of {}-{}={} meters east-west.").format(
            max(NS_dis, key=lambda x: x[1])[1], min(NS_dis, key=lambda x: x[1])[1], np.abs(NS_dis[0][1] - NS_dis[1][1]),
            max(EW_dis, key=lambda x: x[1])[1], min(EW_dis, key=lambda x: x[1])[1], np.abs(EW_dis[0][1] - EW_dis[1][1])
        )
        if end_poi not in name_adr_poi and start_poi not in name_adr_poi:
            print("Both start and end POI are not in the map!")
            continue
        if start_poi in name_adr_poi:  # 使用起点POI所在道路，终点POI
            start_aoi_id = name_adr_poi[start_poi]['aoi_id']
            start_aoi_info = map.aois[start_aoi_id]
            result = get_aoi_address(map, start_aoi_info)
            if result is None:
                print("No address for start AOI!")
                continue
            aoi_s, rd_name, junc_name, aoi_junc_direction, rd_belong_dir = result
            routine = 'After starting from {}, you '.format(start_poi) + ", ".join(diag_routes[cnt]) + ". Finally you arrive at {}.{} is on {}.".format(end_poi, start_poi, rd_name)
            if rd_belong_dir in EW:
                distance = np.abs(NSEW['NS'][0] - NSEW['NS'][1])
            elif rd_belong_dir in NS:
                distance = np.abs(NSEW['EW'][0] - NSEW['EW'][1])
            if distance < DIS2CORNER:
                continue
            distances = [int(distance / 2), int(distance * 2), int(distance + 500), distance]
            random.shuffle(distances)
            letters = ['A', 'B', 'C', 'D']
            choices = ['{}.{}'.format(x, y) for x, y in zip(letters, distances)]
            st3 = "Step3: Find the direction of {}. From the address of POI on {}, which is {}, we can find the direction to road is {}, which means {} runs {}.".format(
                rd_name, rd_name, start_poi_addr, rd_belong_dir, rd_name, dir_map[rd_belong_dir])
            res_dis = dict(zip(["A", "B", "C", "D"], distances))
            for k in res_dis:
                if res_dis[k] == distance:
                    label = k
            st4 = "Step4: Answer:{}. Calculate the distance between {} and {}. If the road runs south-north, then the answer is the distance you move east-west in step 3.\
            If the road runs east-west, then the answer is the distance you move south-north in step 3. Because {} runs {},\
            so the final answer is the distance you move {} in step 3, which is {} meters.".format(label, end_poi, rd_name,
                                                                                                rd_name,
                                                                                                dir_map[rd_belong_dir],
                                                                                                dir_map2[dir_map[
                                                                                                    rd_belong_dir]],
                                                                                                distance)
            question = "What is the distance between {} and {}?".format(end_poi, rd_name)
            dialog['diag'].append(
                {'role': 'user', 'content': routine + question + '\n'.join(choices) + "\nLet's think step by step.\n"})
            dialog['diag'].append({'role': 'assistant', 'content': st1 + st2 + st3 + st4})
            poi2rd_dis.append(dialog)
            index_dis += 1
        elif end_poi in name_adr_poi:  # 使用终点POI所在道路，起点POI
            end_aoi_id = name_adr_poi[end_poi]['aoi_id']
            end_aoi_info = map.aois[end_aoi_id]
            result = get_aoi_address(map, end_aoi_info)
            if result is None:
                continue
            aoi_s, rd_name, junc_name, aoi_junc_direction, rd_belong_dir = result
            routine = 'After starting from {}, you '.format(start_poi) + ", ".join(diag_routes[cnt]) + ".Finally you arrive at {}.{} is on {}.".format(end_poi, end_poi, rd_name)
            if rd_belong_dir in EW:
                distance = np.abs(NSEW['NS'][0] - NSEW['NS'][1])
            elif rd_belong_dir in NS:
                distance = np.abs(NSEW['EW'][0] - NSEW['EW'][1])
            if distance < DIS2CORNER:
                continue
            distances = [str(int(distance / 2)), str(int(distance * 2)), str(int(distance + 500)), str(int(distance))]
            random.shuffle(distances)
            letters = ['A', 'B', 'C', 'D']
            choices = ['{}.{}'.format(x, y) for x, y in zip(letters, distances)]
            st3 = "Step3: Find the direction of {}. From the address of POI on {}, which is {}, we can find the direction to corner is {}, which means {} runs {}.".format(
                rd_name, rd_name, start_poi_addr, rd_belong_dir, rd_name, dir_map[rd_belong_dir])
            res_dis = dict(zip(["A", "B", "C", "D"], distances))
            res_dis_prompt = dict(zip(["A", "B", "C", "D"], distances))
            for k in res_dis:
                if res_dis[k] == distance:
                    label = k
            st4 = "Step4: Answer:{}.Calculate the distance between {} and {}. If the road runs south-north, then the answer is the distance you move east-west in step 3.".format(label, start_poi, rd_name) + "If the road runs east-west, then the answer is the distance you move south-north in step 3.Because {} runs {},".format(rd_name, dir_map[rd_belong_dir]) +" so the final answer is the distance you move {} in step 3, which is {} meters.".format(dir_map2[dir_map[rd_belong_dir]], distance)
            question = "What is the distance between {} and {}?".format(start_poi, rd_name)
            dialog['diag'].append(
                {'role': 'user', 'content': routine + question + '\n'.join(choices) + "\nLet's think step by step.\n"})
            dialog['diag'].append({'role': 'assistant', 'content': st1 + st2 + st3 + st4})
            poi2rd_dis.append(dialog)
            index_dis += 1
        if index_dis == 1000:
            break
    return poi2rd_dis


#######AOI2RD_DIS,起终点有一个有AOI名字即可。起点有AOI名字，则使用终点所在道路，反之亦然
def get_aoi2rd_dis(map, diags, diag_routes, aoi_dict, name_adr_poi):
    # shuffle the order of dialogs
    combined = list(zip(diags, diag_routes))
    random.shuffle(combined)
    diags, diag_routes = zip(*combined)
    diags = list(diags)
    diag_routes = list(diag_routes)

    aoi2rd_dis = []
    index_dis = 0
    for cnt, diag in enumerate(diags):
        dialog = {'task': 'cityreasoning-{}'.format(REGION_EXP), 'id': 'aoi2rd_dis' + str(index_dis), 'diag': []}
        content = json.loads(diag[1]["content"])
        start_poi_name = content['start_name']
        start_poi_addr = content['start_addr']
        start_poi = "{}{}".format(start_poi_name, start_poi_addr)
        routes = content['routes']
        dir_dis_fin = dir_all_dis(routes, secondary_directions, primary_directions, secondary_dir_to_primary_dirs)[0]
        segment = ','.join([str(item) for item in dir_dis_fin])
        st1 = "Step 1: Determine the direction of each segment of the journey in the form of '(The direction you are facing when walking along the road,distance)'. The distance moved in the secondary direction needs to be multiplied by 0.7 and decomposed into the primary direction.:\n" + segment + '\n'

        # 提取"destination"和"by"之间的信息
        end_poi_name = content['dest_name']
        end_poi_addr = content['dest_addr']
        end_poi = "{}{}".format(end_poi_name, end_poi_addr)
        NSEW = {'NS': [], 'EW': []}
        for dir, dis in dir_dis_fin:
            if dir in NS:
                NSEW['NS'].append(dis)
            elif dir in EW:
                NSEW['EW'].append(dis)
            else:
                print(dir)
        NS_dis = []
        EW_dis = []
        for dir, dis in dir_dis_fin:
            if dir in NS:
                NS_dis.append((dir, dis))
            elif dir in EW:
                EW_dis.append((dir, dis))
        st2 = ("Step 2: Calculate distances you move on south-north and east-west.\
                Consider distances you move southward and northward,\
            you move a total of {}-{}={} meters south-north. Consider distances you move westward and eastward,\
            you move a total of {}-{}={} meters east-west.").format(
            max(NS_dis, key=lambda x: x[1])[1], min(NS_dis, key=lambda x: x[1])[1], np.abs(NS_dis[0][1] - NS_dis[1][1]),
            max(EW_dis, key=lambda x: x[1])[1], min(EW_dis, key=lambda x: x[1])[1], np.abs(EW_dis[0][1] - EW_dis[1][1])
        )
        if end_poi not in name_adr_poi or start_poi not in name_adr_poi:
            continue
        aoi_id, aoi_id2 = name_adr_poi[start_poi]['aoi_id'], name_adr_poi[end_poi]['aoi_id']
        if aoi_id not in aoi_dict and aoi_id2 not in aoi_dict:
            continue
        if aoi_id2 in aoi_dict:  # 终点AOI在dict中，使用起点AOI所在道路，终点AOI名字
            aoi_info = map.aois[aoi_id]
            result = get_aoi_address(map, aoi_info)
            if result is None:
                print("No address for start AOI!")
                continue
            aoi_s, rd_name, junc_name, aoi_junc_direction, rd_belong_dir = result
            aoi_name = aoi_dict[aoi_id2]['name']
            aoi_addr = aoi_dict[aoi_id2]['address']
            aoi_name_addr = "{}({})".format(aoi_name, aoi_addr)
            routine = 'After starting from {}, you '.format(start_poi) + ", ".join(diag_routes[cnt]) + ". Finally you arrive at {}.{} is on {},{} is in {}.".format(end_poi, start_poi, rd_name, end_poi,
                                                                                aoi_name_addr)
            if rd_belong_dir in EW:
                distance = np.abs(NSEW['NS'][0] - NSEW['NS'][1])
            elif rd_belong_dir in NS:
                distance = np.abs(NSEW['EW'][0] - NSEW['EW'][1])
            if distance < 50:
                continue
            distances = [str(int(distance / 2)), str(int(distance * 2)), str(int(distance + 500)), str(int(distance))]
            choices = ['{}.{}'.format(x, y) for x, y in zip(['A', 'B', 'C', 'D'], distances)]
            st3 = "Step3: Find the direction of {}. From the address of POI on {}, which is {}, we can find the direction to corner is {}, which means {} runs {}.".format(
                rd_name, rd_name, start_poi_addr, rd_belong_dir, rd_name, dir_map[rd_belong_dir])
            res_dis = dict(zip(["A", "B", "C", "D"], distances))
            for k in res_dis:
                if res_dis[k] == str(int(distance)):
                    label = k
            st4 = "Step4: Calculate the distance between {} and {}. If the road runs south-north, then the answer is the distance you move east-west in step 3.\
            If the road runs east-west, then the answer is the distance you move south-north in step 3.Because {} runs {},\
            so the final answer is the distance you move {} in step 3, which is {} meters.\
            Because {} is in {}, so the answer is {}.".format(end_poi, rd_name, rd_name, dir_map[rd_belong_dir],
                                                            dir_map2[dir_map[rd_belong_dir]], distance, end_poi,
                                                            aoi_name_addr, distance)
            question = "What is the distance between {} and {}?".format(aoi_name_addr, rd_name)
            dialog['diag'].append(
                {'role': 'user', 'content': routine + question + '\n'.join(choices) + "\nLet's think step by step.\n"})
            dialog['diag'].append({'role': 'assistant', 'content': "Answer:{}\n".format(label) + st1 + st2 + st3 + st4})
            aoi2rd_dis.append(dialog)
            index_dis += 1

        elif aoi_id in aoi_dict:  # 起点AOI在dict中，使用终点AOI所在道路
            aoi_info2 = map.aois[aoi_id2]
            result2 = get_aoi_address(map, aoi_info2)
            if result2 is None:
                continue
            aoi_s, rd_name, junc_name, aoi_junc_direction, rd_belong_dir = result2
            aoi_name = aoi_dict[aoi_id]['name']
            aoi_addr = aoi_dict[aoi_id]['address']
            aoi_name_addr = "{}({})".format(aoi_name, aoi_addr)
            routine = 'After starting from {}, you '.format(start_poi) + ", ".join(diag_routes[cnt]) + ". Finally you arrive at {}.{} is on {},{} is in {}.".format(end_poi, end_poi, rd_name, start_poi,
                                                                                aoi_name_addr)
            if rd_belong_dir in EW:
                distance = np.abs(NSEW['NS'][0] - NSEW['NS'][1])
            elif rd_belong_dir in NS:
                distance = np.abs(NSEW['EW'][0] - NSEW['EW'][1])
            if distance < 50:
                continue
            distances = [str(int(distance / 2)), str(int(distance * 2)), str(int(distance + 500)), str(int(distance))]
            letters = ['A', 'B', 'C', 'D']
            choices = ['{}.{}'.format(x, y) for x, y in zip(letters, distances)]
            st3 = "Step3: Find the direction of {}. From the address of POI on {}, which is {}, we can find the direction to corner is {}, which means {} runs {}.".format(
                rd_name, rd_name, start_poi_addr, rd_belong_dir, rd_name, dir_map[rd_belong_dir])
            res_dis = dict(zip(["A", "B", "C", "D"], distances))
            for k in res_dis:
                if res_dis[k] == str(int(distance)):
                    label = k
            st4 = "Step4: Calculate the distance between {} and {}. If the road runs south-north, then the answer is the distance you move east-west in step 3.\
            If the road runs east-west, then the answer is the distance you move south-north in step 3. Because {} runs {},\
            so the final answer is the distance you move {} in step 3, which is {} meters.\
            Because {} is in {}, so the answer is {}.".format(aoi_name_addr, rd_name, rd_name, dir_map[rd_belong_dir],
                                                            dir_map2[dir_map[rd_belong_dir]], distance, start_poi,
                                                            aoi_name_addr, distance)
            question = "What is the distance between {} and {}?".format(aoi_name_addr, rd_name)
            dialog['diag'].append(
                {'role': 'user', 'content': routine + question + '\n'.join(choices) + "\nLet's think step by step.\n"})
            dialog['diag'].append({'role': 'assistant', 'content': "Answer:{}\n".format(label) + st1 + st2 + st3 + st4})
            aoi2rd_dis.append(dialog)
            index_dis += 1
        if index_dis == 1000:
            break
    return aoi2rd_dis


# POI2AOI DIS&DIR，起终点有一个有AOI名字即可
def get_poi2aoi(diags, diag_routes, aoi_dict, name_adr_poi):
    # shuffle the order of dialogs
    combined = list(zip(diags, diag_routes))
    random.shuffle(combined)
    diags, diag_routes = zip(*combined)
    diags = list(diags)
    diag_routes = list(diag_routes)

    poi2aoi_dis = []
    poi2aoi_dir = []
    index_dis = 0
    index_dir = 0
    for cnt, diag in enumerate(diags):
        dialog_dis = {'task': 'cityreasoning-{}'.format(REGION_EXP), 'id': 'poi2aoi_dis' + str(index_dis), 'diag': []}
        dialog_dir = {'task': 'cityreasoning-{}'.format(REGION_EXP), 'id': 'poi2aoi_dir' + str(index_dir), 'diag': []}
        content = json.loads(diag[1]["content"])
        start_poi_name = content['start_name']
        start_poi_addr = content['start_addr']
        start_poi = "{}{}".format(start_poi_name, start_poi_addr)
        end_poi_name = content['dest_name']
        end_poi_addr = content['dest_addr']
        end_poi = "{}{}".format(end_poi_name, end_poi_addr)
        # 计算起终点距离
        distance = compute_length(", ".join(diag_routes[cnt]))
        distances = [str(int(distance / 2)), str(int(distance * 2)), str(int(distance + 500)), str(int(distance))]
        random.shuffle(distances)

        ######距离问题的推理步骤
        distances_in_route = re.findall(r'for (\d+) meters', ", ".join(diag_routes[cnt]))
        distance_strs = ','.join([str(item) + 'm' for item in distances_in_route])
        routes = content['routes']
        dir_dis_fin = dir_all_dis(routes, secondary_directions, primary_directions, secondary_dir_to_primary_dirs)[0]
        segment = ','.join([str(item) for item in dir_dis_fin])
        st1_dir = "Step 1: Determine the direction of each segment of the journey in the form of '(The direction you are facing when walking along the road,distance)'. The distance moved in the secondary direction needs to be multiplied by 0.7 and decomposed into the primary direction:\n" + segment + '\n'
        st1_dis = "Step 1: Find the distance walking along the road of each segment of the journey:\n" + distance_strs
        st2_dis = "Step 2: Add all distances above together to get the total distance:\n" + '+'.join(
            distances_in_route) + "={}".format(distance) + ".So the answer is {}m.\n".format(distance)
        ########方向问题的推理步骤
        NSEW = {'NS': [], 'EW': []}
        for dir, dis in dir_dis_fin:
            if dir in NS:
                NSEW['NS'].append(dis)
            elif dir in EW:
                NSEW['EW'].append(dis)
            else:
                print(dir)
        NS_dis = []
        EW_dis = []
        for dir, dis in dir_dis_fin:
            if dir in NS:
                NS_dis.append((dir, dis))
            elif dir in EW:
                EW_dis.append((dir, dis))
        NS_dir = max(NS_dis, key=lambda x: x[1])[0]
        EW_dir = max(EW_dis, key=lambda x: x[1])[0]
        if np.abs(NS_dis[0][1] - NS_dis[1][1]) > np.abs(EW_dis[0][1] - EW_dis[1][1]):
            direction = NS_dir
        elif np.abs(NS_dis[0][1] - NS_dis[1][1]) < np.abs(EW_dis[0][1] - EW_dis[1][1]):
            direction = EW_dir
        else:
            continue
        directions = ['east', 'west', 'south', 'north']
        random.shuffle(directions)
        st2_dir = "Step 2: Calculate distances you move on south-north and east-west.\
            Consider distances you move southward and northward,\
            you move a total of {}-{}={} meters towards {}. Consider distances you move westward and eastward,\
            you move a total of {}-{}={} meters towards {}.".format(
            max(NS_dis, key=lambda x: x[1])[1], min(NS_dis, key=lambda x: x[1])[1], np.abs(NS_dis[0][1] - NS_dis[1][1]),
            NS_dir,
            max(EW_dis, key=lambda x: x[1])[1], min(EW_dis, key=lambda x: x[1])[1], np.abs(EW_dis[0][1] - EW_dis[1][1]),
            EW_dir
        )
        st3_dir = "Step 3: Consider distances you move towards {} and {}. Because the distance you move towards {} is longer, so the answer is {}.".format(
            NS_dir, EW_dir, direction, direction)
        
        if end_poi not in name_adr_poi or start_poi not in name_adr_poi:
            continue
        aoi_id, aoi_id2 = name_adr_poi[end_poi]['aoi_id'], name_adr_poi[start_poi]['aoi_id']
        if aoi_id not in aoi_dict and aoi_id2 not in aoi_dict:
            continue
        # AOI2POI&POI2AOI
        if aoi_id not in aoi_dict:  # AOI2POI
            start_aoi_name, start_aoi_addr = aoi_dict[aoi_id2]['name'], aoi_dict[aoi_id2]['address']
            start_aoi_name_addr = "{}({})".format(start_aoi_name, start_aoi_addr)
            routine = 'After starting from {}, you '.format(start_poi) + ", ".join(diag_routes[cnt]) + ". Finally you arrive at {}.{} is in {}.".format(end_poi, start_poi, start_aoi_name_addr)
            question_dis = "How many meters do I need to walk from {} to {} along the road?".format(start_aoi_name_addr,
                                                                                                    end_poi)
            question_dir = "In which direction is {} from {}?".format(end_poi, start_aoi_name_addr)
            res_dis = dict(zip(["A", "B", "C", "D"], distances))
            res_dir = dict(zip(["A", "B", "C", "D"], directions))
            for k in res_dis:
                if res_dis[k] == str(int(distance)):
                    label1 = k
            for k in res_dir:
                if res_dir[k] == direction:
                    label2 = k
            assistant_dis = "Answer:{}\n".format(label1) + st1_dis + st2_dis
            assistant_dir = "Answer:{}\n".format(label2) + st1_dir + st2_dir + st3_dir
            dialog_dis['diag'].append({'role': 'user', 'content': routine + question_dis + '\n'.join(
                distances) + "\nLet's think step by step.\n"})
            dialog_dis['diag'].append({'role': 'assistant', 'content': assistant_dis})
            dialog_dir['diag'].append({'role': 'user', 'content': routine + question_dir + '\n'.join(
                directions) + "\nLet's think step by step.\n"})
            dialog_dir['diag'].append({'role': 'assistant', 'content': assistant_dir})

            poi2aoi_dis.append(dialog_dis)
            poi2aoi_dir.append(dialog_dir)
            index_dis += 1
            index_dir += 1
        else:  # POI2AOI
            end_aoi_name, end_aoi_addr = aoi_dict[aoi_id]['name'], aoi_dict[aoi_id]['address']
            end_aoi_name_addr = "{}({})".format(end_aoi_name, end_aoi_addr)
            routine = 'After starting from {}, you '.format(start_poi) + ", ".join(diag_routes[cnt]) + ". Finally you arrive at {}.{} is in {}.".format(end_poi, end_poi, end_aoi_name_addr)
            question_dis = "How many meters do I need to walk from {} to {} along the road?".format(start_poi,
                                                                                                    end_aoi_name_addr)
            question_dir = "In which direction is {} from {}?".format(end_aoi_name_addr, start_poi)
            res_dis = dict(zip(["A", "B", "C", "D"], distances))
            res_dir = dict(zip(["A", "B", "C", "D"], directions))
            # print(distance,distances)
            for k in res_dis:
                if res_dis[k] == str(int(distance)):
                    label1 = k
            for k in res_dir:
                if res_dir[k] == direction:
                    label2 = k
            assistant_dis = "Answer:{}\n".format(label1) + st1_dis + st2_dis
            assistant_dir = "Answer:{}\n".format(label2) + st1_dir + st2_dir + st3_dir
            dialog_dis['diag'].append({'role': 'user', 'content': routine + question_dis + '\n'.join(
                distances) + "\nLet's think step by step.\n"})
            dialog_dis['diag'].append({'role': 'assistant', 'content': assistant_dis})

            dialog_dir['diag'].append({'role': 'user', 'content': routine + question_dir + '\n'.join(
                directions) + "\nLet's think step by step.\n"})
            dialog_dir['diag'].append({'role': 'assistant', 'content': assistant_dir})
            poi2aoi_dis.append(dialog_dis)
            poi2aoi_dir.append(dialog_dir)
            index_dis += 1
            index_dir += 1
        if index_dis == 1000:
            break
    return poi2aoi_dir,poi2aoi_dis


########POI2RD_DIR,终点POI到起点道路方向
def get_poi2rd_dir(map, diags, diag_routes, name_adr_poi):
    # shuffle the order of dialogs
    combined = list(zip(diags, diag_routes))
    random.shuffle(combined)
    diags, diag_routes = zip(*combined)
    diags = list(diags)
    diag_routes = list(diag_routes)

    poi2rd_dir = []
    index_dir = 0
    for cnt, diag in enumerate(diags):
        dialog = {'task': 'cityreasoning-{}'.format(REGION_EXP), 'id': 'poi2rd_dir' + str(index_dir), 'diag': []}
        content = json.loads(diag[1]["content"])
        routes = content['routes']
        start_poi_name = content['start_name']
        start_poi_addr = content['start_addr']
        start_poi = "{}{}".format(start_poi_name, start_poi_addr)
        end_poi_name = content['dest_name']
        end_poi_addr = content['dest_addr']
        end_poi = "{}{}".format(end_poi_name, end_poi_addr)
        routine = 'After starting from {}, you '.format(start_poi) + ", ".join(diag_routes[cnt]) + ". Finally you arrive at {}.".format(end_poi)
        dir_dis_fin = dir_all_dis(routes, secondary_directions, primary_directions, secondary_dir_to_primary_dirs)[0]
        segment = ','.join([str(item) for item in dir_dis_fin])
        st1 = "Step 1: Determine the direction of each segment of the journey in the form of '(The direction you are facing when walking along the road,distance)'. The distance moved in the secondary direction needs to be multiplied by 0.7 and decomposed into the primary direction.:\n" + segment + '\n'
        dir_set = set()  # 净位移较大的两个正方向
        NS_dis = []
        EW_dis = []
        for dir, dis in dir_dis_fin:
            if dir in NS:
                NS_dis.append((dir, dis))
            elif dir in EW:
                EW_dis.append((dir, dis))
        dir1 = max(NS_dis, key=lambda x: x[1])[0]
        dir2 = max(EW_dis, key=lambda x: x[1])[0]
        dir_set.add(dir1)
        dir_set.add(dir2)
        NSEW = {'NS': [], 'EW': []}
        for dir, dis in dir_dis_fin:
            if dir in NS:
                NSEW['NS'].append(dis)
            elif dir in EW:
                NSEW['EW'].append(dis)
            else:
                print(dir)
        st2 = "Step 2: Find in which direction is destination {} from origin {}. Consider distances you move southward and northward,\
            you move a total of {}-{}={} meters towards {}. Consider distances you move westward and eastward,\
                you move a total of {}-{}={} meters towards {}.".format(
            end_poi, start_poi, max(NS_dis, key=lambda x: x[1])[1], min(NS_dis, key=lambda x: x[1])[1],
            np.abs(NS_dis[0][1] - NS_dis[1][1]), max(NS_dis, key=lambda x: x[1])[0],
            max(EW_dis, key=lambda x: x[1])[1], min(EW_dis, key=lambda x: x[1])[1], np.abs(EW_dis[0][1] - EW_dis[1][1]),
            max(EW_dis, key=lambda x: x[1])[0])
        if start_poi not in name_adr_poi:
            continue
        start_aoi_id = name_adr_poi[start_poi]['aoi_id']
        start_aoi_info = map.aois[start_aoi_id]
        result = get_aoi_address(map, start_aoi_info)
        if result is None:
            continue
        aoi_s, rd_name, junc_name, aoi_junc_direction, rd_belong_dir = result
        st3 = "Step3: Find the direction of {}. From the address of POI on {}, which is {}, we can find the direction to road is {}, which means {} runs {}.".format(
            rd_name, rd_name, start_poi_addr, rd_belong_dir, rd_name, dir_map[rd_belong_dir])

        # if 270<angle<360:  #终点在起点westnorth方向
        if dir_set == {'west', 'north'}:
            if rd_belong_dir in NS:
                tar_dir = 'north'
            elif rd_belong_dir in EW:
                tar_dir = 'west'
        elif dir_set == {'east', 'north'}:  ##终点在起点eastnorth方向
            if rd_belong_dir in NS:
                tar_dir = 'north'
            elif rd_belong_dir in EW:
                tar_dir = 'east'
        elif dir_set == {'east', 'south'}:  # 终点在起点eastsouth方向
            if rd_belong_dir in NS:
                tar_dir = 'south'
            elif rd_belong_dir in EW:
                tar_dir = 'east'
        elif dir_set == {'west', 'south'}:  # 终点在起点westsouth方向
            if rd_belong_dir in NS:
                tar_dir = 'south'
            elif rd_belong_dir in EW:
                tar_dir = 'west'
        random.shuffle(primary_directions)
        res_dict = dict(zip(["A", "B", "C", "D"], primary_directions))
        letters = ['A', 'B', 'C', 'D']
        choices = ['{}.{}'.format(x, y) for x, y in zip(letters, primary_directions)]
        for k in res_dict:
            if res_dict[k] == tar_dir:
                label = k
        fill = {
            "end_poi": end_poi,
            "start_poi": start_poi,
            "rd_name": rd_name,
            "rd_dir_set": dir_map[rd_belong_dir],
            "dir_set": dir_set,
            "answer": tar_dir
        }
        st4 = ("Step4: Find in which direction is {end_poi} from {rd_name}.If {rd_name} runs south-north,"
            + "then the possible answers are 'east','west'; If {rd_name} runs east-west,"
            + "then the possible answers are 'north','south'. If the origin is on {rd_name}, then we can choose answer from the direction from origin to destination,"
            + "which contains {dir_set}. Because the origin {start_poi} is on {rd_name},"
            + "and {rd_name} runs {rd_dir_set}, so the final answer is {answer}.").format(**fill)
        question = "In which direction is {} from {}?".format(end_poi, rd_name)
        dialog['diag'].append(
            {'role': 'user', 'content': routine + question + '\n'.join(choices) + "\nLet's think step by step.\n"})
        dialog['diag'].append({'role': 'assistant', 'content': "Answer:{}\n".format(label) + st1 + st2 + st3 + st4})
        poi2rd_dir.append(dialog)
        index_dir += 1
        if index_dir == 1000:
            break
    return poi2rd_dir



########AOI2RD_DIR,终点AOI到起点道路方向
""" 
"After Starting from 建国小区(志新west路east侧, 距离志新路和志新west路交叉口north角500m), you walk along 志新west路 for 800 meters 从south到north, then go 从east到west towards 志新west路和清华east路交叉口, walk along 清华east路 for 200 meters 从east到west, then go 从east到west towards 清华east路和清华east路交叉口, walk along 清华east路 for 700 meters 从east到west, then go 从north到south towards 清华east路和学院路交叉口, walk along 学院路 for 100 meters 从north到south, then go 从north到south towards 学院路和学院路交叉口, walk along 学院路 for 500 meters 从north到south .Finally you arrive at 学生活动中心(中关村north大街north侧清华west门north区内, 距离中关村north大街和中关村north大街交叉口westnorth角300m).建国小区(志新west路east侧, 距离志新路和志新west路交叉口north角500m) is on 志新west路,学生活动中心(中关村north大街north侧清华west门north区内, 距离中关村north大街和中关村north大街交叉口westnorth角300m) is in 老场院north区(中关村north大街north侧, 距离中关村north大街和中关村north大街交叉口westnorth角50m以内).In which direction is 老场院north区(中关村north大街north侧, 距离中关村north大街和中关村north大街交叉口westnorth角300m) from 志新west路?\nA.east\nB.south\nC.west\nD.north\n
Step1.Devide above routine into segments:800 meters 从south到north, 200 meters 从east到west,700 meters 从east到west, 100 meters 从north到south,500 meters 从north到south .
Step2.Find in which direction is destination (学生活动中心(中关村north大街north侧清华west门north区内, 距离中关村north大街和中关村north大街交叉口westnorth角50m以内)) from origin(建国小区(志新west路east侧, 距离志新路和志新west路交叉口north角500m)). In the routine,you  need to move northward 800 meters,move westward 200+700=900 meters,move southward 100+500=600 meters.Consider distances you move southward and northward,800 is larger than 600,so you move totally northward.Consider distances you move westward and southward,900 is larger than 0,so you move totally westward.So the direction from origin to destination contains "west"，"north".
Step3.Find the direction of 志新west路.From the address of 建国小区 on 志新west路,which is 志新west路east侧, 距离志新路和志新west路交叉口north角500m,we can find the direction to junction is "north",which means 志新west路 runs south-north.
Step4.Answer:C.Find in which direction is 老场院north区(中关村north大街north侧, 距离中关村north大街和中关村north大街交叉口westnorth角50m以内) from 志新west路.If 志新west路 runs south-north,then the possible answers are  "east", "west"; If 志新west路 runs east-west,then the possible answers are "north","south".If the origin is on 志新west路,then  we can choose answer from the direction from origin to destination ，which contains "west"、"north".Because the origin 建国小区 is on 志新west路,and 志新west路 runs south-north,so the final answer is west.
"""
def get_aoi2rd_dir(map, diags, diag_routes, aoi_dict, name_adr_poi):
    # shuffle the order of dialogs
    combined = list(zip(diags, diag_routes))
    random.shuffle(combined)
    diags, diag_routes = zip(*combined)
    diags = list(diags)
    diag_routes = list(diag_routes)

    aoi2rd_dir = []
    index_dir_aoi = 0
    for cnt, diag in enumerate(diags):
        dialog = {'task': 'cityreasoning-{}'.format(REGION_EXP), 'id': 'aoi2rd_dir' + str(index_dir_aoi), 'diag': []}
        content = json.loads(diag[1]["content"])
        routes = content['routes']
        start_poi_name = content['start_name']
        start_poi_addr = content['start_addr']
        start_poi = "{}{}".format(start_poi_name, start_poi_addr)
        end_poi_name = content['dest_name']
        end_poi_addr = content['dest_addr']
        end_poi = "{}{}".format(end_poi_name, end_poi_addr)
        if end_poi not in name_adr_poi:
            continue

        aoi_id = name_adr_poi[end_poi]['aoi_id']
        if aoi_id not in aoi_dict:
            continue
        end_aoi_name = aoi_dict[aoi_id]['name']
        end_aoi_addr = aoi_dict[aoi_id]['address']
        end_aoi_name_addr = "{}({})".format(end_aoi_name, end_aoi_addr)

        routine = 'After starting from {}, you '.format(start_poi) + ", ".join(diag_routes[cnt]) + ". Finally you arrive at {}.".format(end_poi)
        dir_dis_fin = dir_all_dis(routes, secondary_directions, primary_directions, secondary_dir_to_primary_dirs)[0]
        segment = ','.join([str(item) for item in dir_dis_fin])
        st1 = "Step 1: Determine the direction of each segment of the journey in the form of '(The direction you are facing when walking along the road,distance)'. The distance moved in the secondary direction needs to be multiplied by 0.7 and decomposed into the primary direction:\n" + segment + '\n'
        dir_set = set()  # 净位移较大的两个正方向
        NS_dis = []
        EW_dis = []
        for dir, dis in dir_dis_fin:
            if dir in NS:
                NS_dis.append((dir, dis))
            elif dir in EW:
                EW_dis.append((dir, dis))
        dir1 = max(NS_dis, key=lambda x: x[1])[0]
        dir2 = max(EW_dis, key=lambda x: x[1])[0]
        dir_set.add(dir1)
        dir_set.add(dir2)
        NSEW = {'NS': [], 'EW': []}
        for dir, dis in dir_dis_fin:
            if dir in NS:
                NSEW['NS'].append(dis)
            elif dir in EW:
                NSEW['EW'].append(dis)
            else:
                print(dir)
        st2 = "Step 2: Find in which direction is destination {} from origin {}. Consider distances you move southward and northward,\
            you move a total of {}-{}={} meters towards {}. Consider distances you move westward and eastward,\
                you move a total of {}-{}={} meters towards {}.".format(
            end_poi, start_poi, max(NS_dis, key=lambda x: x[1])[1], min(NS_dis, key=lambda x: x[1])[1],
            np.abs(NS_dis[0][1] - NS_dis[1][1]), max(NS_dis, key=lambda x: x[1])[0],
            max(EW_dis, key=lambda x: x[1])[1], min(EW_dis, key=lambda x: x[1])[1], np.abs(EW_dis[0][1] - EW_dis[1][1]),
            max(EW_dis, key=lambda x: x[1])[0])
        if start_poi not in name_adr_poi:
            continue

        start_aoi_id = name_adr_poi[start_poi]['aoi_id']
        start_aoi_info = map.aois[start_aoi_id]
        result = get_aoi_address(map, start_aoi_info)
        if result is None:
            continue
        aoi_s, rd_name, junc_name, aoi_junc_direction, rd_belong_dir = result
        st3 = "Step3: Find the direction of {}. From the address of POI on {}, which is {}, we can find the direction to corner is {}, which means {} runs {}.".format(
            rd_name, rd_name, start_poi_addr, rd_belong_dir, rd_name, dir_map[rd_belong_dir])

        # if 270<angle<360:  #终点在起点westnorth方向
        if dir_set == {'west', 'north'}:
            if rd_belong_dir in NS:
                tar_dir = 'west'
            elif rd_belong_dir in EW:
                tar_dir = 'north'
        elif dir_set == {'east', 'north'}:  ##终点在起点eastnorth方向
            if rd_belong_dir in NS:
                tar_dir = 'east'
            elif rd_belong_dir in EW:
                tar_dir = 'north'
        elif dir_set == {'east', 'south'}:  # 终点在起点eastsouth方向
            if rd_belong_dir in NS:
                tar_dir = 'east'
            elif rd_belong_dir in EW:
                tar_dir = 'south'
        elif dir_set == {'west', 'south'}:  # 终点在起点westsouth方向
            if rd_belong_dir in NS:
                tar_dir = 'west'
            elif rd_belong_dir in EW:
                tar_dir = 'south'
        random.shuffle(primary_directions)
        res_dict = dict(zip(["A", "B", "C", "D"], primary_directions))
        letters = ['A', 'B', 'C', 'D']
        choices = ['{}.{}'.format(x, y) for x, y in zip(letters, primary_directions)]
        for k in res_dict:
            if res_dict[k] == tar_dir:
                label = k
        fill = {
            "end_aoi": end_aoi_name_addr,
            "start_poi": start_poi,
            "rd_name": rd_name,
            "rd_dir_set": dir_map[rd_belong_dir],
            "dir_set": dir_set,
            "answer": tar_dir,
            "label": label
        }
        st4 = ("Step4.Answer:{label}. Find in which direction is {end_aoi} from {rd_name}. If {rd_name} runs south-north,"
            + "then the possible answers are  'east','west'; If {rd_name} runs east-west,"
            + "then the possible answers are 'north','south'. If the origin is on {rd_name}, then  we can choose answer from the direction from origin to destination,"
            + "which contains {dir_set}.Because the origin {start_poi} is on {rd_name},"
            + "and {rd_name} runs {rd_dir_set},so the final answer is {answer}.").format(**fill)
        question = "In which direction is {} from {}?".format(end_aoi_name_addr, rd_name)
        dialog['diag'].append(
            {'role': 'user', 'content': routine + question + '\n'.join(choices) + "\nLet's think step by step.\n"})
        dialog['diag'].append({'role': 'assistant', 'content': st1 + st2 + st3 + st4})
        aoi2rd_dir.append(dialog)
        index_dir_aoi += 1
        if index_dir_aoi == 1000:
            break
    return aoi2rd_dir


def get_poi2poi_dir(diags, diag_routes):
    # #######对方向推理训练数据进行泛化
    # shuffle the order of dialogs
    combined = list(zip(diags, diag_routes))
    random.shuffle(combined)
    diags, diag_routes = zip(*combined)
    diags = list(diags)
    diag_routes = list(diag_routes)

    all_train_data = []
    for cnt, train in enumerate(diags):
        if cnt <= 1000:
            all_train_data.append((train, cnt))

    dir_seg1 = all_train_data[:int(len(all_train_data) * 0.25)]
    dir_seg2 = all_train_data[int(len(all_train_data) * 0.25) + 1:int(len(all_train_data) * 0.5)]
    dir_seg3 = all_train_data[int(len(all_train_data) * 0.5) + 1:int(len(all_train_data) * 0.75)]
    dir_seg4 = all_train_data[int(len(all_train_data) * 0.75) + 1:]
    # ######单轮，整数
    dialogs_dir1 = []
    index_dir = 0
    task_name = 'cityreasoning-{}'.format(REGION_EXP)
    for train, cnt in dir_seg1:
        dialog = {'task': task_name, 'id': 'poi2poi_dir' + str(index_dir), 'diag': []}
        content = json.loads(train[1]["content"])
        start_poi_name = content['start_name']
        start_poi_addr = content['start_addr']
        start_poi = "{}{}".format(start_poi_name, start_poi_addr)
        end_poi_name = content['dest_name']
        end_poi_addr = content['dest_addr']
        end_poi = "{}{}".format(end_poi_name, end_poi_addr)
        routes = content['routes']
        dir_dis_fin = dir_all_dis(routes, secondary_directions, primary_directions, secondary_dir_to_primary_dirs)[0]
        user_head = "Let's think step by step.\n"
        ques = "In which direction is {} from {}?\n".format(end_poi, start_poi)
        directions = ['east', 'south', 'west', 'north']
        random.shuffle(directions)
        letters = ['A', 'B', 'C', 'D']
        choices = ['{}.{}'.format(x, y) for x, y in zip(letters, directions)]
        segment = ','.join([str(item) for item in dir_dis_fin])
        st1 = "Step 1: Determine the direction of each segment of the journey in the form of '(The direction you are facing when walking along the road,distance)'. The distance moved in the secondary direction needs to be multiplied by 0.7 and decomposed into the primary direction.:\n" + segment + '\n'
        st2 = "Step 2: Analyze the overall direction of the journey:\n" + ','.join(
            [str(item) for item in [(dir_dis[0], int(dir_dis[1])) for dir_dis in dir_dis_fin]]) + '\n'
        east_west = sorted(
            [(direction, value) for direction, value in dir_dis_fin if direction in ('east', 'west')],
            key=lambda x: x[1], reverse=True)
        north_south = sorted(
            [(direction, value) for direction, value in dir_dis_fin if direction in ('north', 'south')],
            key=lambda x: x[1], reverse=True)
        
        result = [(east_west[0][0], east_west[0][1], east_west[1][1]),
                (north_south[0][0], north_south[0][1], north_south[1][1])]
        fin_direction = max(result, key=lambda x: abs(x[1] - x[2]))
        fin_direction2 = min(result, key=lambda x: abs(x[1] - x[2]))
        st3 = "Step 3:{} is larger than {}, so there is a {} travel. {} is larger than {}, so there is a {} travel.\n".format(
            int(east_west[0][1]), int(east_west[1][1]), east_west[0][0], int(north_south[0][1]), int(north_south[1][1]),
            north_south[0][0])
        st4 = "Step 4: Compare above two directions, {}:{}-{}={},{}:{}-{}={}. {} is larger than {}, so the overall direction is {}. So the answer is {}.\n".format(
            fin_direction[0], int(fin_direction[1]), int(fin_direction[2]), int(fin_direction[1] - fin_direction[2]),
            fin_direction2[0], int(fin_direction2[1]), int(fin_direction2[2]), int(fin_direction2[1] - fin_direction2[2]),
            int(fin_direction[1] - fin_direction[2]), int(fin_direction2[1] - fin_direction2[2]), fin_direction[0],
            fin_direction[0])
        for cnt2, direction in enumerate(directions):
            if direction == fin_direction[0]:
                answer = letters[cnt2]
        routine = "After starting from {}{}, you can arrive at the destination {}{} by following the navigation instruction below:\n".format(start_poi_name, start_poi_addr, end_poi_name, end_poi_addr) + ", ".join(diag_routes[cnt])
        assistant2 = "Answer:{}\n".format(answer) + st1 + st2 + st3 + st4
        dialog['diag'].append(
            {'role': 'user', 'content': routine + ". " + ques + '\n'.join(choices) + "\nLet's think step by step.\n"})
        dialog['diag'].append({'role': 'assistant', 'content': assistant2})
        dialogs_dir1.append(dialog)
        index_dir += 1

    ########单轮，非整数
    for train, cnt in dir_seg2:
        dialog = {'task': task_name, 'id': 'poi2poi_dir' + str(index_dir), 'diag': []}
        user_head = "Let's think step by step.\n"
        content = json.loads(train[1]["content"])
        start_poi_name = content['start_name']
        start_poi_addr = content['start_addr']
        start_poi = "{}{}".format(start_poi_name, start_poi_addr)
        end_poi_name = content['dest_name']
        end_poi_addr = content['dest_addr']
        end_poi = "{}{}".format(end_poi_name, end_poi_addr)
        routes = content['routes']
        dir_dis_fin = dir_all_dis(routes, secondary_directions, primary_directions, secondary_dir_to_primary_dirs)[0]
        ques = "In which direction is {} from {}?\n".format(end_poi, start_poi)
        directions = ['east', 'south', 'west', 'north']
        random.shuffle(directions)
        letters = ['A', 'B', 'C', 'D']
        choices = ['{}.{}'.format(x, y) for x, y in zip(letters, directions)]
        segment = ','.join([str(item) for item in dir_dis_fin])
        st1 = "Step 1: Determine the direction of each segment of the journey in the form of '(The direction you are facing when walking along the road,distance)'. The distance moved in the secondary direction needs to be multiplied by 0.7 and decomposed into the primary direction.:\n" + segment + '\n'
        st2 = "Step 2: Analyze the overall direction of the journey:\n" + ','.join(
            [str(item) for item in dir_dis_fin]) + '\n'
        east_west = sorted(
            [(direction, value) for direction, value in dir_dis_fin if direction in ('east', 'west')],
            key=lambda x: x[1], reverse=True)
        north_south = sorted(
            [(direction, value) for direction, value in dir_dis_fin if direction in ('north', 'south')],
            key=lambda x: x[1], reverse=True)
        result = [(east_west[0][0], east_west[0][1], east_west[1][1]),
                (north_south[0][0], north_south[0][1], north_south[1][1])]
        fin_direction = max(result, key=lambda x: abs(x[1] - x[2]))
        fin_direction2 = min(result, key=lambda x: abs(x[1] - x[2]))
        st3 = "Step 3: {} is larger than {}, so there is a {} travel. {} is larger than {}, so there is a {} travel.\n".format(
            east_west[0][1], east_west[1][1], east_west[0][0], north_south[0][1], north_south[1][1], north_south[0][0])
        st4 = "Step 4: Compare above two directions, {}:{}-{}={},{}:{}-{}={}.{} is larger than {}, so the overall direction is {}. So the answer is {}.\n".format(
            fin_direction[0], fin_direction[1], fin_direction[2], fin_direction[1] - fin_direction[2], fin_direction2[0],
            fin_direction2[1], fin_direction2[2], fin_direction2[1] - fin_direction2[2],
                                                                fin_direction[1] - fin_direction[2],
                                                                fin_direction2[1] - fin_direction2[2], fin_direction[0],
            fin_direction[0])
        for cnt2, direction in enumerate(directions):
            if direction == fin_direction[0]:
                answer = letters[cnt2]
        routine = "After starting from {}{}, you can arrive at the destination {}{} by following the navigation instruction below:\n".format(start_poi_name, start_poi_addr, end_poi_name, end_poi_addr) + ", ".join(diag_routes[cnt])
        assistant2 = "Answer:{}\n".format(answer) + st1 + st2 + st3 + st4
        dialog['diag'].append(
            {'role': 'user', 'content': routine + ". " + ques + '\n'.join(choices) + "\nLet's think step by step.\n"})
        dialog['diag'].append({'role': 'assistant', 'content': assistant2})
        dialogs_dir1.append(dialog)
        index_dir += 1
    #######多轮，整数
    for train, cnt in dir_seg3:
        dialog = {'task': task_name, 'id': 'poi2poi_dir' + str(index_dir), 'diag': []}
        user_head = "Let's think step by step.\n"
        content = json.loads(train[1]["content"])
        start_poi_name = content['start_name']
        start_poi_addr = content['start_addr']
        start_poi = "{}{}".format(start_poi_name, start_poi_addr)
        end_poi_name = content['dest_name']
        end_poi_addr = content['dest_addr']
        end_poi = "{}{}".format(end_poi_name, end_poi_addr)
        routes = content['routes']
        dir_dis_fin = dir_all_dis(routes, secondary_directions, primary_directions, secondary_dir_to_primary_dirs)[0]
        ques = "In which direction is {} from {}?".format(end_poi, start_poi)
        directions = ['east', 'south', 'west', 'north']
        random.shuffle(directions)
        letters = ['A', 'B', 'C', 'D']
        choices = ['{}.{}'.format(x, y) for x, y in zip(letters, directions)]
        user1 = ques + '\n' + '\n'.join(
            choices) + '\n' + 'To answer the question, please think about the navigation instruction about starting from {} and arriving at the destination {}'.format(
            start_poi, end_poi)
        user2 = 'Given the navigation instruction,please choose the most suitable one among A, B, C and D as the answer to above question.' + user_head
        segment = ','.join([str(item) for item in dir_dis_fin])

        st1 = "Step 1: Determine the direction of each segment of the journey in the form of '(The direction you are facing when walking along the road,distance)'. The distance moved in the secondary direction needs to be multiplied by 0.7 and decomposed into the primary direction.:\n" + segment + '\n'
        st2 = "Step 2: Analyze the overall direction of the journey:\n" + ','.join(
            [str(item) for item in [(dir_dis[0], int(dir_dis[1])) for dir_dis in dir_dis_fin]]) + '\n'
        east_west = sorted(
            [(direction, value) for direction, value in dir_dis_fin if direction in ('east', 'west')],
            key=lambda x: x[1], reverse=True)
        north_south = sorted(
            [(direction, value) for direction, value in dir_dis_fin if direction in ('north', 'south')],
            key=lambda x: x[1], reverse=True)
        result = [(east_west[0][0], east_west[0][1], east_west[1][1]),
                (north_south[0][0], north_south[0][1], north_south[1][1])]
        fin_direction = max(result, key=lambda x: abs(x[1] - x[2]))
        fin_direction2 = min(result, key=lambda x: abs(x[1] - x[2]))
        st3 = "Step 3: {} is larger than {}, so there is a {} travel. {} is larger than {}, so there is a {} travel.\n".format(
            int(east_west[0][1]), int(east_west[1][1]), east_west[0][0], int(north_south[0][1]), int(north_south[1][1]),
            north_south[0][0])
        st4 = "Step 4: Compare above two directions,{}:{}-{}={},{}:{}-{}={}.{} is larger than {}, so the overall direction is {}. So the answer is {}.\n".format(
            fin_direction[0], int(fin_direction[1]), int(fin_direction[2]), int(fin_direction[1] - fin_direction[2]),
            fin_direction2[0], int(fin_direction2[1]), int(fin_direction2[2]), int(fin_direction2[1] - fin_direction2[2]),
            int(fin_direction[1] - fin_direction[2]), int(fin_direction2[1] - fin_direction2[2]), fin_direction[0],
            fin_direction[0])
        for cnt2, direction in enumerate(directions):
            if direction == fin_direction[0]:
                answer = letters[cnt2]
        routine = "After starting from {}{}, you can arrive at the destination {}{} by following the navigation instruction below:\n".format(start_poi_name, start_poi_addr, end_poi_name, end_poi_addr) + ", ".join(diag_routes[cnt])
        assistant1 = "Navigation instruction:\n" + routine
        assistant2 = "Answer:{}\n".format(answer) + st1 + st2 + st3 + st4
        dialog['diag'].append({'role': 'user', 'content': user1})
        dialog['diag'].append({'role': 'assistant', 'content': assistant1})
        dialog['diag'].append({'role': 'user', 'content': user2})
        dialog['diag'].append({'role': 'assistant', 'content': assistant2})
        dialogs_dir1.append(dialog)
        index_dir += 1

    ######多轮，非整数
    for train, cnt in dir_seg4:
        dialog = {'task': task_name, 'id': 'poi2poi_dir' + str(index_dir), 'diag': []}
        user_head = "Let's think step by step.\n"
        content = json.loads(train[1]["content"])
        start_poi_name = content['start_name']
        start_poi_addr = content['start_addr']
        start_poi = "{}{}".format(start_poi_name, start_poi_addr)
        end_poi_name = content['dest_name']
        end_poi_addr = content['dest_addr']
        end_poi = "{}{}".format(end_poi_name, end_poi_addr)
        routes = content['routes']
        dir_dis_fin = dir_all_dis(routes, secondary_directions, primary_directions, secondary_dir_to_primary_dirs)[0]
        directions = ['east', 'south', 'west', 'north']
        random.shuffle(directions)
        letters = ['A', 'B', 'C', 'D']
        choices = ['{}.{}'.format(x, y) for x, y in zip(letters, directions)]
        user1 = ques + '\n' + '\n'.join(
            choices) + '\n' + 'To answer the question,please think about the navigation instruction about starting from {} and  arriving at the destination  {}'.format(
            start_poi, end_poi)
        user2 = 'Given the navigation instruction,please choose the most suitable one among A, B, C and D as the answer to above question.' + user_head
        dir_dis_fin= dir_all_dis(routes, secondary_directions, primary_directions,
                                           secondary_dir_to_primary_dirs)[0]
        segment = ','.join([str(item) for item in dir_dis_fin])

        st1 = "Step 1: Determine the direction of each segment of the journey in the form of '(The direction you are facing when walking along the road,distance)'. The distance moved in the secondary direction needs to be multiplied by 0.7 and decomposed into the primary direction.:\n" + segment + '\n'
        st2 = "Step 2: Analyze the overall direction of the journey:\n" + ','.join(
            [str(item) for item in dir_dis_fin]) + '\n'
        east_west = sorted(
            [(direction, value) for direction, value in dir_dis_fin if direction in ('east', 'west')],
            key=lambda x: x[1], reverse=True)
        north_south = sorted(
            [(direction, value) for direction, value in dir_dis_fin if direction in ('north', 'south')],
            key=lambda x: x[1], reverse=True)
        result = [(east_west[0][0], east_west[0][1], east_west[1][1]),
                (north_south[0][0], north_south[0][1], north_south[1][1])]
        fin_direction = max(result, key=lambda x: abs(x[1] - x[2]))
        fin_direction2 = min(result, key=lambda x: abs(x[1] - x[2]))
        st3 = "Step 3: {} is larger than {}, so there is a {} travel. {} is larger than {}, so there is a {} travel.\n".format(
            east_west[0][1], east_west[1][1], east_west[0][0], north_south[0][1], north_south[1][1], north_south[0][0])
        st4 = "Step 4: Compare above two directions, {}:{}-{}={}, {}:{}-{}={}. {} is larger than {}, so the overall direction is {}. So the answer is {}.\n".format(
            fin_direction[0], fin_direction[1], fin_direction[2], fin_direction[1] - fin_direction[2], fin_direction2[0],
            fin_direction2[1], fin_direction2[2], fin_direction2[1] - fin_direction2[2],
                                                                fin_direction[1] - fin_direction[2],
                                                                fin_direction2[1] - fin_direction2[2], fin_direction[0],
            fin_direction[0])
        for cnt2, direction in enumerate(directions):
            if direction == fin_direction[0]:
                answer = letters[cnt2]
        routine = "After starting from {}{}, you can arrive at the destination {}{} by following the navigation instruction below:\n".format(start_poi_name, start_poi_addr, end_poi_name, end_poi_addr) + ", ".join(diag_routes[cnt])
        assistant1 = "Navigation instruction:\n" + routine
        assistant2 = "Answer:{}\n".format(answer) + st1 + st2 + st3 + st4
        dialog['diag'].append({'role': 'user', 'content': user1})
        dialog['diag'].append({'role': 'assistant', 'content': assistant1})
        dialog['diag'].append({'role': 'user', 'content': user2})
        dialog['diag'].append({'role': 'assistant', 'content': assistant2})
        dialogs_dir1.append(dialog)
        index_dir += 1
    return dialogs_dir1

def get_poi2poi_dis(diags, diag_routes):
    # shuffle the order of dialogs
    combined = list(zip(diags, diag_routes))
    random.shuffle(combined)
    diags, diag_routes = zip(*combined)
    diags = list(diags)
    diag_routes = list(diag_routes)
    
    all_train_data = []
    for cnt, train in enumerate(diags):
        if cnt <= 1000:
            all_train_data.append((train, cnt))
    dis_seg1 = all_train_data[:int(len(all_train_data) * 0.125)]
    dis_seg2 = all_train_data[int(len(all_train_data) * 0.125) + 1:int(len(all_train_data) * 0.25)]
    dis_seg3 = all_train_data[int(len(all_train_data) * 0.25) + 1:int(len(all_train_data) * 0.375)]
    dis_seg4 = all_train_data[int(len(all_train_data) * 0.375) + 1:int(len(all_train_data) * 0.5)]
    dis_seg5 = all_train_data[int(len(all_train_data) * 0.5) + 1:int(len(all_train_data) * 0.625)]
    dis_seg6 = all_train_data[int(len(all_train_data) * 0.625) + 1:int(len(all_train_data) * 0.75)]
    dis_seg7 = all_train_data[int(len(all_train_data) * 0.75) + 1:int(len(all_train_data) * 0.875)]
    dis_seg8 = all_train_data[int(len(all_train_data) * 0.875) + 1:]
    ######单轮，非整，无限制
    dialogs_dis1 = []
    index = 0
    task_name = 'cityreasoning-{}'.format(REGION_EXP)
    for train, cnt in dis_seg1:
        dialog = {'task': task_name, 'id': 'poi2poi_dis' + str(index), 'diag': []}
        content = json.loads(train[1]["content"])
        start_poi_name = content['start_name']
        start_poi_addr = content['start_addr']
        start_poi = "{}{}".format(start_poi_name, start_poi_addr)
        end_poi_name = content['dest_name']
        end_poi_addr = content['dest_addr']
        end_poi = "{}{}".format(end_poi_name, end_poi_addr)
        routine = "After starting from {}{}, you can arrive at the destination {}{} by following the navigation instruction below:\n".format(start_poi_name, start_poi_addr, end_poi_name, end_poi_addr) + ", ".join(diag_routes[cnt])
        # Extracting numbers before "meters"
        distances = re.findall(r'for (\d+) meters', routine)
        total_length = np.sum([float(distance) for distance in distances])
        ques = "How many meters do I need to walk from {} to {} along the road?".format(start_poi, end_poi)
        choice_items = [total_length * 2, total_length / 2, total_length - 1000, total_length]
        random.shuffle(choice_items)
        letters = ['A', 'B', 'C', 'D']
        choices = ['{}.{}'.format(x, y) for x, y in zip(letters, choice_items)]
        distance_strs = ','.join([str(item) + 'm' for item in distances])

        st1 = "Step 1: Find the distance traveled for each segment of the road:\n" + distance_strs + '\n'
        st2 = "Step 2: Add all distances above together to get the total distance:\n" + compute_length_template(distances)[0] + ". So the answer is {}m.\n".format(compute_length_template(distances)[1])
        for cnt2, choice in enumerate(choice_items):
            if choice == total_length:
                answer = letters[cnt2]
        assistant1 = "Navigation instruction:\n" + routine
        assistant2 = "Answer:{}\n".format(answer) + st1 + st2
        dialog['diag'].append(
            {'role': 'user', 'content': routine + ques + '\n'.join(choices) + "\nLet's think step by step.\n"})
        dialog['diag'].append({'role': 'assistant', 'content': assistant2})
        dialogs_dis1.append(dialog)
        index += 1
    ######单轮，整，无限制
    for train, cnt in dis_seg2:
        dialog = {'task': task_name, 'id': 'poi2poi_dis' + str(index), 'diag': []}
        content = json.loads(train[1]["content"])
        start_poi_name = content['start_name']
        start_poi_addr = content['start_addr']
        start_poi = "{}{}".format(start_poi_name, start_poi_addr)
        end_poi_name = content['dest_name']
        end_poi_addr = content['dest_addr']
        end_poi = "{}{}".format(end_poi_name, end_poi_addr)
        routine = "After starting from {}{}, you can arrive at the destination {}{} by following the navigation instruction below:\n".format(start_poi_name, start_poi_addr, end_poi_name, end_poi_addr) + ", ".join(diag_routes[cnt])
        # Extracting numbers before "meters"
        distances = re.findall(r'for (\d+) meters', routine)
        total_length = np.sum([float(distance) for distance in distances])
        ques = "How many meters do I need to walk from {} to {} along the road?".format(start_poi, end_poi)
        choice_items = [int(total_length * 2), int(total_length / 2), int(total_length - 1000), int(total_length)]
        random.shuffle(choice_items)
        letters = ['A', 'B', 'C', 'D']
        choices = ['{}.{}'.format(x, y) for x, y in zip(letters, choice_items)]
        distance_strs = ','.join([str(item) + 'm' for item in distances])

        st1 = "Step 1: Find the distance traveled for each segment of the road:\n" + distance_strs + '\n'
        st2 = "Step 2: Add all distances above together to get the total distance:\n" + compute_length_template(distances)[
            0] + ". So the answer is {}m.\n".format(compute_length_template(distances)[1])
        for cnt2, choice in enumerate(choice_items):
            if choice == total_length:
                answer = letters[cnt2]
        assistant2 = "Answer:{}\n".format(answer) + st1 + st2
        dialog['diag'].append(
            {'role': 'user', 'content': routine + ques + '\n'.join(choices) + "\nLet's think step by step.\n"})
        dialog['diag'].append({'role': 'assistant', 'content': assistant2})
        dialogs_dis1.append(dialog)
        index += 1
    ######单轮，非整，限制
    for train, cnt in dis_seg3:
        dialog = {'task': task_name, 'id': 'poi2poi_dis' + str(index), 'diag': []}
        content = json.loads(train[1]["content"])
        start_poi_name = content['start_name']
        start_poi_addr = content['start_addr']
        start_poi = "{}{}".format(start_poi_name, start_poi_addr)
        end_poi_name = content['dest_name']
        end_poi_addr = content['dest_addr']
        end_poi = "{}{}".format(end_poi_name, end_poi_addr)
        routine = "After starting from {}{}, you can arrive at the destination {}{} by following the navigation instruction below:\n".format(start_poi_name, start_poi_addr, end_poi_name, end_poi_addr) + ", ".join(diag_routes[cnt])

        # Extracting numbers before "meters"
        distances = re.findall(r'for (\d+) meters', routine)
        total_length = np.sum([float(distance) for distance in distances])
        ques = "You only need to consider the distances walking along the roads.How many meters do I need to walk from {} to {} along the road?".format(
            start_poi, end_poi)
        choice_items = [total_length * 2, total_length / 2, total_length - 1000, total_length]
        random.shuffle(choice_items)
        letters = ['A', 'B', 'C', 'D']
        choices = ['{}.{}'.format(x, y) for x, y in zip(letters, choice_items)]
        distance_strs = ','.join([str(item) + 'm' for item in distances])

        st1 = "Step 1: Find the distance traveled for each segment of the road:\n" + distance_strs + '\n'
        st2 = "Step 2: Add all distances above together to get the total distance:\n" + compute_length_template(distances)[
            0] + ". So the answer is {}m.\n".format(compute_length_template(distances)[1])
        for cnt2, choice in enumerate(choice_items):
            if choice == total_length:
                answer = letters[cnt2]
        assistant2 = "Answer:{}\n".format(answer) + st1 + st2
        dialog['diag'].append(
            {'role': 'user', 'content': routine + ques + '\n'.join(choices) + "\nLet's think step by step.\n"})
        dialog['diag'].append({'role': 'assistant', 'content': assistant2})
        dialogs_dis1.append(dialog)
        index += 1
    ######单轮，整，限制
    for train, cnt in dis_seg4:
        dialog = {'task': task_name, 'id': 'poi2poi_dis' + str(index), 'diag': []}
        content = json.loads(train[1]["content"])
        start_poi_name = content['start_name']
        start_poi_addr = content['start_addr']
        start_poi = "{}{}".format(start_poi_name, start_poi_addr)
        end_poi_name = content['dest_name']
        end_poi_addr = content['dest_addr']
        end_poi = "{}{}".format(end_poi_name, end_poi_addr)
        routine = "After starting from {}{}, you can arrive at the destination {}{} by following the navigation instruction below:\n".format(start_poi_name, start_poi_addr, end_poi_name, end_poi_addr) + ", ".join(diag_routes[cnt])
        # Extracting numbers before "meters"
        distances = re.findall(r'for (\d+) meters', routine)
        total_length = np.sum([float(distance) for distance in distances])
        ques = "You only need to consider the distances walking along the roads. How many meters do I need to walk from {} to {} along the road?".format(
            start_poi, end_poi)
        choice_items = [int(total_length * 2), int(total_length / 2), int(total_length - 1000), int(total_length)]
        random.shuffle(choice_items)
        letters = ['A', 'B', 'C', 'D']
        choices = ['{}.{}'.format(x, y) for x, y in zip(letters, choice_items)]
        distance_strs = ','.join([str(item) + 'm' for item in distances])

        st1 = "Step 1: Find the distance traveled for each segment of the road:\n" + distance_strs + '\n'
        st2 = "Step 2: Add all distances above together to get the total distance:\n" + compute_length_template(distances)[
            0] + ". So the answer is {}m.\n".format(compute_length_template(distances)[1])
        for cnt2, choice in enumerate(choice_items):
            if choice == total_length:
                answer = letters[cnt2]
        assistant2 = "Answer:{}\n".format(answer) + st1 + st2
        dialog['diag'].append(
            {'role': 'user', 'content': routine + ques + '\n'.join(choices) + "\nLet's think step by step.\n"})
        dialog['diag'].append({'role': 'assistant', 'content': assistant2})
        dialogs_dis1.append(dialog)
        index += 1
    ######多轮，非整，无限制
    for train, cnt in dis_seg5:
        dialog = {'task': task_name, 'id': 'poi2poi_dis' + str(index - 1), 'diag': []}
        user_head = "Let's think step by step.\n"
        content = json.loads(train[1]["content"])
        start_poi_name = content['start_name']
        start_poi_addr = content['start_addr']
        start_poi = "{}{}".format(start_poi_name, start_poi_addr)
        end_poi_name = content['dest_name']
        end_poi_addr = content['dest_addr']
        end_poi = "{}{}".format(end_poi_name, end_poi_addr)
        routine = "After starting from {}{}, you can arrive at the destination {}{} by following the navigation instruction below:\n".format(start_poi_name, start_poi_addr, end_poi_name, end_poi_addr) + ", ".join(diag_routes[cnt])
        # Extracting numbers before "meters"
        distances = re.findall(r'for (\d+) meters', routine)
        total_length = np.sum([float(distance) for distance in distances])
        ques = "How many meters do I need to walk from {} to {} along the road?".format(start_poi, end_poi)
        choice_items = [total_length * 2, total_length / 2, total_length - 1000, total_length]
        random.shuffle(choice_items)
        letters = ['A', 'B', 'C', 'D']
        choices = ['{}.{}'.format(x, y) for x, y in zip(letters, choice_items)]
        user1 = ques + '\n' + '\n'.join(
            choices) + '\n' + 'To answer the question, please think about the navigation instruction about starting from {} and arriving at the destination {}'.format(
            start_poi, end_poi)
        user2 = 'Given the navigation instruction, please choose the most suitable one among A, B, C and D as the answer to above question.' + user_head
        distance_strs = ','.join([str(item) + 'm' for item in distances])

        st1 = "Step 1: Find the distance traveled for each segment of the road:\n" + distance_strs + '\n'
        st2 = "Step 2: Add all distances above together to get the total distance:\n" + compute_length_template(distances)[
            0] + ". So the answer is {}m.\n".format(compute_length_template(distances)[1])
        for cnt2, choice in enumerate(choice_items):
            if choice == total_length:
                answer = letters[cnt2]
        assistant1 = "Navigation instruction:\n" + routine
        assistant2 = "Answer:{}\n".format(answer) + st1 + st2
        dialog['diag'].append({'role': 'user', 'content': user1})
        dialog['diag'].append({'role': 'assistant', 'content': assistant1})
        dialog['diag'].append({'role': 'user', 'content': user2})
        dialog['diag'].append({'role': 'assistant', 'content': assistant2})
        dialogs_dis1.append(dialog)
        index += 1
    ######多轮，整，无限制
    for train, cnt in dis_seg6:
        dialog = {'task': task_name, 'id': 'poi2poi_dis' + str(index - 1), 'diag': []}
        user_head = "Let's think step by step.\n"
        content = json.loads(train[1]["content"])
        start_poi_name = content['start_name']
        start_poi_addr = content['start_addr']
        start_poi = "{}{}".format(start_poi_name, start_poi_addr)
        end_poi_name = content['dest_name']
        end_poi_addr = content['dest_addr']
        end_poi = "{}{}".format(end_poi_name, end_poi_addr)
        routine = "After starting from {}{}, you can arrive at the destination {}{} by following the navigation instruction below:\n".format(start_poi_name, start_poi_addr, end_poi_name, end_poi_addr) + ", ".join(diag_routes[cnt])
        # Extracting numbers before "meters"
        distances = re.findall(r'for (\d+) meters', routine)
        total_length = np.sum([float(distance) for distance in distances])
        ques = "How many meters do I need to walk from {} to {} along the road?".format(start_poi, end_poi)
        choice_items = [int(total_length * 2), int(total_length / 2), int(total_length - 1000), int(total_length)]
        random.shuffle(choice_items)
        letters = ['A', 'B', 'C', 'D']
        choices = ['{}.{}'.format(x, y) for x, y in zip(letters, choice_items)]
        user1 = ques + '\n' + '\n'.join(
            choices) + '\n' + 'To answer the question, please think about the navigation instruction about starting from {} and arriving at the destination {}'.format(
            start_poi, end_poi)
        user2 = 'Given the navigation instruction,please choose the most suitable one among A, B, C and D as the answer to above question.' + user_head
        distance_strs = ','.join([str(item) + 'm' for item in distances])

        st1 = "Step 1: Find the distance traveled for each segment of the road:\n" + distance_strs + '\n'
        st2 = "Step 2: Add all distances above together to get the total distance:\n" + compute_length_template(distances)[0] + ". So the answer is {}m.\n".format(compute_length_template(distances)[1])
        for cnt2, choice in enumerate(choice_items):
            if choice == total_length:
                answer = letters[cnt2]
        assistant1 = "Navigation instruction:\n" + routine
        assistant2 = "Answer:{}\n".format(answer) + st1 + st2
        dialog['diag'].append({'role': 'user', 'content': user1})
        dialog['diag'].append({'role': 'assistant', 'content': assistant1})
        dialog['diag'].append({'role': 'user', 'content': user2})
        dialog['diag'].append({'role': 'assistant', 'content': assistant2})
        dialogs_dis1.append(dialog)
        index += 1
    ######多轮，非整，限制
    for train, cnt in dis_seg7:
        dialog = {'task': task_name, 'id': 'poi2poi_dis' + str(index - 1), 'diag': []}
        user_head = "Let's think step by step.\n"
        content = json.loads(train[1]["content"])
        start_poi_name = content['start_name']
        start_poi_addr = content['start_addr']
        start_poi = "{}{}".format(start_poi_name, start_poi_addr)
        end_poi_name = content['dest_name']
        end_poi_addr = content['dest_addr']
        end_poi = "{}{}".format(end_poi_name, end_poi_addr)
        routine = "After starting from {}{}, you can arrive at the destination {}{} by following the navigation instruction below:\n".format(start_poi_name, start_poi_addr, end_poi_name, end_poi_addr) + ", ".join(diag_routes[cnt])
        # Extracting numbers before "meters"
        distances = re.findall(r'for (\d+) meters', routine)
        total_length = np.sum([float(distance) for distance in distances])
        ques = "You only need to consider the distances walking along the roads.How many meters do I need to walk from {} to {} along the road?".format(
            start_poi, end_poi)
        choice_items = [total_length * 2, total_length / 2, total_length - 1000, total_length]
        random.shuffle(choice_items)
        letters = ['A', 'B', 'C', 'D']
        choices = ['{}.{}'.format(x, y) for x, y in zip(letters, choice_items)]
        user1 = ques + '\n' + '\n'.join(
            choices) + '\n' + 'To answer the question, please think about the navigation instruction about starting from {} and arriving at the destination {}'.format(
            start_poi, end_poi)
        user2 = 'Given the navigation instruction,please choose the most suitable one among A, B, C and D as the answer to above question.' + user_head
        distance_strs = ','.join([str(item) + 'm' for item in distances])

        st1 = "Step 1: Find the distance traveled for each segment of the road:\n" + distance_strs + '\n'
        st2 = "Step 2: Add all distances above together to get the total distance:\n" + compute_length_template(distances)[
            0] + ". So the answer is {}m.\n".format(compute_length_template(distances)[1])
        for cnt2, choice in enumerate(choice_items):
            if choice == total_length:
                answer = letters[cnt2]
        assistant1 = "Navigation instruction:\n" + routine
        assistant2 = "Answer:{}\n".format(answer) + st1 + st2
        dialog['diag'].append({'role': 'user', 'content': user1})
        dialog['diag'].append({'role': 'assistant', 'content': assistant1})
        dialog['diag'].append({'role': 'user', 'content': user2})
        dialog['diag'].append({'role': 'assistant', 'content': assistant2})
        dialogs_dis1.append(dialog)
        index += 1
    ######多轮，整，限制
    for train, cnt in dis_seg8:
        dialog = {'task': task_name, 'id': 'poi2poi_dis' + str(index - 1), 'diag': []}
        user_head = "Let's think step by step.\n"
        content = json.loads(train[1]["content"])
        start_poi_name = content['start_name']
        start_poi_addr = content['start_addr']
        start_poi = "{}{}".format(start_poi_name, start_poi_addr)
        end_poi_name = content['dest_name']
        end_poi_addr = content['dest_addr']
        end_poi = "{}{}".format(end_poi_name, end_poi_addr)
        routine = "After starting from {}{}, you can arrive at the destination {}{} by following the navigation instruction below:\n".format(start_poi_name, start_poi_addr, end_poi_name, end_poi_addr) + ", ".join(diag_routes[cnt])
        # Extracting numbers before "meters"
        distances = re.findall(r'for (\d+) meters', routine)
        total_length = np.sum([float(distance) for distance in distances])
        ques = "You only need to consider the distances walking along the roads. How many meters do I need to walk from {} to {} along the road?".format(
            start_poi, end_poi)
        choice_items = [int(total_length * 2), int(total_length / 2), int(total_length - 1000), int(total_length)]
        random.shuffle(choice_items)
        letters = ['A', 'B', 'C', 'D']
        choices = ['{}.{}'.format(x, y) for x, y in zip(letters, choice_items)]
        user1 = ques + '\n' + '\n'.join(
            choices) + '\n' + 'To answer the question, please think about the navigation instruction about starting from {} and arriving at the destination {}'.format(
            start_poi, end_poi)
        user2 = 'Given the navigation instruction, please choose the most suitable one among A, B, C and D as the answer to above question.' + user_head
        distance_strs = ','.join([str(item) + 'm' for item in distances])

        st1 = "Step 1: Find the distance traveled for each segment of the road:\n" + distance_strs + '\n'
        st2 = "Step 2: Add all distances above together to get the total distance:\n" + compute_length_template(distances)[
            0] + ". So the answer is {}m.\n".format(compute_length_template(distances)[1])
        for cnt2, choice in enumerate(choice_items):
            if choice == total_length:
                answer = letters[cnt2]
        assistant1 = "Navigation instruction:\n" + routine
        assistant2 = "Answer:{}\n".format(answer) + st1 + st2
        dialog['diag'].append({'role': 'user', 'content': user1})
        dialog['diag'].append({'role': 'assistant', 'content': assistant1})
        dialog['diag'].append({'role': 'user', 'content': user2})
        dialog['diag'].append({'role': 'assistant', 'content': assistant2})
        dialogs_dis1.append(dialog)
        index += 1
    return dialogs_dis1

def main(args):
    city_map = MAP_DICT[REGION_EXP]
    port = args.port if args.port is not None else MAP_PORT_DICT[REGION_EXP]
    print(f"Loading map {city_map} on port {port}")
    map, process, routing_client = load_map(
        city_map=city_map, 
        cache_dir=MAP_CACHE_PATH, 
        routing_path=ROUTING_PATH, 
        port=port)
    # wait for map to load
    time.sleep(15)
    if not isinstance(args.output_path, str):
        output_path = "simulate/examples/"
    else:
        output_path = args.output_path

    random.seed(42)
   
    poi_message = pd.read_csv(os.path.join(RESOURCE_PATH, "{}_pois.csv".format(REGION_EXP)))
    aoi_message = pd.read_csv(os.path.join(RESOURCE_PATH, "{}_aois.csv".format(REGION_EXP)))

    poi_dict = {}
    for row in poi_message.itertuples():
        key = row.poi_id
        category = map.get_poi(key)['category']
        category = category.split(" > ")[0] if " > " in category else category
        name = row.name
        addr = row.Address
        if not isinstance(name, str):
            continue
        poi_dict[key] = {
            "aoi_id": map.get_poi(key)['aoi_id'], "category": category, "name": name, "Address": addr,
            "coord": map.get_poi(key)['shapely_lnglat'].coords[0],
        }
    print(f"define poi_dict done! len:{len(poi_dict)}")

    aoi_dict = {} 
    # filter "nearby" aoi
    for row in aoi_message.itertuples():
        aoi_id = row.aoi_id
        aoi_name = row.aoi_name
        address = row.Address
        extra = "nearby"
        poi_ids = map.get_aoi(aoi_id)['poi_ids']
        if not isinstance(aoi_name, str) or extra in aoi_name:
            continue
        if not isinstance(address, str):
            continue
        aoi_dict[aoi_id] = {}
        aoi_dict[aoi_id]['category'] = map.get_aoi(aoi_id)['urban_land_use']
        aoi_dict[aoi_id]['coord'] = eval(row.coords)[0]
        aoi_dict[aoi_id]['name'] = aoi_name
        aoi_dict[aoi_id]['address'] = address
        aoi_dict[aoi_id]['poi_ids'] = poi_ids
    print(f"define aoi_dict done! len:{len(aoi_dict)}")

    type_pois = {}  

    for poi_id, poi in poi_dict.items():
        poi_type = poi["category"]
        if poi_type in type_pois:
            type_pois[poi_type].append(poi_id)
        else:
            type_pois[poi_type] = [poi_id]
    
    train_data = []
    # EVAL_DATA=True
    # 输入结构化导航信息
    with open(os.path.join(TRAIN_DATA_PATH, "citywalk-{}-mock-{}.jsonl".format(REGION_EXP, DATA_VERSION)),
              'r',
              encoding="utf-8") as f:
        for line in f:
            train = json.loads(line)
            train_data.append(train)
    diags = [train_data[i]['diag'] for i in range(len(train_data))]


    name_adr_poi = {}
    for id, poi in poi_dict.items():
        name = poi['name']
        addr = poi['Address']
        flag = "{}({})".format(name, addr)
        if flag not in name_adr_poi:
            name_adr_poi[flag] = poi

    ########训练数据提取#######
    diag_records = []
    diag_routes = []
    for cnt, diag in enumerate(diags):
        diag_route = []
        content = json.loads(diag[1]["content"])
        routes = content['routes']
        if len(routes) == 0:
            # print("No route in diag")
            # print(diag)
            continue
        steps = (len(routes) + 1) / 2
        if steps > STEP:
            continue
        # 将结构化导航信息转换为文本描述
        for route in routes:
            if route["type"] == "road":
                diag_route.append(
                    f"walk along {route['road_name']} for {route['road_length']} meters {route['direction']}"
                )
            elif route["type"] == "junc":
                diag_route.append(
                    f"then go {route['direction']} towards {route['junc_name']}"
                )
        diag_records.append(diag)
        diag_routes.append(diag_route)
    # print(f"Length of diag_records: {len(diag_records)}")
    # print(f"Length of diag_routes: {len(diag_routes)}")
    assert len(diag_records) == len(diag_routes), f"Mismatch: diag_records has {len(diag_records)} items, while diag_routes has {len(diag_routes)} items."

    print("Start to generate data for CityReasoning!")
    poi2rd_dir = get_poi2rd_dir(map, diag_records, diag_routes, name_adr_poi)
    poi2rd_dis = get_poi2rd_dis(map, diag_records, diag_routes, name_adr_poi)
    aoi2rd_dir = get_aoi2rd_dir(map, diag_records, diag_routes, aoi_dict, name_adr_poi)
    aoi2rd_dis = get_aoi2rd_dis(map, diag_records, diag_routes, aoi_dict, name_adr_poi)
    poi2aoi_dir, poi2aoi_dis = get_poi2aoi(diag_records, diag_routes, aoi_dict, name_adr_poi)
    poi2poi_dir = get_poi2poi_dir(diag_records, diag_routes)
    poi2poi_dis = get_poi2poi_dis(diag_records, diag_routes)
    print("Data generation done!")
    all_data = poi2rd_dir + poi2rd_dis + aoi2rd_dir + aoi2rd_dis + poi2aoi_dir + poi2aoi_dis + poi2poi_dir + poi2poi_dis

    # 输出路径
    output_file = os.path.join(output_path, "cityreasoning_{}_{}.jsonl".format(REGION_EXP, DATA_VERSION))
    with open(output_file, 'w', encoding='utf-8') as f:
        for data in all_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--port", type=str)
    args = parser.parse_args()
    main(args)
