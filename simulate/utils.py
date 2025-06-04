import os
import re
import enum
import argparse
import math
import signal
import subprocess

import numpy as np
import pandas as pd
from typing import List
from shapely import Polygon, Point
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from pycitydata.map import Map
from citysim.routing import RoutingClient

from config import REGION_EXP, RESOURCE_PATH, MONGODB_URI, MAP_DICT, MAP_PORT_DICT, MAP_CACHE_PATH, ROUTING_PATH, REGION_BOUNDARY

class Action(str, enum.Enum):
    MOVE_IN_DOOR = "move_to"
    LOOK_IN_DOOR = "look_around"
    EXPLORE = "explore_in_building"
    SEARCH = "search_via_map"
    NAVIGATE = "navigate_to"
    WALK = "walk_to"
    DRIVE = "drive_to"
    BUY = "buy"
    EAT = "eat"
    DRINK = "drink"
    WORK = "work_in"


class RunMode(str, enum.Enum):
    CITY_WALK = "citywalk"
    NORMAL = "normal"


class NavigateStatus(str, enum.Enum):
    NO_ROUTE = "no available routings"
    DES_NONE = "destination AOI is None"
    ENDPOINT = "move to the destination {}"
    ONE_STEP = "move {} meters along {} {}"
    NAV_FAIL = "final move to destination failed."


# 获取全部的AOI和POI信息
def find_all_pois(city_map: Map):
    
    region_pois = []

    for poi in list(city_map.pois.items()):
        poi_id = poi[0]
        poi_info = poi[1]
        region_pois.append([poi_id, poi_info["category"], poi_info["name"]])
    
    data = pd.DataFrame(data=region_pois, columns=["poi_id", "category", "name"])
    data.to_csv(os.path.join(RESOURCE_PATH, "{}_pois.csv".format("all")))


def find_all_aois(city_map: Map):
    
    region_aois = []
    for aoi in list(city_map.aois.items()):
        aoi_id = aoi[0]
        aoi_info = aoi[1]
        centroid = aoi_info["shapely_lnglat"].centroid
        land_use = aoi_info["urban_land_use"] if "urban_land_use" in aoi_info else -1
        coords = list(centroid.coords)
        aoi_name = aoi_info["name"]
        if "nearby" in aoi_name or aoi_name == "":
            continue
        region_aois.append([aoi_id, str(aoi_name), land_use, coords])
    
    data = pd.DataFrame(data=region_aois, columns=["aoi_id", "aoi_name", "land_use", "coords"])
    data.to_csv(os.path.join(RESOURCE_PATH, "{}_aois.csv".format("all")))

def bleu_score(reference, candidate):
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()

    smoothie = SmoothingFunction().method4
    score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)
    return score

# TODO 从文本中抽取action，与get_available_actions相配合
def process_action(action, choices, limit=0.01, to_print=False):
    if to_print:
        print("preprocess action: ", action)
    match = re.search("ACTION:(.*)", action)
    if match:
        action = match.group(1)
    else:
        return False

    action = action.strip().lower().split("\n")[0]
    if not choices:
        return action
    if action in choices:
        return action
    try:
        bleus = [bleu_score(choice, action) for choice in choices]
        bleus_np = np.array(bleus)
        max_index = np.argmax(bleus_np)
        max_score = bleus[max_index]
        
        # 最大值可能不止一个，取重合度最高的一个
        idx = np.where(bleus_np==max_score)[0]
        if len(idx)>1:
            lens = []
            for i in idx:
                ilen = len(set(choices[i]).intersection(set(action)))
                lens.append(ilen)
            max_idx = np.argmax(lens)
            max_index = idx[max_idx]

        if max_score > limit:
            if to_print:
                print("processed action: ", choices[max_index], " score: ", max_score)
            return choices[max_index]
    except Exception as e:
        print("encounter exception: ", e)
        print("choices: ", choices)
        print("action: ", action)
    
    # 保证不会产生不被允许的action
    return False

# 简化observation中的POI信息，便于评估任务构造，与player.py中的observation构造过程强绑定
def simplify_observation_pois(city_map, observation, detail_interest=False):
    if "surroundings" not in observation["observations"]:
        return []
    if detail_interest:
        interests_info = observation["observations"]["surroundings"]

        simple_interests = {}
        for key in interests_info:
            if len(interests_info[key])==0:
                continue

            simple_interests[key] = [city_map.pois[poi["id"]]['name'] for poi in interests_info[key]]

        return simple_interests
    else:
        return []
    
def check_in_region(region_exp_polygon, shapely_point: Point):
    # 确认关键元素POI/AOI/Road是否在实验区域内
    return region_exp_polygon.contains(shapely_point)


# 事先提取落在实验区域内的POI数据
def find_pois_in_region_exp(city_map: Map):
    region_pois = []

    for poi in list(city_map.pois.items()):
        poi_id = poi[0]
        poi_info = poi[1]
        if poi_info["name"] == "":
            continue
        if check_in_region(Polygon(REGION_BOUNDARY[REGION_EXP]), poi_info["shapely_lnglat"]):
            region_pois.append([poi_id, poi_info["category"], str(poi_info["name"])])
    
    data = pd.DataFrame(data=region_pois, columns=["poi_id", "category", "name"])
    data.to_csv(os.path.join(RESOURCE_PATH, "{}_pois.csv".format(REGION_EXP)), index=False)


# 事先提取落在实验区域内的AOI数据
def find_aois_in_region_exp(city_map: Map):
    region_aois = []

    for aoi in list(city_map.aois.items()):
        aoi_id = aoi[0]
        aoi_info = aoi[1]
        aoi_name = aoi_info["name"]
        if "nearby" in aoi_name or aoi_name == "":
            continue
        centroid = aoi_info["shapely_lnglat"].centroid
        land_use = aoi_info["urban_land_use"] if "urban_land_use" in aoi_info else -1
        coords = list(centroid.coords)
        if check_in_region(Polygon(REGION_BOUNDARY[REGION_EXP]), centroid):
            region_aois.append([aoi_id, str(aoi_name), land_use, coords])
    
    columns = ["aoi_id", "aoi_name", "land_use", "coords"]
    data = pd.DataFrame(data=region_aois, columns=columns)
    data.to_csv(os.path.join(RESOURCE_PATH, "{}_aois.csv".format(REGION_EXP)), index=False)


def find_roads_in_region_exp(city_map: Map):
    all_roads = city_map.roads
    all_lanes = city_map.lanes

    region_exp_dict = REGION_BOUNDARY
    REGION_EXP_POLYGON = Polygon(region_exp_dict[REGION_EXP])

    cared_roads = []
    for road_id in all_roads:
        lane_ids = all_roads[road_id]["lane_ids"]
        road_name = all_roads[road_id]["name"]
        if road_name == "":
            continue
        in_polygon_flag = False
        for i, lane_id in enumerate(lane_ids):
            if type(lane_id) == list:
                print("lane_id in lane_ids is List!!!!!")
                continue

            lane = all_lanes[lane_id]
            length = lane["length"]
            last_point = Point(lane["shapely_lnglat"].coords[-1])
            if REGION_EXP_POLYGON.contains(last_point):
                in_polygon_flag = True
                break
        if in_polygon_flag:
            cared_roads.append([road_id, road_name])
    
    columns = ["road_id", "road_name"]
    cared_roads_df = pd.DataFrame(data=cared_roads, columns=columns)
    cared_roads_df.to_csv(os.path.join(RESOURCE_PATH, "{}_roads.csv".format(REGION_EXP)))


def cal_angle(start_point, end_point):
        """
        方位角计算
        """
        return (round(90 - math.degrees(math.atan2(end_point.y - start_point.y, end_point.x - start_point.x)), 2)) % 360

def angle2dir_4(angle):
    """
    将方位角离散成4个基本方向
    """
    Direction = ['north', 'east', 'south', 'west']
    
    if angle < 45 or angle >= 315:
        return Direction[0]  
    elif 45 <= angle < 135:
        return Direction[1]  
    elif 135 <= angle < 225:
        return Direction[2]  
    else:  
        return Direction[3]  
    
def angle2dir(angle):
    """
    将方位角离散成8个基本方向
    """
    Direction = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']

    s = 22.5
    for i in range(8):
        if angle < s + 45 * i:
            return Direction[i]
    return Direction[0]

def direction_description(angle):
    basic_direction = angle2dir(angle)
    direction_mapping = {
        'north': 'from south to north',
        'northeast': 'from southwest to northeast',
        'east': 'from west to east',
        'southeast': 'from northwest to southeast',
        'south': 'from north to south',
        'southwest': 'from northeast to southwest',
        'west': 'from east to west',
        'northwest': 'from southeast to northwest',
    }

    return basic_direction, direction_mapping.get(basic_direction, "unknown direction")

def lnglat2grid(center, coords):
    resolution = 0.0001
    x = int((coords[0]-center[0])/resolution)
    y = int((coords[1]-center[1])/resolution)

    return (x, y)

# #######对POI距离推理训练数据进行泛化,单/多轮，是/否整数，有/无限制
def compute_length_template(distances):
    head = "First add two of the numbers:"
    mid = "Then add the sum with another number:"
    tail = "Then add the sum with the last number:"
    total_length = np.sum([int(distance) for distance in distances])
    sums = []
    distance_diags = []
    for cnt, distance in enumerate(distances):
        if len(distances) == 1:
            distance_diag = "The total distance is {}.".format(distances[0])
            sum_item = distances[0]
        elif len(distances) == 2:
            distance_diag = "The total distance is {}.".format(distances[0]+distances[1])
            sum_item = distances[0]+distances[1]
        elif len(distances) > 2:
            if cnt == 0:
                sum_item = int(distances[cnt]) + int(distances[cnt + 1])
                num_str = str(distances[cnt]) + '+' + str(distances[cnt + 1]) + '=' + str(sum_item)
                distance_diag = head + num_str

            elif cnt == len(distances) - 2:
                sum_item = sums[cnt - 1] + int(distances[cnt + 1])
                num_str = str(sums[cnt - 1]) + '+' + str(distances[cnt + 1]) + '=' + str(sum_item)
                distance_diag = tail + num_str
            elif cnt == len(distances) - 1:
                continue
            else:
                sum_item = sums[cnt - 1] + int(distances[cnt + 1])
                num_str = str(sums[cnt - 1]) + '+' + str(distances[cnt + 1]) + '=' + str(sum_item)
                distance_diag = mid + num_str
        sums.append(sum_item)
        distance_diags.append(distance_diag)
    result = '\n'.join([distance_diag for distance_diag in distance_diags]) + '.' + "+".join(
        [distance for distance in distances]) + "=" + str(total_length)
    return result, sums[-1]


def load_map(city_map, cache_dir, routing_path, port):
    m = Map(
            mongo_uri=f"{MONGODB_URI}",
            mongo_db="srt",
            mongo_coll=city_map,
            cache_dir=cache_dir,
        )
    route_command = f"{routing_path} -mongo_uri {MONGODB_URI} -map srt.{city_map} -cache {cache_dir} -listen localhost:{port}"
    cmd = route_command.split(" ")
    print("loading routing service")
    process = subprocess.Popen(args=cmd, cwd="./")
    routing_client = RoutingClient(f"localhost:{port}")
    return m, process, routing_client


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    args = parser.parse_args()

    city_map = MAP_DICT[REGION_EXP]
    port = args.port if args.port is not None else MAP_PORT_DICT[REGION_EXP]
    map, process, routing_client = load_map(
        city_map=city_map, 
        cache_dir=MAP_CACHE_PATH, 
        routing_path=ROUTING_PATH, 
        port=port)

    find_roads_in_region_exp(map)
    find_pois_in_region_exp(map)
    find_aois_in_region_exp(map)
    # find_all_aois(map)

    print("send signal")
    process.send_signal(sig=signal.SIGTERM)
    process.wait()
    

