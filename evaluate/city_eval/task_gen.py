import argparse
import asyncio
import os
import random
import shapely
import signal
import json
import time
from itertools import chain
import numpy as np
import pandas as pd
from shapely import Polygon, Point
from geopy.distance import geodesic

from pycityproto.city.geo.v2.geo_pb2 import AoiPosition, Position, XYPosition, LongLatPosition

from .utils import task_files_adaption, gen_options, save_data, load_map, angle2dir_4, calcu_azimuth, compute_length, dir_all_dis, secondary_dir_to_primary_dirs, NS, EW, primary_directions, secondary_directions
from simulate.player import TextPlayer, category_mapping
from simulate.address_system import get_aoi_address, get_center_point
from simulate.templates import task_template
from config import *

def generate_evaluation_task_road(map, interest_dict, road_dict, all_roads, all_lanes, all_aois, TASK_FILES):
    if OSM_REGION == False:
        pois_ids_set = {poi_id for poi_id in interest_dict if map.get_poi(poi_id)['name']}
    else:
        aoi_ids_set = set(interest_dict.keys())
    cared_roads = {}
    cared_roads_name = {}

    for road_id, road in road_dict.items():
        lane_ids = road["lane_ids"]
        road_name = road["name"]
        if road_name == '':
            continue

        road_length = 0
        road_aoi_ids = []
        near_by_lanes = []
        one_lane_in_road = []
        road_level_count = []
        for i, lane_id in enumerate(lane_ids):
            if isinstance(lane_id, list):
                print("lane_id in lane_ids is List!!!!!")
                continue

            lane = all_lanes[lane_id]
            length = lane["length"]
            if OSM_REGION == False:
                aoi_ids = lane["aoi_ids"]
                road_aoi_ids.extend(aoi_ids)
            
            # using the lane information under the road to estimate the road length
            if len(one_lane_in_road) == 0:
                one_lane_in_road.append(lane_id)
                road_length += length

                left_lanes = lane["left_lane_ids"]
                right_lanes = lane["right_lane_ids"]
                near_by_lanes.extend(left_lanes)
                near_by_lanes.extend(right_lanes)
                road_level_count.append(len(left_lanes) + len(right_lanes))
            else:
                if lane_id not in near_by_lanes:
                    one_lane_in_road.append(lane_id)
                    road_length += length

                    left_lanes = lane["left_lane_ids"]
                    right_lanes = lane["right_lane_ids"]
                    near_by_lanes.extend(left_lanes)
                    near_by_lanes.extend(right_lanes)
                    road_level_count.append(len(left_lanes) + len(right_lanes))


        cared_roads[road_id] = {"lane_ids": lane_ids, "name": road_name, "length": road_length, "aoi_ids": road_aoi_ids}
        if road_name in cared_roads_name:
            cared_roads_name[road_name]["lane_ids"].append(lane_ids)
            cared_roads_name[road_name]["road_ids"].append(road_id)
            cared_roads_name[road_name]["length"].append(road_length)
            if OSM_REGION == False:
                cared_roads_name[road_name]["aoi_ids"].append(road_aoi_ids)
        else:
            if OSM_REGION == False:
                cared_roads_name[road_name] = {"road_ids": [road_id], "lane_ids": [lane_ids], "length": [road_length],
                                            "aoi_ids": [road_aoi_ids],
                                            "level_count": road_level_count}
            else:
                cared_roads_name[road_name] = {"road_ids": [road_id], "lane_ids": [lane_ids], "length": [road_length],
                                           "level_count": road_level_count}
                
    for road_name in cared_roads_name:
        road_cut = max(cared_roads_name[road_name]["level_count"][0] + 1, 1)
        cared_roads_name[road_name]["road_length_estimate"] = float(
            np.sum(cared_roads_name[road_name]["length"]) / road_cut)

    res_length = []
    for road_name in cared_roads_name:
        road_length = int(cared_roads_name[road_name]["road_length_estimate"] / 100) * 100

        res = [max(road_length * 0.1, 100), max(road_length * 0.3, 100), road_length, road_length * 2, road_length * 3]
        question = "How long is {} road?".format(road_name)
        answer = road_length
        res_dict = gen_options(res, question, answer)
        res_length.append(res_dict)

    task_df = pd.DataFrame(data=res_length)
    task_df["is_seen"] = False
    task_df.to_csv(TASK_FILES["city_image"]["road_length"])
    print("city_image road_length task OK!")
    if OSM_REGION == False:
        for road_name in cared_roads_name:
            road_aoi_ids = cared_roads_name[road_name]["aoi_ids"]
            road_poi_ids = []
            for aoi_ids in road_aoi_ids:
                for aoi_id in aoi_ids:
                    road_poi_ids.extend(all_aois[aoi_id]["poi_ids"])
            road_poi_ids = list(set(road_poi_ids))

            final_poi_ids = []
            for rpi in road_poi_ids:
                if rpi in pois_ids_set:
                    final_poi_ids.append(rpi)

            filter_final_poi_ids = []
            for p in final_poi_ids:
                poi_name = map.get_poi(p)['name']
                if poi_name != '':
                    filter_final_poi_ids.append(p)

            cared_roads_name[road_name]["arrived_pois"] = filter_final_poi_ids
        res_arrived_pois = []
        for road_name in cared_roads_name:
            res_dict = {}
            arrived_pois = cared_roads_name[road_name]["arrived_pois"]
            if len(arrived_pois) > 15:
                question_type = "FindCannot"
            else:
                question_type = "FindCan"

            if question_type == "FindCannot":
                res_temp = random.sample(arrived_pois, 15)
                res_temp_name = [map.get_poi(p)['name'] for p in res_temp]
                not_temp = random.sample(list(pois_ids_set.difference(set(arrived_pois))), 5)
                not_temp_name = [map.get_poi(p)['name'] for p in not_temp]
                res = [res_temp_name[:5], res_temp_name[5:10], res_temp_name[10:], not_temp_name]
                question = "Which POIs cannot be directly accessed via {}?".format(road_name)
                answer = not_temp_name
                res_dict = gen_options(res, question, answer)
            else:
                try:
                    res_temp = random.sample(arrived_pois, 5)
                except ValueError as e:
                    continue
                res_temp_name = [map.get_poi(p)['name'] for p in res_temp]
                not_temp = random.sample(list(pois_ids_set.difference(set(arrived_pois))), 15)
                not_temp_name = [map.get_poi(p)['name'] for p in not_temp]
                res = [not_temp_name[:5], not_temp_name[5:10], not_temp_name[10:], res_temp_name]
                answer = res_temp_name
                question = "Which POIs can be directly accessed via {}?".format(road_name)
                res_dict = gen_options(res, question, answer)
            res_arrived_pois.append(res_dict)
    else:
        res_arrived_aois = []
        for rd_id, road in road_dict.items():
            road_name = road['name']
            if road_name == '':
                continue
            arrived_aois = set(road_dict[rd_id]['aoi_ids']) & set(interest_dict.keys())
            if len(arrived_aois) == 0:
                continue
            if len(arrived_aois) > 15:
                question_type = "FindCannot"
            else:
                question_type = "FindCan"

            if question_type == "FindCannot":
                res_temp = random.sample(list(arrived_aois), 15)
                res_temp_name = [interest_dict[aoi_id] for aoi_id in res_temp if aoi_id in interest_dict]
                not_temp = random.sample(list(aoi_ids_set.difference(set(arrived_aois))), 5)
                not_temp_name = [interest_dict[aoi_id] for aoi_id in not_temp if aoi_id in interest_dict]
                res = [res_temp_name[:5], res_temp_name[5:10], res_temp_name[10:], not_temp_name]

                res_q = ["A", "B", "C", "D"]
                random.shuffle(res_q)
                label = res_q[-1]
                res_dict = dict(zip(res_q, res))
                res_dict["question"] = "Which AOIs cannot be directly accessed via {}?".format(road_name)
                res_dict["answer"] = label
            else:
                try:
                    res_temp = random.sample(list(arrived_aois), 5)
                except ValueError as e:
                    continue
                res_temp_name = [interest_dict[aoi_id] for aoi_id in res_temp]
                not_temp = random.sample(list(aoi_ids_set.difference(set(arrived_aois))), 15)
                not_temp_name = [interest_dict[aoi_id] for aoi_id in not_temp]
                res = [not_temp_name[:5], not_temp_name[5:10], not_temp_name[10:], res_temp_name]

                res_q = ["A", "B", "C", "D"]
                random.shuffle(res_q)
                label = res_q[-1]
                res_dict = dict(zip(res_q, res))
                res_dict["answer"] = label
                res_dict["question"] = "Which AOIs can be directly accessed via {}?".format(road_name)

            res_arrived_aois.append(res_dict)
    if OSM_REGION == False:
        task_df = pd.DataFrame(data=res_arrived_pois)
    else:
        task_df = pd.DataFrame(data=res_arrived_aois)
    task_df["is_seen"] = False
    if OSM_REGION == False:
        task_df.to_csv(TASK_FILES["city_image"]["road_arrived_pois"])
    else:
        task_df.to_csv(TASK_FILES["city_image"]["road_aoi"])
    print("city_image road_arrived_pois/aois task OK!")


def generate_evaluation_task_road_junc(all_juncs, all_roads, all_lanes, REGION_EXP_POLYGON, TASK_FILES):

    cared_juncs = {}
    cared_roads_name = []

    for road_id, road in all_roads.items():
        lane_ids = road["lane_ids"]
        if OSM_REGION == True:
            road_name = road['name']
        else:
            road_name = road['external']["name"]
        if road_name == '':
            continue

        road_in_region = False
        for i, lane_id in enumerate(lane_ids):
            if isinstance(lane_id, list):
                print("lane_id in lane_ids is List!!!!!")
                continue

            lane = all_lanes[lane_id]
            length = lane["length"]
            last_point = Point(lane["shapely_lnglat"].coords[-1])

            if REGION_EXP_POLYGON.contains(last_point):
                road_in_region = True
                break

        if road_in_region:
            if road_name not in cared_roads_name:
                cared_roads_name.append(road_name)

    # get basic information of junctions and roads
    for junc_id in all_juncs:
        road_in_region = False
        lane_ids = all_juncs[junc_id]["lane_ids"]

        pre_road_names = []
        suc_road_names = []
        pre_road_ids = []
        suc_road_ids = []
        for lane_id in lane_ids:
            lane = all_lanes[lane_id]
            pre_lane_id = lane["predecessors"][0]["id"]
            suc_lane_id = lane["successors"][0]["id"]
            pre_lane = all_lanes[pre_lane_id]
            suc_lane = all_lanes[suc_lane_id]

            last_point = Point(lane["shapely_lnglat"].coords[-1])

            pre_road_id = pre_lane["parent_id"]
            if pre_road_id in all_roads:
                if OSM_REGION == True:
                    pre_road_name = all_roads[pre_road_id]["name"]
                else:
                    pre_road_name = all_roads[pre_road_id]["external"]["name"]
                if pre_road_name == "":
                    continue
                pre_road_names.append(pre_road_name)
                pre_road_ids.append(pre_road_id)
            else:
                continue

            suc_road_id = suc_lane["parent_id"]
            if suc_road_id in all_roads:
                if OSM_REGION == True:
                    suc_road_name = all_roads[suc_road_id]['name']
                else:
                    suc_road_name = all_roads[suc_road_id]['external']["name"]
                if suc_road_name == "":
                    continue
                suc_road_names.append(suc_road_name)
                suc_road_ids.append(suc_road_id)
            else:
                continue

            if REGION_EXP_POLYGON.contains(last_point):
                road_in_region = True

        if road_in_region:
            pre_road_name = set(pre_road_names)
            suc_road_name = set(suc_road_names)
            pre_road_id = set(pre_road_ids)
            suc_road_id = set(suc_road_ids)
            cared_juncs[junc_id] = {"pre_road_id": pre_road_id, "suc_road_id": suc_road_id,
                                    "pre_road_name": pre_road_name, "suc_road_name": suc_road_name, "gps": last_point}

    for junc in cared_juncs:
        road_list = list(cared_juncs[junc]["pre_road_name"])
        if len(road_list) <= 1:
            junc_name = ""
        elif len(road_list) == 2:
            junc_name = "the junction of {} and {}".format(road_list[0], road_list[1])
        elif len(road_list) == 3:
            junc_name = "the junction of {}, {} and {}".format(road_list[0], road_list[1], road_list[2])
        cared_juncs[junc]["name"] = junc_name

    road_junc_gen(cared_juncs, cared_roads_name, TASK_FILES)
    road_linkage_gen(cared_juncs, TASK_FILES)


def road_junc_gen(cared_juncs, cared_roads_name, TASK_FILES):
    road_juncs = {}
    for junc in cared_juncs:
        road_list = cared_juncs[junc]["pre_road_name"]
        junc_name = cared_juncs[junc]["name"]
        junc_coord = cared_juncs[junc]["gps"]
        if len(road_list) <= 1:
            continue

        for road in list(road_list):
            if road not in road_juncs:
                road_juncs[road] = {"detail": [(junc_name, junc_coord)]}
            else:
                road_juncs[road]["detail"].append((junc_name, junc_coord))

    for rj in road_juncs:
        if len(road_juncs[rj]["detail"]) >= 2:
            lng_list = []
            lat_list = []
            for item in road_juncs[rj]["detail"]:
                lng_list.append(item[1].x)
                lat_list.append(item[1].y)
            lng_max = np.ptp(lng_list)
            lat_max = np.ptp(lat_list)
            start_junc, end_junc = [p for p in road_juncs[rj]["detail"]][:2]

            if lng_max >= lat_max:
                # the difference of longitude is larger
                lng_list_index = list(np.argsort(lng_list))
                for i, idx in enumerate(lng_list_index):
                    if i == 0:
                        start_junc = road_juncs[rj]["detail"][idx]
                    elif idx == (len(lng_list) - 1):
                        end_junc = road_juncs[rj]["detail"][idx]
            else:
                # the difference of latitude is larger
                lat_list_index = list(np.argsort(lat_list))
                for i, idx in enumerate(lat_list_index):
                    if i == 0:
                        start_junc = road_juncs[rj]["detail"][idx]
                    elif i == (len(lat_list) - 1):
                        end_junc = road_juncs[rj]["detail"][idx]
            # print(rj, "start:", start_junc, "end:", end_junc)
            road_juncs[rj]["start_junc"] = start_junc
            road_juncs[rj]["end_junc"] = end_junc

    res_road_endpoint = []
    for rj in road_juncs:
        # only one junction
        if len(road_juncs[rj]["detail"]) < 2:
            continue

        res_label = [road_juncs[rj]["start_junc"][0], road_juncs[rj]["end_junc"][0]]

        res = [res_label]
        try:
            # negative sample
            current_road_name = rj
            start_road_names = random.sample(list(set(cared_roads_name).difference({rj})), 3)
            end_road_names = random.sample(list(set(cared_roads_name).difference(set([rj] + start_road_names))), 3)
            for i, (sr, er) in enumerate(zip(start_road_names, end_road_names)):
                start_juncs = ["the junction of {} and {}".format(current_road_name, sr),
                            "the junction of {} and {}".format(sr, current_road_name)]
                end_juncs = ["the junction of {} and {}".format(current_road_name, er),
                            "the junction of {} and {}".format(er, current_road_name)]
                for junc in start_juncs + end_juncs:
                    assert junc not in res_label, "make sure negative sample is not in positive sample"
                res.append([random.choice(start_juncs), random.choice(end_juncs)])
        except:
            # print("randomly sample error")
            # print("current_label:{}", res_label)
            # print("negative_sample:{}", start_juncs + end_juncs)
            continue
        question = "Which of the following is the starting intersection and ending intersection of {}?".format(rj)
        answer = res_label
        res_dict = gen_options(res, question, answer)

        res_road_endpoint.append(res_dict)

    task_df = pd.DataFrame(data=res_road_endpoint)
    task_df["is_seen"] = False
    task_df.to_csv(TASK_FILES["city_image"]["road_od"])
    print("city_image road_od task OK!")


def road_linkage_gen(cared_juncs, TASK_FILES):
    road_linkage = {}
    for junc in cared_juncs:
        road_list = cared_juncs[junc]["pre_road_name"]
        if len(road_list) <= 1:
            continue

        for road in list(road_list):
            if road not in road_linkage:
                road_linkage[road] = set(road_list)
            else:
                road_linkage[road] = road_linkage[road].union(set(road_list))
    for road in road_linkage:
        road_linkage[road].remove(road)

    all_roads = set(road_linkage.keys())
    road_nearby = []
    for road_name in road_linkage:
        link_roads = road_linkage[road_name]
        if len(link_roads) > 2:
            res = random.sample(list(link_roads), 3)
            select = random.sample(list(all_roads.difference(link_roads)), 1)
            res.extend(select)
            random.shuffle(res)
            res_dict = dict(zip(["A", "B", "C", "D"], res))
            for k in res_dict:
                if res_dict[k] == select[0]:
                    label = k
            res_dict["question"] = "Which road cannot directly reach {}?".format(road_name)
            res_dict["answer"] = label
        else:
            select = random.sample(list(link_roads), 1)
            res = random.sample(list(all_roads.difference(link_roads)), 3)
            res.extend(select)
            random.shuffle(res)
            question = "Which road directly reach {}?".format(road_name)
            answer = select[0]
            res_dict = gen_options(res, question, answer)
        road_nearby.append(res_dict)

    task_df = pd.DataFrame(data=road_nearby)
    task_df["is_seen"] = False
    task_df.to_csv(TASK_FILES["city_image"]["road_link"])
    print("city_image road_link task OK!")

def generate_evalation_task_node(map, interest_message, interest_dict, category_supported, TASK_FILES):
    if OSM_REGION == False:
        filtered_interests_data = interest_message.dropna(subset=["name"]).query('name != ""')
    else:
        filtered_interests_data = interest_message.dropna(subset=["aoi_name"]).query('aoi_name != ""')

    if filtered_interests_data.shape[0] < 1000:
        node_num = filtered_interests_data.shape[0]
    else:
        node_num = 1000

    interests_data = filtered_interests_data.sample(node_num, random_state=42)
    if OSM_REGION == False:
        interests2name = interests_data.set_index("poi_id")["name"].to_dict()
    else:
        interests2name = interests_data.set_index("aoi_id")["aoi_name"].to_dict()
   
    seen_interests = []
    unseen_interests = []
    for p in interests2name:
        if OSM_REGION == False:
            coords = map.get_poi(p)["position"]
            (x, y) = (coords["x"], coords["y"])
        else:
            # 取中心点
            coords = get_center_point(map.get_aoi(p)["shapely_xy"])
            (x, y) = (coords.x, coords.y)
        
        p_name = interests2name[p]
        try:
            res_dict = landmark_gen((x, y), p_name, interest_dict, category_supported, map)
            if res_dict:
                print("res_dict is not none")
                unseen_interests.append(res_dict)
        except IndexError as e:
            pass
    save_data(unseen_interests, TASK_FILES["urban_semantics"]["landmark_env"])
    print("urban_semantics landmark_env task OK!")

def gps_gen(input_coor, interest_name):
    resolution = 0.01
    round_demical = 4

    lng, lat = input_coor
    lng, lat = round(lng, round_demical), round(lat, round_demical)

    res = [[lng, lat], [lng - resolution, lat - resolution], [lng - resolution, lat + resolution],
           [lng + resolution, lat - resolution], [lng + resolution, lat + resolution]]
    for i, coor in enumerate(res):
        coor[0] = round(coor[0], round_demical)
        coor[1] = round(coor[1], round_demical)
        res[i] = coor
    coor_str = ",".join([str(lng), str(lat)])
    res_str = [",".join([str(xx) for xx in x]) for x in res]
    answer = coor_str
    question = "What is the longitude and latigude coordinates of {}.".format(interest_name)
    res_dict = gen_options(res_str, question, answer)
    return res_dict



def landmark_gen(input_coor, interest_name, interest_dict, category_supported, map):
    x, y = input_coor
    radius = 100
    limit = 10
    resolution = 500

    nearby_interests = []
    for category_prefix in category_supported.keys():
        if OSM_REGION == False:
            interest_list = map.query_pois(input_coor, radius, category_prefix, limit)
            interest_list = [poi[0]['name'] for poi in interest_list if poi[0]['name']]
            nearby_interests.extend(interest_list[:min(limit, len(interest_list))])
        else:
            interest_list = map.query_aois(input_coor, radius, category_prefix, limit)
            interest_list = [aoi[0]["name"] for aoi in interest_list if aoi[0]['id'] in interest_dict]
            nearby_interests.extend(interest_list[:min(limit, len(interest_list))])
    
    if len(nearby_interests) < 5:
        return None

    candidate_interest = [interest_name]
    for center in [(x - resolution, y - resolution), (x - resolution, y + resolution), (x + resolution, y - resolution),
                   (x + resolution, y + resolution)]:
        if OSM_REGION == False:
            interest_list = map.query_pois(center, radius, "", limit)
            interest_list = [poi[0]['name'] for poi in interest_list if poi[0]['name']]
        else:
            interest_list = map.query_aois(center, radius, "", limit)
            interest_list = [aoi[0]["name"] for aoi in interest_list]
            interest_list = list(set(interest_list) & set(interest_dict.keys()))
            if len(interest_list) == 0:
                interest_list.append(random.choice(list(set(interest_dict.keys()).difference(set(interest_list)))))
            interest_list = [interest_dict[aoi]['name'] for aoi in interest_list]
        candidate_interest.append(random.choice(interest_list))
    res_str = [str(x) for x in candidate_interest]
    if OSM_REGION == False:
        question = "Which point of interest (POI) is most likely to appear in the described environment among the following multiple POIs? Environment:{}".format(
        ",".join(nearby_interests))
    else:
        question = "Which area of interest (AOI) is most likely to appear in the described environment among the following multiple AOIs? Environment:{}".format(
        ",".join(nearby_interests))
    answer = str(interest_name)
    res_dict = gen_options(res_str, question, answer)
    return res_dict


def get_nearby_pois(input_coor, map, category_supported):
    lng, lat = input_coor
    radius = 500
    limit = 10
    resolution = 500
    nearby_pois = []
    input_coor = map.lnglat2xy(lng, lat)
    for category_prefix in category_supported.keys():
        poi_list = map.query_pois(input_coor, radius, category_prefix, limit)
        poi_list = [poi[0]["id"] for poi in poi_list]
        nearby_pois.extend(poi_list)
    return nearby_pois


def poi2cor_gen(input_coor, poi_id, poi_dict):
    poi_name = poi_dict[int(poi_id)]['name']
    resolution = 0.01
    round_demical = 4
    lng, lat = input_coor
    lng, lat = round(lng, round_demical), round(lat, round_demical) 
    res = [(lng, lat), (lng - resolution, lat - resolution), (lng - resolution, lat + resolution),
           (lng + resolution, lat - resolution), (lng + resolution, lat + resolution)]
    coor_str = ",".join([str(lng), str(lat)])
    res_str = [",".join([str(round(xx, round_demical)) for xx in x]) for x in res]
    answer = coor_str
    question = "What is the longitude and latigude coordinates of {}.".format(poi_name)
    res_dict = gen_options(res_str, question, answer)
    return res_dict


def poi2addr_gen(nearby_addrs, poi_dict, poi_id):
    tar_addr = poi_dict[int(poi_id)]['Address']
    tar_name = poi_dict[int(poi_id)]['name']
    res_str = []
    for addr in nearby_addrs:
        if len(res_str) < 4:
            if addr != tar_addr:
                res_str.append(addr)
    res_str.append(tar_addr)
    answer = tar_addr
    question = "What is the address of {}?".format(tar_name)
    res_dict = gen_options(res_str, question, answer)
    return res_dict


def poi2type_gen(tar_type, all_pois, type_pois):
    all_poi_cate = type_pois.keys()
    available_types = [cate for cate in all_poi_cate if cate != tar_type]

    res_str = random.sample(available_types, 4)
    res_str.append(tar_type)
    if len(set(type_pois[tar_type])) < 4:
        return None
    tar_poi_ids = random.sample(list(set(type_pois[tar_type])), 4)
    poi_names = [all_pois[poi_id]['name'] for poi_id in tar_poi_ids]
    answer = tar_type
    question = "Which type do following POIs belong to? POIs:{}".format(",".join(poi_names))
    res_dict = gen_options(res_str, question, answer)
    return res_dict


def type2poi_gen(tar_type, all_pois, type_pois):  
    type1_ids = []  # POI_ID not in tar_type
    for type1, ids in type_pois.items():
        if type1 != tar_type:
            type1_ids += ids
    if len(set(type1_ids)) < 4 or len(set(type_pois[tar_type])) < 1:
        return None
    res_poi_ids = random.sample(list(set(type1_ids)), 4)
    tar_poi_id = random.sample(list(set(type_pois[tar_type])), 1)[0]
    tar_poi_name = all_pois[tar_poi_id]['name']
    poi_names = [all_pois[poi_id]['name'] for poi_id in res_poi_ids]
    poi_names.append(tar_poi_name)
    answer = tar_poi_name
    question = "Which POI belongs to {}?".format(tar_type)
    res_dict = gen_options(poi_names, question, answer)
    return res_dict


def aoi_poi_gen(aoi_id, all_pois, all_aois, poi_dict, aoi_dict):
    # aoi_id = random.sample(list(all_aois.keys()),1)
    poi_ids = [poi_id for poi_id in all_aois[aoi_id]['poi_ids'] if all_pois[poi_id]['name']]
    if len(poi_ids) < 1:
        return None
    tar_poi_id = random.sample(poi_ids, 1)[0]
    tar_poi = all_pois[tar_poi_id]['name']
    res_poi_ids = random.sample(list(set(poi_dict.keys()) - set(poi_ids)), 3)
    res_pois = [all_pois[poi_id]['name'] for poi_id in res_poi_ids]
    pois_in_aoi = [all_pois[poi_id]['name'] for poi_id in poi_ids]
    while len(set(res_pois) - set(pois_in_aoi)) < 3: 
        res_poi_id = random.sample(list(set(poi_dict.keys()) - set(poi_ids)), 1)[0]
        res_pois.append(all_pois[res_poi_id]['name'])
    res_pois.append(tar_poi)
    question = "Which of the following POIs is likely to belong to the AOI {}?POIs:{}".format(
        aoi_dict[aoi_id]['name'], ",".join(res_pois))
    answer = tar_poi
    res_dict = gen_options(res_pois, question, answer)
    return res_dict


def poi_aoi_gen(aoi_id, all_pois, all_aois, aoi_dict):
    tar_aoi = aoi_dict[aoi_id]['name']
    valid_poi_ids = [poi_id for poi_id in all_aois[aoi_id]['poi_ids'] if all_pois[poi_id]['name']]
    if len(valid_poi_ids) < 4:
        return None
    poi_ids = random.sample(valid_poi_ids, 4)
    res_aoi_ids = random.sample(list(set(aoi_dict.keys()) - {aoi_id}), 3)
    res_aois = [aoi_dict[aoi_id]['name'] for aoi_id in res_aoi_ids]
    res_aois.append(tar_aoi)
    pois_in_aoi = [all_pois[poi_id]['name'] for poi_id in poi_ids]
    answer = tar_aoi
    question = "Which AOI does the following POIs belong to？POIs:{}".format(",".join(pois_in_aoi))
    res_dict = gen_options(res_aois, question, answer)
    return res_dict


def select_types(poi_ids, poi_dict):
    result_type = []
    poi_ids = [poi_id for poi_id in poi_ids if poi_id in poi_dict]
    poi_types = [poi_dict[poi_id]['category'] for poi_id in poi_ids]
    result_type += [random.sample([poi_id for poi_id in poi_ids if poi_dict[poi_id]['category'] == poi_type],
                                  int((poi_types.count(poi_type)) / 4) + 1) for poi_type in set(poi_types)]
    return list(chain(*result_type))


def aoi2addr_gen(nearby_addrs, aoi_id, aoi_dict):
    tar_addr = aoi_dict[aoi_id]['address']
    tar_name = aoi_dict[aoi_id]['name']
    nearby_addrs.append(tar_addr)

    return gen_options(nearby_addrs, "What is the address of {}?".format(tar_name), tar_addr)


def aoi2type_gen(map, aoi_id, aoi_dict, poi_dict):
    # for external data
    landuse_str = ["OtherNon-construction", "Residential", "TrafficStation&Park", "Sports", "Entertainment", "OtherPublicFacilities", "Education","Park&GreenLand","CommercialService&IndustryFacilities","Resort&Fitness","Restaurant&Bar","ReligiousFacilities","Hospital"]
    try:
        tar_type = task_template(aoi_dict[aoi_id]['category'])
    except KeyError as e:
        return None
    options = [landuse for landuse in landuse_str if landuse != tar_type]
    random_options = random.sample(options, 4)
    random_options.append(tar_type)
    poi_ids = map.get_aoi(aoi_id)['poi_ids']
    poi_in_type = select_types(poi_ids, poi_dict)
    poi_names = [poi_dict[poi_id]['name'] for poi_id in poi_in_type if poi_id in poi_dict]
    if len(poi_names) < 3:
        return None
    question = "Given that {} contains these POIs:{}.Which landuse type does {} belong to?".format(
        aoi_dict[aoi_id]['name'], ",".join(poi_names), aoi_dict[aoi_id]['name']
    )

    return gen_options(random_options, question, tar_type)


def type2aoi_gen(map, landuse, aoi_dict, poi_dict):  
    tar_aoi = []
    res_aoi = []
    for id, aoi in aoi_dict.items():
        type_ = aoi['category']
        if str(type_) == str(landuse): 
            tar_aoi.append(id)
        else:
            res_aoi.append(id)
    if tar_aoi == []:
        return None
    print("tar_aoi is not none")
    aoi_id1 = random.sample(tar_aoi, 1)[0]
    aoi_id2, aoi_id3, aoi_id4 = random.sample(res_aoi, 3)
    poi_names1 = [poi_dict[poi_id]['name'] for poi_id in select_types(map.get_aoi(aoi_id1)['poi_ids'], poi_dict)]
    poi_names2 = [poi_dict[poi_id]['name'] for poi_id in select_types(map.get_aoi(aoi_id2)['poi_ids'], poi_dict) if poi_id in poi_dict]
    poi_names3 = [poi_dict[poi_id]['name'] for poi_id in select_types(map.get_aoi(aoi_id3)['poi_ids'], poi_dict) if poi_id in poi_dict]
    poi_names4 = [poi_dict[poi_id]['name'] for poi_id in select_types(map.get_aoi(aoi_id4)['poi_ids'], poi_dict) if poi_id in poi_dict]
    aoi_names = [aoi_dict[aoi_id1]['name'], aoi_dict[aoi_id2]['name'], aoi_dict[aoi_id3]['name'], aoi_dict[aoi_id4]['name']]
    if len(poi_names1) < 2 or len(poi_names2) < 2 or len(poi_names3) < 2 or len(poi_names4) < 2:
        print("poi_names is not enough")
        return None
    answer = aoi_dict[aoi_id1]['name']
    question = "Given that:1.{} contains these POIs:{}\n2.{} contains these POIs:{}\n3.{} contains these POIs:{}\n4.{} contains these POIs:{}\nWhich of the above AOIs({},{},{},{}) is designated as {}?".format(
        aoi_dict[aoi_id1]['name'], ",".join(poi_names1), aoi_dict[aoi_id2]['name'], ",".join(poi_names2), aoi_dict[aoi_id3]['name'], ",".join(poi_names3), aoi_dict[aoi_id4]['name'],
        ",".join(poi_names4), aoi_dict[aoi_id1]['name'], aoi_dict[aoi_id2]['name'], aoi_dict[aoi_id3]['name'], aoi_dict[aoi_id4]['name'], task_template(landuse))
    res_dict = gen_options(aoi_names, question, answer)
    return res_dict


def districts_poi_type_gen(aoi_id, type_num, aoi_dict):
    res_str = [3, 4, 5, 6, 7]
    answer = type_num
    question = "How many types of POIs are there in {}?".format(aoi_dict[aoi_id]['name'])
    res_dict = gen_options(res_str, question, answer)
    return res_dict


def type_gen(aoi_id, aoi_dict, category_supported):
    """推断AOI的类型"""
    type_ = aoi_dict[aoi_id]['category']
    aoi_name = aoi_dict[aoi_id]['name']
    type_name = category_supported[type_]
    res_types = list(set(category_supported.keys()).difference(set([type_name])))
    res_type = random.choices(res_types, k=3)
    res_type_name = [category_supported[type] for type in res_type]
    res_type_name.append(type_name)
    random.shuffle(res_type_name)
    res_dict = dict(zip(["A", "B", "C", "D"], res_type_name))
    for k in res_dict:
        if res_dict[k] == type_name:
            label = k
    res_dict["answer"] = label
    res_dict["question"] = "What is the land use type of {}?".format(aoi_name)
    res_aois = []
    ids = list(aoi_dict.keys())
    if len(res_aois) < 5:
        random.shuffle(ids)
        for id in ids:
            type = aoi_dict[id]['category']
        if type in res_types:
            res_aois.append(id)
    res_aois.append(aoi_id)
    res_aoi_name = [aoi_dict[id]['name'] for id in res_aois]
    random.shuffle(res_aoi_name)
    res_dict2 = dict(zip(["A", "B", "C", "D"], res_aoi_name))
    for k in res_dict2:
        if res_dict2[k] == aoi_name:
            label = k
    res_dict2["answer"] = label
    res_dict2["question"] = "Which of the following Area of Interests(AOI) has a land use type of {}?".format(type_name)

    return res_dict, res_dict2


def save_data(unseen_pois, save_path):
    unseen_pois_df = pd.DataFrame(data=unseen_pois)
    unseen_pois_df.to_csv(save_path)


def get_node_num(dict):
    if len(dict.keys()) > 200:
        return 200
    else:
        return len(dict.keys())

def generate_evaluation_task_aoi_loc(aoi_dict, TASK_FILES):
    question_num = get_node_num(aoi_dict)
    aois = list(aoi_dict.keys())[:question_num]
    aoi2addr = []
    for aoi_id in aois:
        res = [item for item in aois if item != aoi_id]
        aoi_id1, aoi_id2, aoi_id3 = random.sample(res, 3)
        if len({aoi_dict[aoi_id1]['address'], aoi_dict[aoi_id2]['address'], aoi_dict[aoi_id3]['address']}) < 3:
            continue
        nearby_addrs = [aoi_dict[aoi_id1]['address'], aoi_dict[aoi_id2]['address'], aoi_dict[aoi_id3]['address']]
        aoi2addr.append(aoi2addr_gen(nearby_addrs, aoi_id, aoi_dict))
    save_data(aoi2addr, TASK_FILES["city_image"]["aoi2addr"])
    print("city_image aoi2addr task OK!")

def generate_evaluation_task_aoi_loc_osm(aoi_dict, TASK_FILES):
    question_num = get_node_num(aoi_dict)
    aois = list(aoi_dict.keys())[:question_num]
    aoi2coor = []
    for aoi_id in aois:
        lng, lat = aoi_dict[aoi_id]['coord']
        aoi_name = aoi_dict[aoi_id]['name']
        aoi2coor.append(gps_gen((lng, lat), aoi_name))
    save_data(aoi2coor, TASK_FILES["city_image"]["aoi2coor"])
    print("city_image aoi2coor task OK!")

    aoi2addr = []
    for aoi_id in aois:
        res = [item for item in aois if item != aoi_id]
        aoi_id1, aoi_id2, aoi_id3 = random.sample(res, 3)
        if len({aoi_dict[aoi_id1]['address'], aoi_dict[aoi_id2]['address'], aoi_dict[aoi_id3]['address']}) < 3:
            continue
        nearby_addrs = [aoi_dict[aoi_id1]['address'], aoi_dict[aoi_id2]['address'], aoi_dict[aoi_id3]['address']]
        aoi2addr.append(aoi2addr_gen(nearby_addrs, aoi_id, aoi_dict))
    save_data(aoi2addr, TASK_FILES["city_image"]["aoi2addr"])
    print("city_image aoi2addr task OK!")


def get_land_uses(aoi_dict):
    landuses = set()
    for id, aoi in aoi_dict.items():
        landuse = aoi['category']
        landuses.add(landuse)
    return list(landuses)


def generate_evaluation_task_aoi_type(map, aoi_dict, poi_dict, TASK_FILES):
    question_num = get_node_num(aoi_dict)
    aois = list(aoi_dict.keys())[:question_num]
    aoi2type = []
    for aoi_id in aois:
        res_dict_aoi2type = aoi2type_gen(map, aoi_id, aoi_dict, poi_dict)
        if res_dict_aoi2type:
            aoi2type.append(res_dict_aoi2type)
    save_data(aoi2type, TASK_FILES["urban_semantics"]["aoi2type"])
    print("urban_semantics aoi2type task OK!")
    type2aoi = []
    if OSM_REGION == True:
        landuses = [
            "E3", "R", "S4", "A4", "B31", "U9", "A3", "G1", "B", "B32", "B13", "A9", "A5"
        ]
    else:
        landuses = [
            "7", "8", "5", "6", "10"
        ]
    right = 0
    while right < 50:
        p = random.sample(landuses, 1)[0]
        res_dict_type2aoi = type2aoi_gen(map, p, aoi_dict, poi_dict)
        if res_dict_type2aoi:
            type2aoi.append(res_dict_type2aoi)
            right += 1
    save_data(type2aoi, TASK_FILES["urban_semantics"]["type2aoi"])
    print("urban_semantics type2aoi task OK!")


def generate_districts_poi_type(map, aoi_dict, poi_dict, TASK_FILES):
    question_num = get_node_num(aoi_dict)
    aois = list(aoi_dict.keys())[:question_num]
    districts_poi_type = []
    for aoi_id in aois:
        poi_ids = list(set(map.get_aoi(aoi_id)['poi_ids']) & set(poi_dict.keys()))
        types = set()
        for p in poi_ids:
            types.add(map.get_poi(p)['category'])
        type_num = len(types)
        # print(f"type num: {aoi.type_num}")
        if type_num < 3 or type_num > 7:
            continue
        districts_poi_type.append(districts_poi_type_gen(aoi_id, type_num, aoi_dict))
    save_data(districts_poi_type, TASK_FILES["city_image"]["districts_poi_type"])
    print("city_image districts_poi_type task OK!")


def generate_evaluation_task_poi_loc(map, poi_dict, poi_message, all_pois, category_supported, TASK_FILES):
    question_num = get_node_num(poi_dict)
    pois = random.sample(list(poi_dict.keys()), question_num)
    name_counts = poi_message['name'].value_counts(sort=True)
    many_names = name_counts[name_counts > 10]  
    poi2cor = []
    for p in pois:
        coords = map.get_poi(p)["position"]
        if all_pois[p]['name'] in many_names:
            continue

        if len(coords) > 0:
            lng, lat = map.xy2lnglat(x=coords["x"], y=coords["y"])  
            poi2cor.append(poi2cor_gen((lng, lat), p, poi_dict))
    save_data(poi2cor, TASK_FILES["city_image"]["poi2coor"])
    print("city_image poi2coor task OK!")
    poi2addr = []
    for p in pois:
        coords = map.get_poi(p)["position"] 
        lng, lat = map.xy2lnglat(x=coords["x"], y=coords["y"])
        nearby_id = get_nearby_pois((lng, lat), map, category_supported)
        nearby_addrs = [poi_dict[id]['Address'] for id in nearby_id if id in poi_dict]
        if all_pois[p]['name'] in many_names:
            continue
        if len(list(set(nearby_addrs))) > 4 and p in poi_dict:
            poi2addr.append(poi2addr_gen(nearby_addrs, poi_dict, p))
    save_data(poi2addr, TASK_FILES["city_image"]["poi2addr"])
    print("city_image poi2addr task OK!")


def generate_evaluation_task_poi_type(aoi_dict, all_aois, all_pois, type_pois, TASK_FILES):
    question_num = get_node_num(aoi_dict)
    aois = random.sample(list(aoi_dict.keys()), question_num)
    poi2type = []
    for key in type_pois.keys():
        res_poi2type = poi2type_gen(key, all_pois, type_pois)
        if res_poi2type:
            poi2type.append(res_poi2type)
    save_data(poi2type, TASK_FILES["urban_semantics"]["poi2type"])
    print("urban_semantics poi2type task OK!")
    type2poi = []
    for key in type_pois.keys():
        res_type2poi = type2poi_gen(key, all_pois, type_pois)
        if res_type2poi:
            type2poi.append(res_type2poi)
    save_data(type2poi, TASK_FILES["urban_semantics"]["type2poi"])
    print("urban_semantics type2poi task OK!")


def generate_evaluation_task_poi_aoi(aoi_dict, all_pois, all_aois, poi_dict, TASK_FILES):
    question_num = get_node_num(aoi_dict)
    aois = random.sample(list(aoi_dict.keys()), question_num)
    aoi_poi = []
    for p in aois:
        if len(all_aois[p]['poi_ids']) <= 3 or not isinstance(aoi_dict[p]['name'], str):
            continue
        ret_aoi_poi = aoi_poi_gen(p, all_pois, all_aois, poi_dict, aoi_dict)
        if ret_aoi_poi:
            aoi_poi.append(ret_aoi_poi)
    save_data(aoi_poi, TASK_FILES["city_image"]["aoi_poi"])
    print("city_image aoi_poi task OK!")
    poi_aoi = []
    for p in aois:
        if len(all_aois[p]['poi_ids']) <= 3 or not isinstance(aoi_dict[p]['name'], str):
            continue
        res_poi_aoi = poi_aoi_gen(p, all_pois, all_aois, aoi_dict)
        if res_poi_aoi:
            poi_aoi.append(res_poi_aoi)
        
    save_data(poi_aoi, TASK_FILES["city_image"]["poi_aoi"])
    print("city_image poi_aoi task OK!")

def landmark_path(map, interest_dict, interests_along_route, start_interest, end_interest):
    interests_along_route = list(set(interests_along_route))
    all_interests = random.choices(list(interest_dict.keys()), k=10)
    if OSM_REGION == False:
        all_interests_name = set([map.get_poi(poi_id)['name'] for poi_id in all_interests])
    else:
        all_interests_name = set([map.get_aoi(aoi_id)['name'] for aoi_id in all_interests])
    negative_samples = all_interests_name.difference(set(interests_along_route))
    negative_sample = random.choice(list(negative_samples))
    if len(interests_along_route) >= 3:
        res_interests = random.sample(interests_along_route, k=3)
    else:
        print("interests_along_route is less than 3")
        return None

    res_interests.append(negative_sample)
    if OSM_REGION == False:
        question = "Which of the following POIs will not be passed when traveling from {} to {}?".format(
        start_interest, end_interest)
    else:
        question = "Which of the following AOIs will not be passed when traveling from {} to {}?".format(
        start_interest, end_interest)
    answer = negative_sample
    res_dict = gen_options(res_interests, question, answer)
    return res_dict


async def generate_evaluation_task_landmark(map, routing_client, interest_dict, TASK_FILES):
    route_arrive_interests = []
    count = 0
    while count < 50:
        if OSM_REGION == False:
            # 随机选出两个不同的POI ID
            poi_ids = list(interest_dict.keys())
            start_poi_id, end_poi_id = random.sample(poi_ids, 2)
            start_aoi_id = interest_dict[start_poi_id]["aoi_id"]
            end_aoi_id = interest_dict[end_poi_id]["aoi_id"]
            if start_aoi_id == end_aoi_id:
                continue
        else:
            aoi_ids = list(interest_dict.keys())
            start_aoi_id, end_aoi_id = random.sample(aoi_ids, 2)
        player = TextPlayer(map, routing_client, start_aoi_id, MIN_ROAD_LENGTH, REGION_EXP)
        route = await player.get_driving_route(end_aoi_id)
        if route is None:
            continue
        # 限制步数不超过12步，不少于3步
        if len(route["road_ids"]) > 12 or len(route["road_ids"]) < 3:
            continue
        road_list = []
        for road_id in route["road_ids"]:
            road_info = player._city_map.get_road(road_id)
            lane_info = player._city_map.get_lane(road_info["lane_ids"][0])
            road_list = player.road_info_collect(road_info, lane_info, road_list)
        if OSM_REGION == False:
            start_poi_name = interest_dict[start_poi_id]['name']
            start_poi_addr = interest_dict[start_poi_id]['Address']
            start_interest = "{}{}".format(start_poi_name, start_poi_addr)
            end_poi_name = interest_dict[end_poi_id]['name']
            end_poi_addr = interest_dict[end_poi_id]['Address']
            end_interest = "{}{}".format(end_poi_name, end_poi_addr)
            if not start_poi_name or not end_poi_name:
                continue
        else:
            start_aoi_name = interest_dict[start_aoi_id]['name']
            end_aoi_name = interest_dict[end_aoi_id]['name']
            start_aoi_addr = interest_dict[start_aoi_id]['address']
            end_aoi_addr = interest_dict[end_aoi_id]['address']
            start_interest = "{}{}".format(start_aoi_name, start_aoi_addr)
            end_interest = "{}{}".format(end_aoi_name, end_aoi_addr)
        interests_along_route = []
        while(len(road_list) > 0):
            road_name, road_length, lane_id, direction, _ = road_list.pop(0)
            lane_info = map.get_lane(lane_id)
            endpoint_lnglat = lane_info["shapely_lnglat"].coords[-1]
            endpoint_xy = lane_info["shapely_xy"].coords[-1]

            # update position
            player.position = Position(
                xy_position=XYPosition(x=endpoint_xy[0], y=endpoint_xy[1]),
                longlat_position=LongLatPosition(longitude=endpoint_lnglat[0], latitude=endpoint_lnglat[1]),
            )
            interest_info, _ = player.get_nearby_interests()

            for category, interests in interest_info.items():
                for interest in interests:
                    if 'name' in interest:
                        interests_along_route.append(interest['name'])

        # print("length of pois_along_route: ", len(set(pois_along_route)))
        res_dict = landmark_path(map, interest_dict, interests_along_route, start_interest, end_interest)
        # print("res_dict: ", res_dict) 
        if not res_dict:
            continue
        route_arrive_interests.append(res_dict)
        count += 1
    save_data(route_arrive_interests, TASK_FILES["city_image"]["landmark_path"])
    print("city_image landmark_path task OK!")

def generate_evaluation_task_boudary(all_roads, all_lanes, aoi_dict, REGION_EXP_POLYGON, TASK_FILES):
    cared_roads_name = {}
    for road_id in all_roads:
        lane_ids = all_roads[road_id]["lane_ids"]
        if OSM_REGION == True:
            road_name = all_roads[road_id]['name']
        else:
            road_name = all_roads[road_id]['external']['name']
        if road_name == '':
            continue

        road_in_region = False
        aoi_ids_include = []
        for i, lane_id in enumerate(lane_ids):
            if isinstance(lane_id, list):
                print("lane_id in lane_ids is List!!!!!")
                continue

            lane = all_lanes[lane_id]
            last_point = Point(lane["shapely_lnglat"].coords[-1])
            aoi_ids = lane["aoi_ids"]

            if REGION_EXP_POLYGON.contains(last_point):
                road_in_region = True
                aoi_ids_include.extend(aoi_ids)

        if road_in_region:
            include_aois = []
            for aoi_id in aoi_ids_include:
                if aoi_id in aoi_dict:
                    include_aois.append(aoi_dict[aoi_id]['name'])
            if road_name not in cared_roads_name:
                cared_roads_name[road_name] = include_aois
            else:
                cared_roads_name[road_name].extend(include_aois)
    for road_name in cared_roads_name:
        cared_roads_name[road_name] = list(set(cared_roads_name[road_name]))

    aoi_boundary_dict = {}
    for aoi_id in aoi_dict:
        aoi_name = aoi_dict[aoi_id]['name']
        aoi_boundary_dict[aoi_id] = {"name": aoi_name, "boundary": []}
        for road_name in cared_roads_name:
            if aoi_name in cared_roads_name[road_name]:
                aoi_boundary_dict[aoi_id]["boundary"].append(road_name)

    res_aoi_boundary = []
    candidates_roads = list(cared_roads_name.keys())
    for aoi_id in aoi_boundary_dict:
        if len(aoi_boundary_dict[aoi_id]["boundary"]) == 0:
            continue

        aoi_name = aoi_boundary_dict[aoi_id]["name"]
        boundary = aoi_boundary_dict[aoi_id]["boundary"]

        if len(boundary) >= 3:
            res = random.sample(boundary, 3)

            random.shuffle(candidates_roads)
            for x in candidates_roads:
                if x not in boundary:
                    negative_sample = x
                    break
            res.append(negative_sample)
            question = "Which road is not the boundary of AOI {}".format(aoi_name)
            answer = negative_sample
            res_dict = gen_options(res, question, answer)
        else:
            # 边界较少，询问哪个是边界
            current_roads = list(set(candidates_roads).difference(set(boundary)))
            res = random.sample(current_roads, 3) + random.sample(boundary, 1)
            answer = res[-1]
            question = "Which road is the boundary of AOI {}".format(aoi_name)
            res_dict = gen_options(res, question, answer)
        res_aoi_boundary.append(res_dict)

    task_df = pd.DataFrame(data=res_aoi_boundary)
    task_df["is_seen"] = False
    task_df.to_csv(TASK_FILES["city_image"]["boundary_road"])
    print("city_image boundary_road task OK!")

def generate_evalation_task_boundary_poi(map, aoi_dict, poi_dict, TASK_FILES):
    pois_ids = poi_dict.keys()
    aois_data = []
    available_aois = []
    min_pois = 5
    for aoi_id in aoi_dict.keys():
        info = map.get_aoi(id=aoi_id)
        if len(info["poi_ids"]) >= min_pois:
            aois_data.append(info)
            available_aois.append(aoi_id)
    print("available aois:{}".format(len(available_aois)))

    aoi_boudary_pois = []
    for i, aoi_id in enumerate(available_aois):
        info = aois_data[i]

        if aoi_id in aoi_dict:
            aoi_name = aoi_dict[aoi_id]['name']
        else:
            continue

        poi_ids = info["poi_ids"]
        boudary = info["shapely_xy"]
        center = boudary.centroid
        poi_distance = []
        for poi_id in poi_ids:
            poi_info = map.get_poi(poi_id)
            if poi_info['name'] == '':
                continue
            pos = poi_info["position"]
            pos_shapely = Point((pos["x"], pos["y"]))
            dis = shapely.distance(center, pos_shapely)
            poi_distance.append((poi_id, dis))

        poi_distance = sorted(poi_distance, key=lambda x: x[1])

        if len(poi_distance) < 3:
            continue
        random_pois = random.sample(list(pois_ids), 10)
        random_pois_clean = []
        for poi_id in random_pois:
            if poi_id != poi_distance[0][0] and poi_id != poi_distance[-1][0]:
                if map.get_poi(poi_id)['name'] != '':
                    random_pois_clean.append(poi_id)
        if len(random_pois_clean) < 2:
            continue

        poi_names = [map.get_poi(p[0])['name'] for p in poi_distance]
        res_pois = [poi_names[0], poi_names[1], poi_names[-1],
                    map.get_poi(random_pois_clean[0])['name'],
                    map.get_poi(random_pois_clean[1])['name']]
        label_name = poi_names[-1]
        question = "Which point of interest (POI) is most likely to appear in the boundary of AOI:{}?".format(aoi_name)
        answer = str(label_name)
        res_dict = gen_options(res_pois, question, answer)
        aoi_boudary_pois.append(res_dict)

    task_df = pd.DataFrame(data=aoi_boudary_pois)
    task_df.to_csv(TASK_FILES["city_image"]["aoi_boundary_poi"])
    print("city_image aoi_boundary_poi task OK!")

def generate_evalation_task_districts(map, aoi_message, TASK_FILES):

    def districts_gen(aois, max_group, pois_name_str):
        res = list(range(1, max_group + 1))
        answer = len(aois)
        question = "How many regions can the following POIs be divided into? POIs:{}".format(pois_name_str)
        res_dict = gen_options(res, question, answer)
        return res_dict

    aois_data = []
    available_aois = []
    min_pois = 2
    for aoi_id in aoi_message.aoi_id.to_list():
        info = map.get_aoi(id=aoi_id)
        if len(info["poi_ids"]) >= min_pois:
            aois_data.append(info)
            available_aois.append(aoi_id)
    print("available aois:{}".format(len(available_aois)))

    aois_group = []
    group_size = 50
    max_group = 5
    for i in range(max_group):
        for _ in range(2 * group_size):
            item = tuple(random.sample(available_aois, i + 1))
            if item not in aois_group:
                aois_group.append(item)
            if len(aois_group) == group_size * (i + 1):
                break
    random.shuffle(aois_group)

    tasks = []
    for aois in aois_group:
        pois = []
        for aoi in aois:
            pois_in = map.get_aoi(aoi)["poi_ids"]
            pois_in = [poi_id for poi_id in pois_in if map.get_poi(poi_id)['name']]
            max_len = min(len(pois_in), 6)
            min_len = min(len(pois_in), 1)
            if max_len <= min_len:
                continue
            pois.extend(random.sample(pois_in, random.sample(range(min_len, max_len), 1)[0]))
        pois_names = []
        for poi in pois:
            try:
                poi_name = map.get_poi(poi)['name']
                pois_names.append(poi_name)
            except IndexError as e:
                print(e)
                continue

        if len(pois_names) <= len(aois):
            continue

        random.shuffle(pois_names)
        tasks.append(districts_gen(aois, 5, ",".join(pois_names)))

    task_df = pd.DataFrame(data=tasks)
    task_df["is_seen"] = False
    task_df.to_csv(TASK_FILES["urban_semantics"]["aoi_group"])
    print("urban_semantics aoi_group task OK!")



def get_aoi_type(aoi_id, poi_ids, aoi_dict, poi_dict):
    type_dict = {}
    aoi = aoi_dict[aoi_id]
    for poi_id in poi_ids:
        type1 = poi_dict[poi_id]['category']
        if type1 not in type_dict:
            type_dict[type1] = []
        type_dict[type1].append(poi_id)
    max_length = 0
    most_type = None
    for type1, poi_ids in type_dict.items():
        # 检查当前value的长度是否大于已知的最大长度
        if len(poi_ids) > max_length:
            max_length = len(poi_ids)
            most_type = type1
    return type_dict, most_type


def generate_poi2poi_disdir(diags, diag_routes, poi_dict, name_adr_poi, TASK_FILES):  # 评估步数
    train_dir_routine = []
    train_diags_dir = []
    train_diags_dis = []
    train_dis_routine = []
    for cnt,diag in enumerate(diags):
        content = json.loads(diag[1]["content"])
        start_poi_name = content['start_name']
        start_poi_addr = content['start_addr']
        start_poi = "{}{}".format(start_poi_name, start_poi_addr)
        # 提取"destination"和"by"之间的信息
        end_poi_name = content['dest_name']
        end_poi_addr = content['dest_addr']
        end_poi = "{}{}".format(end_poi_name, end_poi_addr)
        if start_poi not in name_adr_poi or end_poi not in name_adr_poi:
            continue
        start_poi_info = name_adr_poi[start_poi]
        end_poi_info = name_adr_poi[end_poi]
        lon1, lat1 = start_poi_info['coord']
        lon2, lat2 = end_poi_info['coord']
        poi_ids = random.sample(list(set(poi_dict.keys())), 3)
        pois = [poi_dict[poi_id]['name'] for poi_id in poi_ids]
        for id, poi in enumerate(pois):
            if poi == start_poi_name or poi == end_poi_name:
                pois[id] = poi_dict[random.sample(list(set(poi_dict.keys())), 1)[0]]['name']
        pois.append(end_poi_name)
        random.shuffle(pois)
        ques_dir = "In which direction is {} from {}?".format(end_poi, start_poi)
        ques_dis = "How many meters do I need to walk from {} to {} along the road?".format(start_poi, end_poi)
        routinue = 'After starting from {}, you '.format(start_poi) + ", ".join(diag_routes[cnt]) + ". Finally you arrive at {}.".format(end_poi)
        res_dict = dict(zip(["A", "B", "C", "D"], pois))
        for k in res_dict:
            if res_dict[k] == end_poi_name:
                label = k
        direction = angle2dir_4(calcu_azimuth(lat1, lon1, lat2, lon2))  # 计算点1到点2的方向角
        directions = ["east", "south", "west", "north"]
        random.shuffle(directions)
        res_dir_prompt = dict(zip(["A", "B", "C", "D"], directions))
        res_diags_dir = dict(zip(["A", "B", "C", "D"], directions))
        for k in res_dir_prompt:
            if res_dir_prompt[k] == direction:
                label = k
        res_dir_prompt["answer"] = label
        res_diags_dir["answer"] = label
        res_dir_prompt["question"] = routinue + ques_dir
        res_diags_dir["question"] = ques_dir
        train_dir_routine.append(res_dir_prompt)
        train_diags_dir.append(res_diags_dir)
        distance = compute_length(", ".join(diag_routes[cnt]))
        distances = [distance / 2, distance * 2, distance, distance - 1000]
        random.shuffle(distances)
        res_dis_prompt = dict(zip(["A", "B", "C", "D"], distances))
        res_diags_dis = dict(zip(["A", "B", "C", "D"], distances))
        for k in res_dis_prompt:
            if res_dis_prompt[k] == distance:
                label = k
        res_dis_prompt["answer"] = label
        res_dis_prompt["question"] = routinue + ques_dis
        res_diags_dis["answer"] = label
        res_diags_dis["question"] = ques_dis
        train_dis_routine.append(res_dis_prompt)
        train_diags_dis.append(res_diags_dis)

    train_dis_routine_df = pd.DataFrame(data=train_dis_routine)
    train_dir_routine_df = pd.DataFrame(data=train_dir_routine)
    train_diags_dis_df = pd.DataFrame(data=train_diags_dis)
    train_diags_dir_df = pd.DataFrame(data=train_diags_dir)
    train_dir_routine_df.to_csv(TASK_FILES["spatial_reasoning_route"]["poi2poi_dir_routine"])
    train_dis_routine_df.to_csv(TASK_FILES["spatial_reasoning_route"]["poi2poi_dis_routine"])
    train_diags_dir_df.to_csv(TASK_FILES["spatial_reasoning_noroute"]["poi2poi_dir_noroutine"])
    train_diags_dis_df.to_csv(TASK_FILES["spatial_reasoning_noroute"]["poi2poi_dis_noroutine"])
    print("spatial_reasoning poi2poi task OK!")


def get_diag_num(diags):
    if len(diags) < REASON_QUES_NUM:
        return len(diags)
    else:
        return REASON_QUES_NUM


#######POI2RD_DIS
def generate_poi2rd_dis(map, diags, diag_routes, NS, EW, primary_directions, secondary_directions, name_adr_poi, poi_dict, TASK_FILES):
    poi2rd_dis = []
    poi2rd_dis_prompt = []
    reason_ques_num = get_diag_num(diags)
    for cnt, diag in enumerate(diags[:reason_ques_num]):
        content = json.loads(diag[1]["content"])
        start_poi_name = content['start_name']
        start_poi_addr = content['start_addr']
        start_poi = "{}{}".format(start_poi_name, start_poi_addr)
        # 提取"destination"和"by"之间的信息
        end_poi_name = content['dest_name']
        end_poi_addr = content['dest_addr']
        routes = content['routes']
        end_poi = "{}{}".format(end_poi_name, end_poi_addr)
        dir_dis_fin = dir_all_dis(routes, secondary_directions, primary_directions, secondary_dir_to_primary_dirs)[0]
        NSEW = {'NS': [], 'EW': []}
        for dir, dis in dir_dis_fin:
            if dir in NS:
                NSEW['NS'].append(dis)
            elif dir in EW:
                NSEW['EW'].append(dis)
            else:
                print(dir)
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
            routine = 'After starting from {}, you '.format(start_poi) + ", ".join(diag_routes[cnt])  + ".Finally you arrive at {}.{} is on {}.".format(end_poi, start_poi, rd_name)
            if rd_belong_dir in EW:
                distance = np.abs(NSEW['NS'][0] - NSEW['NS'][1])
            elif rd_belong_dir in NS:
                distance = np.abs(NSEW['EW'][0] - NSEW['EW'][1])
            if distance < DIS2CORNER:
                continue
            distances = [int(distance / 2), int(distance * 2), int(distance + 500), distance]
            random.shuffle(distances)
            res_dis = dict(zip(["A", "B", "C", "D"], distances))
            res_dis_prompt = dict(zip(["A", "B", "C", "D"], distances))
            for k in res_dis:
                if res_dis[k] == distance:
                    label = k
            res_dis['answer'] = label
            res_dis_prompt['answer'] = label
            res_dis['question'] = "What is the distance between {} and {}?".format(end_poi, rd_name)
            res_dis_prompt["question"] = routine + "What is the distance between {} and {}?".format(end_poi, rd_name)
            poi2rd_dis.append(res_dis)
            poi2rd_dis_prompt.append(res_dis_prompt)
        elif end_poi in name_adr_poi:  # 使用终点POI所在道路，起点POI
            end_aoi_id = name_adr_poi[end_poi]['aoi_id']
            end_aoi_info = map.aois[end_aoi_id]
            result = get_aoi_address(map, end_aoi_info)
            if result is None:
                print("No address for end AOI!")
                continue
            aoi_s, rd_name, junc_name, aoi_junc_direction, rd_belong_dir = result
            routine = 'After starting from {}, you '.format(start_poi) + ", ".join(diag_routes[cnt]) + ".Finally you arrive at {}.{} is on {}.".format(end_poi, end_poi, rd_name)
            if rd_belong_dir in EW:
                distance = np.abs(NSEW['NS'][0] - NSEW['NS'][1])
            elif rd_belong_dir in NS:
                distance = np.abs(NSEW['EW'][0] - NSEW['EW'][1])
            if distance < DIS2CORNER:
                continue
            distances = [int(distance / 2), int(distance * 2), int(distance + 500), distance]
            random.shuffle(distances)
            res_dis = dict(zip(["A", "B", "C", "D"], distances))
            res_dis_prompt = dict(zip(["A", "B", "C", "D"], distances))
            for k in res_dis:
                if res_dis[k] == distance:
                    label = k
            res_dis['answer'] = label
            res_dis_prompt['answer'] = label
            res_dis['question'] = "What is the distance between {} and {}?".format(start_poi, rd_name)
            res_dis_prompt["question"] = routine + "What is the distance between {} and {}?".format(start_poi, rd_name)
            poi2rd_dis.append(res_dis)
            poi2rd_dis_prompt.append(res_dis_prompt)
    poi2rd_dis_df = pd.DataFrame(data=poi2rd_dis)
    poi2rd_dis_prompt_df = pd.DataFrame(data=poi2rd_dis_prompt)
    poi2rd_dis_df.to_csv(TASK_FILES["spatial_reasoning_noroute"]["poi2rd_dis_noroutine"])
    poi2rd_dis_prompt_df.to_csv(TASK_FILES["spatial_reasoning_route"]["poi2rd_dis_routine"])
    print("spatial_reasoning poi2rd_dis task OK!")


#######AOI2RD_DIS,起终点有一个有AOI名字即可。起点有AOI名字，则使用终点所在道路，反之亦然
def generate_aoi2rd_dis(map, diags, diag_routes, NS, EW, primary_directions, name_adr_poi, aoi_dict, secondary_directions, TASK_FILES):
    aoi2rd_dis = []
    aoi2rd_dis_prompt = []
    reason_ques_num = get_diag_num(diags)
    for cnt, diag in enumerate(diags[:reason_ques_num]):
        content = json.loads(diag[1]["content"])
        start_poi_name = content['start_name']
        start_poi_addr = content['start_addr']
        start_poi = "{}{}".format(start_poi_name, start_poi_addr)
        # 提取"destination"和"by"之间的信息
        end_poi_name = content['dest_name']
        end_poi_addr = content['dest_addr']
        routes = content['routes']
        end_poi = "{}{}".format(end_poi_name, end_poi_addr)
        dir_dis_fin = dir_all_dis(routes, secondary_directions, primary_directions, secondary_dir_to_primary_dirs)[0]
        NSEW = {'NS': [], 'EW': []}
        for dir, dis in dir_dis_fin:
            if dir in NS:
                NSEW['NS'].append(dis)
            elif dir in EW:
                NSEW['EW'].append(dis)
            else:
                print(dir)
        if end_poi not in name_adr_poi or start_poi not in name_adr_poi:
            continue
        aoi_id, aoi_id2 = name_adr_poi[start_poi]['aoi_id'], name_adr_poi[end_poi]['aoi_id']
        if aoi_id not in aoi_dict and aoi_id2 not in aoi_dict:
            continue
        if aoi_id not in aoi_dict:  # 起点AOI无名字，使用起点AOI所在道路，终点AOI名字
            aoi_info = map.aois[aoi_id]
            result = get_aoi_address(map, aoi_info)
            if result is None:
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
            distances = [int(distance / 2), int(distance * 2), int(distance + 500), distance]
            random.shuffle(distances)
            res_dis_aoi = dict(zip(["A", "B", "C", "D"], distances))
            res_dis_prompt_aoi = dict(zip(["A", "B", "C", "D"], distances))
            for k in res_dis_aoi:
                if res_dis_aoi[k] == distance:
                    label = k
            res_dis_aoi['answer'] = label
            res_dis_prompt_aoi['answer'] = label
            res_dis_aoi['question'] = "What is the distance between {} and {}?".format(aoi_name_addr, rd_name)
            res_dis_prompt_aoi["question"] = routine + "What is the distance between {} and {}?".format(aoi_name_addr,
                                                                                                        rd_name)
            aoi2rd_dis.append(res_dis_aoi)
            aoi2rd_dis_prompt.append(res_dis_prompt_aoi)

        if aoi_id2 not in aoi_dict:  # 终点AOI无名字，使用终点AOI所在道路
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
            distances = [int(distance / 2), int(distance * 2), int(distance + 500), distance]
            random.shuffle(distances)
            res_dis_aoi = dict(zip(["A", "B", "C", "D"], distances))
            res_dis_prompt_aoi = dict(zip(["A", "B", "C", "D"], distances))
            for k in res_dis_aoi:
                if res_dis_aoi[k] == distance:
                    label = k
            res_dis_aoi['answer'] = label
            res_dis_prompt_aoi['answer'] = label
            res_dis_aoi['question'] = "What is the distance between {} and {}?".format(aoi_name_addr, rd_name)
            res_dis_prompt_aoi["question"] = routine + "What is the distance between {} and {}?".format(aoi_name_addr,
                                                                                                        rd_name)
            aoi2rd_dis.append(res_dis_aoi)
            aoi2rd_dis_prompt.append(res_dis_prompt_aoi)

    aoi2rd_dis_df = pd.DataFrame(data=aoi2rd_dis)
    aoi2rd_dis_prompt_df = pd.DataFrame(data=aoi2rd_dis_prompt)

    aoi2rd_dis_df.to_csv(TASK_FILES["spatial_reasoning_noroute"]["aoi2rd_dis_noroutine"])
    aoi2rd_dis_prompt_df.to_csv(TASK_FILES["spatial_reasoning_route"]["aoi2rd_dis_routine"])
    print("spatial_reasoning aoi2rd_dis task OK!")


def generate_aoi2rd_dis_osm(map, diags, diag_routes, NS, EW, primary_directions, name_adr_aoi, secondary_directions, TASK_FILES):
    aoi2rd_dis = []
    aoi2rd_dis_prompt = []
    reason_ques_num = get_diag_num(diags)
    for cnt, diag in enumerate(diags[:reason_ques_num]):
        content = json.loads(diag[1]["content"])
        start_aoi_name = content['start_name']
        start_aoi_addr = content['start_addr']
        start_aoi = "{}{}".format(start_aoi_name, start_aoi_addr)
        # 提取"destination"和"by"之间的信息
        end_aoi_name = content['dest_name']
        end_aoi_addr = content['dest_addr']
        end_aoi = "{}{}".format(end_aoi_name, end_aoi_addr)
        routes = content['routes']
        dir_dis_fin = dir_all_dis(routes, secondary_directions, primary_directions, secondary_dir_to_primary_dirs)[0]
        NSEW = {'NS': [], 'EW': []}
        for dir, dis in dir_dis_fin:
            if dir in NS:
                NSEW['NS'].append(dis)
            elif dir in EW:
                NSEW['EW'].append(dis)
            else:
                print(dir)
        if end_aoi not in name_adr_aoi and start_aoi not in name_adr_aoi:
            continue
        
        if end_aoi not in name_adr_aoi:  # 起点AOI无名字，使用起点AOI所在道路，终点AOI名字
            start_aoi_id = name_adr_aoi[start_aoi]['aoi_id']
            start_aoi_info = map.aois[start_aoi_id]
            result = get_aoi_address(map, start_aoi_info)
            if result is None:
                continue
            aoi_s, rd_name, junc_name, aoi_junc_direction, rd_belong_dir = result
            routine = 'After starting from {}, you '.format(start_aoi) + ", ".join(diag_routes[cnt]) + ". Finally you arrive at {}.{} is on {}.".format(end_aoi, start_aoi, rd_name)
            if rd_belong_dir in EW:
                distance = np.abs(NSEW['NS'][0] - NSEW['NS'][1])
            elif rd_belong_dir in NS:
                distance = np.abs(NSEW['EW'][0] - NSEW['EW'][1])
            if distance < 50:
                continue
            distances = [int(distance / 2), int(distance * 2), int(distance + 500), distance]
            random.shuffle(distances)
            res_dis_aoi = dict(zip(["A", "B", "C", "D"], distances))
            res_dis_prompt_aoi = dict(zip(["A", "B", "C", "D"], distances))
            for k in res_dis_aoi:
                if res_dis_aoi[k] == distance:
                    label = k
            res_dis_aoi['answer'] = label
            res_dis_prompt_aoi['answer'] = label
            res_dis_aoi['question'] = "What is the distance between {} and {}?".format(end_aoi, rd_name)
            res_dis_prompt_aoi["question"] = routine + "What is the distance between {} and {}?".format(end_aoi, rd_name)
            aoi2rd_dis.append(res_dis_aoi)
            aoi2rd_dis_prompt.append(res_dis_prompt_aoi)

        if start_aoi not in name_adr_aoi:  # 终点AOI无名字，使用终点AOI所在道路
            end_aoi_id = name_adr_aoi[end_aoi]['aoi_id']
            end_aoi_info = map.aois[end_aoi_id]
            result2 = get_aoi_address(map, end_aoi_info)
            if result2 is None:
                continue
            aoi_s, rd_name, junc_name, aoi_junc_direction, rd_belong_dir = result2
            routine = 'After starting from {}, you '.format(start_aoi) + ", ".join(diag_routes[cnt]) + ". Finally you arrive at {}.{} is on {}.".format(end_aoi, end_aoi, rd_name)
            if rd_belong_dir in EW:
                distance = np.abs(NSEW['NS'][0] - NSEW['NS'][1])
            elif rd_belong_dir in NS:
                distance = np.abs(NSEW['EW'][0] - NSEW['EW'][1])
            if distance < 50:
                continue
            distances = [int(distance / 2), int(distance * 2), int(distance + 500), distance]
            random.shuffle(distances)
            res_dis_aoi = dict(zip(["A", "B", "C", "D"], distances))
            res_dis_prompt_aoi = dict(zip(["A", "B", "C", "D"], distances))
            for k in res_dis_aoi:
                if res_dis_aoi[k] == distance:
                    label = k
            res_dis_aoi['answer'] = label
            res_dis_prompt_aoi['answer'] = label
            res_dis_aoi['question'] = "What is the distance between {} and {}?".format(start_aoi, rd_name)
            res_dis_prompt_aoi["question"] = routine + "What is the distance between {} and {}?".format(start_aoi, rd_name)
            aoi2rd_dis.append(res_dis_aoi)
            aoi2rd_dis_prompt.append(res_dis_prompt_aoi)

    aoi2rd_dis_df = pd.DataFrame(data=aoi2rd_dis)
    aoi2rd_dis_prompt_df = pd.DataFrame(data=aoi2rd_dis_prompt)

    aoi2rd_dis_df.to_csv(TASK_FILES["spatial_reasoning_noroute"]["aoi2rd_dis_noroutine"])
    aoi2rd_dis_prompt_df.to_csv(TASK_FILES["spatial_reasoning_route"]["aoi2rd_dis_routine"])
    print("spatial_reasoning aoi2rd_dis task OK!")


# AOI2AOI_DIS&DIR，过滤条件严格（起终点都要有AOI名字），和AOI2POI&POI2AOI分开
def generate_aoi2aoi_disdir(diags, diag_routes, aoi_dict, name_adr_poi, TASK_FILES):
    aoi2aoi_dis = []
    aoi2aoi_dis_prompt = []
    aoi2aoi_dir = []
    aoi2aoi_dir_prompt = []
    #########修改此处长度：1/2
    reason_ques_num = get_diag_num(diags)
    for cnt, diag in enumerate(diags[:reason_ques_num]):
        content = json.loads(diag[1]["content"])
        start_poi_name = content['start_name']
        start_poi_addr = content['start_addr']
        start_poi = "{}{}".format(start_poi_name, start_poi_addr)
        # 提取"destination"和"by"之间的信息
        end_poi_name = content['dest_name']
        end_poi_addr = content['dest_addr']
        end_poi = "{}{}".format(end_poi_name, end_poi_addr)
        distance = compute_length(", ".join(diag_routes[cnt]))
        distances = [int(distance / 2), int(distance * 2), int(distance + 500), distance]
        random.shuffle(distances)
        # 计算终点相对于起点方向
        if end_poi not in name_adr_poi or start_poi not in name_adr_poi:
            continue
        lon2, lat2 = name_adr_poi[end_poi]['coord']
        lon1, lat1 = name_adr_poi[start_poi]['coord']
        direction = angle2dir_4(calcu_azimuth(lat1, lon1, lat2, lon2))  # 计算点1(start_poi)到点2(end_aoi)的方向角
        directions = ['east', 'west', 'south', 'north']
        random.shuffle(directions)

        aoi_id, aoi_id2 = name_adr_poi[end_poi]['aoi_id'], name_adr_poi[start_poi]['aoi_id']
        if aoi_id in aoi_dict and aoi_id2 in aoi_dict:  # AOI2AOI
            res_dis_aoi = dict(zip(["A", "B", "C", "D"], distances))
            res_dis_prompt_aoi = dict(zip(["A", "B", "C", "D"], distances))
            res_dir_aoi = dict(zip(["A", "B", "C", "D"], directions))
            res_dir_prompt_aoi = dict(zip(["A", "B", "C", "D"], directions))
            end_aoi_name, end_aoi_addr = aoi_dict[aoi_id]['name'], aoi_dict[aoi_id]['address']
            end_aoi_name_addr = "{}({})".format(end_aoi_name, end_aoi_addr)
            start_aoi_name, start_aoi_addr = aoi_dict[aoi_id2]['name'], aoi_dict[aoi_id2]['address']
            start_aoi_name_addr = "{}({})".format(start_aoi_name, start_aoi_addr)
            routine = 'After starting from {}, you '.format(start_poi) + ", ".join(diag_routes[cnt]) + ".Finally you arrive at {}.{} is in {},{} is in {}.".format(end_poi, end_poi, end_aoi_name_addr,
                                                                                 start_poi, start_aoi_name_addr)
            res_dis_aoi['question'] = "How many meters do I need to walk from {} to {} along the road?".format(
                start_aoi_name_addr, end_aoi_name_addr)
            res_dis_prompt_aoi[
                "question"] = routine + "How many meters do I need to walk from {} to {} along the road?".format(
                start_aoi_name_addr, end_aoi_name_addr)
            res_dir_aoi['question'] = "In which direction is {} from {}?".format(end_aoi_name_addr, start_aoi_name_addr)
            res_dir_prompt_aoi["question"] = routine + "In which direction is {} from {}?".format(end_aoi_name_addr,
                                                                                                  start_aoi_name_addr)
            for k in res_dis_aoi:
                if res_dis_aoi[k] == distance:
                    label1 = k
            for k in res_dir_aoi:
                if res_dir_aoi[k] == direction:
                    label2 = k
            res_dis_aoi['answer'] = label1
            res_dis_prompt_aoi["answer"] = label1
            res_dir_aoi['answer'] = label2
            res_dir_prompt_aoi["answer"] = label2
            aoi2aoi_dis.append(res_dis_aoi)
            aoi2aoi_dis_prompt.append(res_dis_prompt_aoi)
            aoi2aoi_dir.append(res_dir_aoi)
            aoi2aoi_dir_prompt.append(res_dir_prompt_aoi)
    aoi2aoi_dis_df = pd.DataFrame(data=aoi2aoi_dis)
    aoi2aoi_dis_prompt_df = pd.DataFrame(data=aoi2aoi_dis_prompt)
    aoi2aoi_dir_df = pd.DataFrame(data=aoi2aoi_dir)
    aoi2aoi_dir_prompt_df = pd.DataFrame(data=aoi2aoi_dir_prompt)
    aoi2aoi_dis_df.to_csv(TASK_FILES["spatial_reasoning_noroute"]["aoi2aoi_dis_noroutine"])
    aoi2aoi_dis_prompt_df.to_csv(TASK_FILES["spatial_reasoning_route"]["aoi2aoi_dis_routine"])
    aoi2aoi_dir_df.to_csv(TASK_FILES["spatial_reasoning_noroute"]["aoi2aoi_dir_noroutine"])
    aoi2aoi_dir_prompt_df.to_csv(TASK_FILES["spatial_reasoning_route"]["aoi2aoi_dir_routine"])
    print("spatial_reasoning aoi2aoi task OK!")


def generate_aoi2aoi_disdir_osm(diags, diag_routes, aoi_dict, name_adr_aoi, TASK_FILES):
    aoi2aoi_dis = []
    aoi2aoi_dis_prompt = []
    aoi2aoi_dir = []
    aoi2aoi_dir_prompt = []
    #########修改此处长度：1/2
    reason_ques_num = get_diag_num(diags)
    for cnt, diag in enumerate(diags[:reason_ques_num]):
        content = json.loads(diag[1]["content"])
        start_aoi_name = content['start_name']
        start_aoi_addr = content['start_addr']
        start_aoi = "{}{}".format(start_aoi_name, start_aoi_addr)
        # 提取"destination"和"by"之间的信息
        end_aoi_name = content['dest_name']
        end_aoi_addr = content['dest_addr']
        end_aoi = "{}{}".format(end_aoi_name, end_aoi_addr)
        distance = compute_length(", ".join(diag_routes[cnt]))
        distances = [int(distance / 2), int(distance * 2), int(distance + 500), distance]
        random.shuffle(distances)
        # 计算终点相对于起点方向
        if end_aoi not in name_adr_aoi or start_aoi not in name_adr_aoi:
            continue
        lon2, lat2 = name_adr_aoi[end_aoi]['coord']
        lon1, lat1 = name_adr_aoi[start_aoi]['coord']
        direction = angle2dir_4(calcu_azimuth(lat1, lon1, lat2, lon2))  # 计算点1(start_poi)到点2(end_aoi)的方向角
        directions = ['east', 'west', 'south', 'north']
        random.shuffle(directions)

        aoi_ids = random.sample(list(set(aoi_dict.keys())), 3)
        aois = [aoi_dict[aoi_id]['name'] for aoi_id in aoi_ids]
        for id, aoi in enumerate(aois):
            if aoi == start_aoi_name or aoi == end_aoi_name:
                aois[id] = aoi_dict[random.sample(list(set(aoi_dict.keys())), 1)[0]]['name']
        aois.append(end_aoi_name)
        random.shuffle(aois)
        
        res_dis_aoi = dict(zip(["A", "B", "C", "D"], distances))
        res_dis_prompt_aoi = dict(zip(["A", "B", "C", "D"], distances))
        res_dir_aoi = dict(zip(["A", "B", "C", "D"], directions))
        res_dir_prompt_aoi = dict(zip(["A", "B", "C", "D"], directions))
        routine = 'After starting from {}, you '.format(start_aoi) + ", ".join(diag_routes[cnt]) + ".Finally you arrive at {}.".format(end_aoi)
        res_dis_aoi['question'] = "How many meters do I need to walk from {} to {} along the road?".format(
            start_aoi, end_aoi)
        res_dis_prompt_aoi[
            "question"] = routine + "How many meters do I need to walk from {} to {} along the road?".format(
            start_aoi, end_aoi)
        res_dir_aoi['question'] = "In which direction is {} from {}?".format(end_aoi, start_aoi)
        res_dir_prompt_aoi["question"] = routine + "In which direction is {} from {}?".format(end_aoi, start_aoi)
        for k in res_dis_aoi:
            if res_dis_aoi[k] == distance:
                label1 = k
        for k in res_dir_aoi:
            if res_dir_aoi[k] == direction:
                label2 = k
        res_dis_aoi['answer'] = label1
        res_dis_prompt_aoi["answer"] = label1
        res_dir_aoi['answer'] = label2
        res_dir_prompt_aoi["answer"] = label2
        aoi2aoi_dis.append(res_dis_aoi)
        aoi2aoi_dis_prompt.append(res_dis_prompt_aoi)
        aoi2aoi_dir.append(res_dir_aoi)
        aoi2aoi_dir_prompt.append(res_dir_prompt_aoi)
    aoi2aoi_dis_df = pd.DataFrame(data=aoi2aoi_dis)
    aoi2aoi_dis_prompt_df = pd.DataFrame(data=aoi2aoi_dis_prompt)
    aoi2aoi_dir_df = pd.DataFrame(data=aoi2aoi_dir)
    aoi2aoi_dir_prompt_df = pd.DataFrame(data=aoi2aoi_dir_prompt)
    aoi2aoi_dis_df.to_csv(TASK_FILES["spatial_reasoning_noroute"]["aoi2aoi_dis_noroutine"])
    aoi2aoi_dis_prompt_df.to_csv(TASK_FILES["spatial_reasoning_route"]["aoi2aoi_dis_routine"])
    aoi2aoi_dir_df.to_csv(TASK_FILES["spatial_reasoning_noroute"]["aoi2aoi_dir_noroutine"])
    aoi2aoi_dir_prompt_df.to_csv(TASK_FILES["spatial_reasoning_route"]["aoi2aoi_dir_routine"])
    print("spatial_reasoning aoi2aoi task OK!")


# AOI2POI DIS&DIR，起终点有一个有AOI名字即可
def generate_aoi2poi_disdir(diags, diag_routes, poi_dict, aoi_dict, name_adr_poi, TASK_FILES):
    poi2aoi_dis = []
    poi2aoi_dis_prompt = []
    poi2aoi_dir = []
    poi2aoi_dir_prompt = []
    reason_ques_num = get_diag_num(diags)
    print(reason_ques_num)
    for cnt, diag in enumerate(diags[:reason_ques_num]):
        content = json.loads(diag[1]["content"])
        start_poi_name = content['start_name']
        start_poi_addr = content['start_addr']
        start_poi = "{}{}".format(start_poi_name, start_poi_addr)
        # 提取"destination"和"by"之间的信息
        end_poi_name = content['dest_name']
        end_poi_addr = content['dest_addr']
        end_poi = "{}{}".format(end_poi_name, end_poi_addr)
        # 计算起终点距离
        distance = compute_length(", ".join(diag_routes[cnt]))
        distances = [int(distance / 2), int(distance * 2), int(distance + 500), distance]
        random.shuffle(distances)
        # 计算终点相对于起点方向
        if end_poi not in name_adr_poi or start_poi not in name_adr_poi:
            continue
        lon2, lat2 = name_adr_poi[end_poi]['coord']
        lon1, lat1 = name_adr_poi[start_poi]['coord']
        direction = angle2dir_4(calcu_azimuth(lat1, lon1, lat2, lon2))  # 计算点1(start_poi)到点2(end_aoi)的方向角
        directions = ['east', 'west', 'south', 'north']
        random.shuffle(directions)

        aoi_id, aoi_id2 = name_adr_poi[end_poi]['aoi_id'], name_adr_poi[start_poi]['aoi_id']
        if aoi_id not in aoi_dict and aoi_id2 not in aoi_dict:
            continue
        # AOI2POI&POI2AOI
        res_dis = dict(zip(["A", "B", "C", "D"], distances))
        res_dis_prompt = dict(zip(["A", "B", "C", "D"], distances))
        res_dir = dict(zip(["A", "B", "C", "D"], directions))
        res_dir_prompt = dict(zip(["A", "B", "C", "D"], directions))
        if aoi_id not in aoi_dict:  # AOI2POI
            start_aoi_name, start_aoi_addr = aoi_dict[aoi_id2]['name'], aoi_dict[aoi_id2]['address']
            start_aoi_name_addr = "{}({})".format(start_aoi_name, start_aoi_addr)
            routine = 'After starting from {}, you '.format(start_poi) + ", ".join(diag_routes[cnt]) + ".Finally you arrive at {}.{} is in {}.".format(end_poi, start_poi, start_aoi_name_addr)
            res_dis['question'] = "How many meters do I need to walk from {} to {} along the road?".format(
                start_aoi_name_addr, end_poi)
            res_dis_prompt[
                "question"] = routine + "How many meters do I need to walk from {} to {} along the road?".format(
                start_aoi_name_addr, end_poi)
            res_dir['question'] = "In which direction is {} from {}?".format(end_poi, start_aoi_name_addr)
            res_dir_prompt["question"] = routine + "In which direction is {} from {}?".format(end_poi,
                                                                                              start_aoi_name_addr)
            for k in res_dis:
                if res_dis[k] == distance:
                    label1 = k
            for k in res_dir:
                if res_dir[k] == direction:
                    label2 = k
            res_dis['answer'] = label1
            res_dis_prompt["answer"] = label1
            res_dir['answer'] = label2
            res_dir_prompt["answer"] = label2
            poi2aoi_dis.append(res_dis)
            poi2aoi_dis_prompt.append(res_dis_prompt)
            poi2aoi_dir.append(res_dir)
            poi2aoi_dir_prompt.append(res_dir_prompt)
        else:  # POI2AOI
            end_aoi_name, end_aoi_addr = aoi_dict[aoi_id]['name'], aoi_dict[aoi_id]['address']
            end_aoi_name_addr = "{}({})".format(end_aoi_name, end_aoi_addr)
            routine = 'After starting from {}, you '.format(start_poi) + ", ".join(diag_routes[cnt]) + ".Finally you arrive at {}.{} is in {}.".format(end_poi, end_poi, end_aoi_name_addr)
            res_dis['question'] = "How many meters do I need to walk from {} to {} along the road?".format(start_poi,
                                                                                                           end_aoi_name_addr)
            res_dis_prompt[
                "question"] = routine + "How many meters do I need to walk from {} to {} along the road?".format(
                start_poi, end_aoi_name_addr)
            res_dir['question'] = "In which direction is {} from {}?".format(end_aoi_name_addr, start_poi)
            res_dir_prompt["question"] = routine + "In which direction is {} from {}?".format(end_aoi_name_addr,
                                                                                              start_poi)
            for k in res_dis:
                if res_dis[k] == distance:
                    label1 = k
            for k in res_dir:
                if res_dir[k] == direction:
                    label2 = k
            res_dis['answer'] = label1
            res_dis_prompt["answer"] = label1
            res_dir['answer'] = label2
            res_dir_prompt["answer"] = label2
            poi2aoi_dis.append(res_dis)
            poi2aoi_dis_prompt.append(res_dis_prompt)
            poi2aoi_dir.append(res_dir)
            poi2aoi_dir_prompt.append(res_dir_prompt)
    poi2aoi_dis_df = pd.DataFrame(data=poi2aoi_dis)
    poi2aoi_dis_prompt_df = pd.DataFrame(data=poi2aoi_dis_prompt)
    poi2aoi_dir_df = pd.DataFrame(data=poi2aoi_dir)
    poi2aoi_dir_prompt_df = pd.DataFrame(data=poi2aoi_dir_prompt)
    poi2aoi_dis_df.to_csv(TASK_FILES["spatial_reasoning_noroute"]["poi2aoi_dis_noroutine"])
    poi2aoi_dis_prompt_df.to_csv(TASK_FILES["spatial_reasoning_route"]["poi2aoi_dis_routine"])
    poi2aoi_dir_df.to_csv(TASK_FILES["spatial_reasoning_noroute"]["poi2aoi_dir_noroutine"])
    poi2aoi_dir_prompt_df.to_csv(TASK_FILES["spatial_reasoning_route"]["poi2aoi_dir_routine"])
    print("spatial_reasoning poi2aoi task OK!")


########POI2RD_DIR,终点POI到起点道路方向
def generate_poi2rd_dir(map, diags, diag_routes, NS, EW, primary_directions, secondary_directions, name_adr_poi, poi_dict, TASK_FILES):
    poi2rd_dir = []
    poi2rd_dir_prompt = []
    reason_ques_num = get_diag_num(diags)
    for cnt, diag in enumerate(diags[:reason_ques_num]):
        # tar_rd_set = set()
        content = json.loads(diag[1]["content"])
        routes = content['routes']
        dir_dis_fin = dir_all_dis(routes, secondary_directions, primary_directions, secondary_dir_to_primary_dirs)[0]
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
        content = json.loads(diag[1]["content"])
        start_poi_name = content['start_name']
        start_poi_addr = content['start_addr']
        start_poi = "{}{}".format(start_poi_name, start_poi_addr)
        # 提取"destination"和"by"之间的信息
        end_poi_name = content['dest_name']
        end_poi_addr = content['dest_addr']
        end_poi = "{}{}".format(end_poi_name, end_poi_addr)
        routine = 'After starting from {}, you '.format(start_poi) + ", ".join(diag_routes[cnt]) + ".Finally you arrive at {}.".format(end_poi)
        if start_poi not in name_adr_poi:
            continue
        start_aoi_id = name_adr_poi[start_poi]['aoi_id']
        start_aoi_info = map.aois[start_aoi_id]
        result = get_aoi_address(map, start_aoi_info)
        if result is None:
            continue
        aoi_s, rd_name, junc_name, aoi_junc_direction, rd_belong_dir = result
        # if 270<angle<360:  #终点在起点西北方向
        if dir_set == {'west', 'north'}:
            if rd_belong_dir in NS:
                tar_dir = 'west'
            elif rd_belong_dir in EW:
                tar_dir = 'north'
        elif dir_set == {'east', 'north'}:  ##终点在起点东北方向
            if rd_belong_dir in NS:
                tar_dir = 'east'
            elif rd_belong_dir in EW:
                tar_dir = 'north'
        elif dir_set == {'east', 'south'}:  # 终点在起点东南方向
            if rd_belong_dir in NS:
                tar_dir = 'east'
            elif rd_belong_dir in EW:
                tar_dir = 'south'
        elif dir_set == {'west', 'south'}:  # 终点在起点西南方向
            if rd_belong_dir in NS:
                tar_dir = 'west'
            elif rd_belong_dir in EW:
                tar_dir = 'south'
        random.shuffle(primary_directions)
        res_dict = dict(zip(["A", "B", "C", "D"], primary_directions))
        res_dict_prompt = dict(zip(["A", "B", "C", "D"], primary_directions))
        for k in res_dict:
            if res_dict[k] == tar_dir:
                label = k
        res_dict['answer'] = label
        res_dict_prompt['answer'] = label
        res_dict['question'] = "In which direction is {} from {}?".format(end_poi, rd_name)
        res_dict_prompt['question'] = routine + "{} is on {}.In which direction is {} from {}?".format(start_poi,
                                                                                                       rd_name, end_poi,
                                                                                                       rd_name)
        poi2rd_dir.append(res_dict)
        poi2rd_dir_prompt.append(res_dict_prompt)

    poi2rd_dir_df = pd.DataFrame(data=poi2rd_dir)
    poi2rd_dir_prompt_df = pd.DataFrame(data=poi2rd_dir_prompt)

    poi2rd_dir_df.to_csv(TASK_FILES["spatial_reasoning_noroute"]["poi2rd_dir_noroutine"])
    poi2rd_dir_prompt_df.to_csv(TASK_FILES["spatial_reasoning_route"]["poi2rd_dir_routine"])
    print("spatial_reasoning poi2rd_dir task OK!")


########AOI2RD_DIR,终点AOI到起点所在道路方向
def generate_aoi2rd_dir(map, diags, diag_routes, NS, EW, primary_directions, secondary_directions, name_adr_poi, aoi_dict, TASK_FILES):
    aoi2rd_dir = []
    aoi2rd_dir_prompt = []
    reason_ques_num = get_diag_num(diags)
    for cnt, diag in enumerate(diags[:reason_ques_num]):
        # tar_rd_set = set()
        content = json.loads(diag[1]["content"])
        routes = content['routes']
        dir_dis_fin = dir_all_dis(routes, secondary_directions, primary_directions, secondary_dir_to_primary_dirs)[0]
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
        content = json.loads(diag[1]["content"])
        start_poi_name = content['start_name']
        start_poi_addr = content['start_addr']
        start_poi = "{}{}".format(start_poi_name, start_poi_addr)
        # 提取"destination"和"by"之间的信息
        end_poi_name = content['dest_name']
        end_poi_addr = content['dest_addr']
        end_poi = "{}{}".format(end_poi_name, end_poi_addr)
        routine = 'After starting from {}, you '.format(start_poi) + ", ".join(diag_routes[cnt]) + ".Finally you arrive at {}.".format(end_poi)
        if end_poi not in name_adr_poi or start_poi not in name_adr_poi:
            continue
        aoi_id = name_adr_poi[end_poi]['aoi_id']
        if aoi_id not in aoi_dict:
            continue
        end_aoi_name = aoi_dict[aoi_id]['name']
        end_aoi_addr = aoi_dict[aoi_id]['address']
        end_aoi_name_addr = "{}({})".format(end_aoi_name, end_aoi_addr)

        start_aoi_id = name_adr_poi[start_poi]['aoi_id']
        start_aoi_info = map.aois[start_aoi_id]
        result = get_aoi_address(map, start_aoi_info)
        if result is None:
            continue
        aoi_s, rd_name, junc_name, aoi_junc_direction, rd_belong_dir = result
        
        # if 270<angle<360:  #终点在起点西北方向
        if dir_set == {'west', 'north'}:
            if rd_belong_dir in NS:
                tar_dir = 'west'
            elif rd_belong_dir in EW:
                tar_dir = 'north'
        elif dir_set == {'east', 'north'}:  ##终点在起点东北方向
            if rd_belong_dir in NS:
                tar_dir = 'east'
            elif rd_belong_dir in EW:
                tar_dir = 'north'
        elif dir_set == {'east', 'south'}:  # 终点在起点东南方向
            if rd_belong_dir in NS:
                tar_dir = 'east'
            elif rd_belong_dir in EW:
                tar_dir = 'south'
        elif dir_set == {'west', 'south'}:  # 终点在起点西南方向
            if rd_belong_dir in NS:
                tar_dir = 'west'
            elif rd_belong_dir in EW:
                tar_dir = 'south'
        random.shuffle(primary_directions)
        res_dict_aoi = dict(zip(["A", "B", "C", "D"], primary_directions))
        res_dict_prompt_aoi = dict(zip(["A", "B", "C", "D"], primary_directions))
        for k in res_dict_aoi:
            if res_dict_aoi[k] == tar_dir:
                label = k
        res_dict_aoi['answer'] = label
        res_dict_prompt_aoi['answer'] = label
        res_dict_aoi['question'] = "In which direction is {} from {}?".format(end_aoi_name_addr, rd_name)
        res_dict_prompt_aoi['question'] = routine + "{} is in {},{} is on {}.In which direction is {} from {}?".format(
            end_poi, end_aoi_name_addr, start_poi, rd_name, end_aoi_name_addr, rd_name)
        aoi2rd_dir.append(res_dict_aoi)
        aoi2rd_dir_prompt.append(res_dict_prompt_aoi)
    aoi2rd_dir_df = pd.DataFrame(data=aoi2rd_dir)
    aoi2rd_dir_prompt_df = pd.DataFrame(data=aoi2rd_dir_prompt)
    aoi2rd_dir_df.to_csv(TASK_FILES["spatial_reasoning_noroute"]["aoi2rd_dir_noroutine"])
    aoi2rd_dir_prompt_df.to_csv(TASK_FILES["spatial_reasoning_route"]["aoi2rd_dir_routine"])
    print("spatial_reasoning aoi2rd_dir task OK!")


def generate_aoi2rd_dir_osm(map, diags, diag_routes, NS, EW, primary_directions, secondary_directions, name_adr_aoi, TASK_FILES):
    aoi2rd_dir = []
    aoi2rd_dir_prompt = []
    reason_ques_num = get_diag_num(diags)
    for cnt, diag in enumerate(diags[:reason_ques_num]):
        # tar_rd_set = set()
        content = json.loads(diag[1]["content"])
        routes = content['routes']
        dir_dis_fin = dir_all_dis(routes, secondary_directions, primary_directions, secondary_dir_to_primary_dirs)[0]
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
        content = json.loads(diag[1]["content"])
        start_aoi_name = content['start_name']
        start_aoi_addr = content['start_addr']
        start_aoi = "{}{}".format(start_aoi_name, start_aoi_addr)
        # 提取"destination"和"by"之间的信息
        end_aoi_name = content['dest_name']
        end_aoi_addr = content['dest_addr']
        end_aoi = "{}{}".format(end_aoi_name, end_aoi_addr)
        routine = 'After starting from {}, you '.format(start_aoi) + ", ".join(diag_routes[cnt]) + ".Finally you arrive at {}.".format(end_aoi)
        if start_aoi not in name_adr_aoi:
            continue
        start_aoi_id = name_adr_aoi[start_aoi]['aoi_id']
        start_aoi_info = map.aois[start_aoi_id]
        result = get_aoi_address(map, start_aoi_info)
        if result is None:
            continue
        aoi_s, rd_name, junc_name, aoi_junc_direction, rd_belong_dir = result
        
        # if 270<angle<360:  #终点在起点西北方向
        if dir_set == {'west', 'north'}:
            if rd_belong_dir in NS:
                tar_dir = 'west'
            elif rd_belong_dir in EW:
                tar_dir = 'north'
        elif dir_set == {'east', 'north'}:  ##终点在起点东北方向
            if rd_belong_dir in NS:
                tar_dir = 'east'
            elif rd_belong_dir in EW:
                tar_dir = 'north'
        elif dir_set == {'east', 'south'}:  # 终点在起点东南方向
            if rd_belong_dir in NS:
                tar_dir = 'east'
            elif rd_belong_dir in EW:
                tar_dir = 'south'
        elif dir_set == {'west', 'south'}:  # 终点在起点西南方向
            if rd_belong_dir in NS:
                tar_dir = 'west'
            elif rd_belong_dir in EW:
                tar_dir = 'south'
        random.shuffle(primary_directions)
        res_dict_aoi = dict(zip(["A", "B", "C", "D"], primary_directions))
        res_dict_prompt_aoi = dict(zip(["A", "B", "C", "D"], primary_directions))
        for k in res_dict_aoi:
            if res_dict_aoi[k] == tar_dir:
                label = k
        res_dict_aoi['answer'] = label
        res_dict_prompt_aoi['answer'] = label
        res_dict_aoi['question'] = "In which direction is {} from {}?".format(end_aoi, rd_name)
        res_dict_prompt_aoi['question'] = routine + "{} is on {}.In which direction is {} from {}?".format(
            start_aoi, rd_name, end_aoi, rd_name)
        aoi2rd_dir.append(res_dict_aoi)
        aoi2rd_dir_prompt.append(res_dict_prompt_aoi)
    aoi2rd_dir_df = pd.DataFrame(data=aoi2rd_dir)
    aoi2rd_dir_prompt_df = pd.DataFrame(data=aoi2rd_dir_prompt)
    aoi2rd_dir_df.to_csv(TASK_FILES["spatial_reasoning_noroute"]["aoi2rd_dir_noroutine"])
    aoi2rd_dir_prompt_df.to_csv(TASK_FILES["spatial_reasoning_route"]["aoi2rd_dir_routine"])
    print("spatial_reasoning aoi2rd_dir task OK!")

def generate_evaluation_task_aoi_type_osm(aoi_dict, category_supported, TASK_FILES):
    aoi2type = []
    type2aoi = []
    aois = list(aoi_dict.keys())
    for aoi_id in aois:
        if aoi_dict[aoi_id]['category'] not in category_supported:
            continue
        aoi2type_session, type2aoi_session = type_gen(aoi_id, aoi_dict, category_supported)
        aoi2type.append(aoi2type_session)
        type2aoi.append(type2aoi_session)
    print('aoi2type type2aoi task OK!')
    save_data(aoi2type, TASK_FILES["urban_semantics"]["aoi2type"])
    save_data(type2aoi, TASK_FILES["urban_semantics"]["type2aoi"])

def generate_evaluation_task_aoi_near(map, aoi_dict, road_dict, TASK_FILES):
    unseen_aois = []
    junc_dict = {}  # 交叉口及附近的AOI
    for aoi_id in aoi_dict:
        aoi_info = map.aois[aoi_id]
        result = get_aoi_address(map, aoi_info)
        if result is None:
            print("No address for start AOI!")
            continue
        aoi_s, rd_name, junc_name, aoi_junc_direction, rd_belong_dir = result
        key = junc_name
        if key not in junc_dict:
            junc_dict[key] = []
        junc_dict[key].append(aoi_id)
    for key in junc_dict:
        aois = junc_dict[key]
        if len(aois) < 2:
            continue
        no_junc = []
        for id in road_dict:
            tar_aois = set(road_dict[id]['aoi_ids']) & set(aoi_dict.keys())
            if len(set(aois) & set(road_dict[id]['aoi_ids'])) == 0 and len(tar_aois) >= 3:
                no_junc = random.sample(list(tar_aois), 3)
                break
        res = no_junc + [aois[1]]
        random.shuffle(res)
        res = [aoi_dict[aoi_id]['name'] for aoi_id in res]
        res_dict = dict(zip(["A", "B", "C", "D"], res))
        for k in res_dict:
            if res_dict[k] == aoi_dict[aois[1]]['name']:
                label = k
        res_dict["question"] = "Which AOI is adjacent to {}?".format(aoi_dict[aois[0]]['name'])
        res_dict["answer"] = label
        unseen_aois.append(res_dict)
    print('city_image aoi_near OK!')
    save_data(unseen_aois, TASK_FILES["city_image"]["aoi_near"])

async def main(args):
    city_map = MAP_DICT[REGION_EXP]
    port = args.port if args.port is not None else MAP_PORT_DICT[REGION_EXP]
    print(f"Loading map {city_map} on port {port}")
    map, process, routing_client = load_map(
        city_map=city_map, 
        cache_dir=MAP_CACHE_PATH, 
        routing_path=ROUTING_PATH, 
        port=port)
    # wait for map to load
    time.sleep(10)
    if not isinstance(args.output_path, str):
        output_path = "evaluate/city_eval/tasks/{}/{}".format(REGION_EXP, args.evaluate_version)
    else:
        output_path = args.output_path

    random.seed(42)
    REGION_EXP_POLYGON = Polygon(REGION_BOUNDARY[REGION_EXP])
    if OSM_REGION == False:
        TASK_FILES = task_files_adaption(EVAL_TASK_MAPPING_v2, output_path)
    else:
        TASK_FILES = task_files_adaption(EVAL_TASK_MAPPING_v1, output_path)
    category_supported = category_mapping()

    all_roads = map.roads
    all_lanes = map.lanes
    all_aois = map.aois
    all_pois = map.pois
    all_juncs = map.juncs
    aoi_message = pd.read_csv(os.path.join(RESOURCE_PATH, "{}_aois.csv".format(REGION_EXP)))
    road_message = pd.read_csv(os.path.join(RESOURCE_PATH, "{}_roads.csv".format(REGION_EXP)))

    if OSM_REGION == False:
        poi_message = pd.read_csv(os.path.join(RESOURCE_PATH, "{}_pois.csv".format(REGION_EXP)))
        poi_dict = {}
        for row in poi_message.itertuples():
            key = row.poi_id
            category = map.get_poi(key)['category']
            name = row.name
            addr = row.Address
            if not isinstance(name, str):
                continue
            poi_dict[key] = {
                "aoi_id": map.get_poi(key)['aoi_id'], "category": category, "name": name, "Address": addr,
                "coord": map.get_poi(key)['shapely_lnglat'].coords[0],
            }
        print(f"define poi_dict done! len:{len(poi_dict)}")
        type_pois = {}  
        for poi_id, poi in poi_dict.items():
            poi_type = poi["category"]
            if poi_type in type_pois:
                type_pois[poi_type].append(poi_id)
            else:
                type_pois[poi_type] = [poi_id]
        name_adr_poi = {}
        for id, poi in poi_dict.items():
            name = poi['name']
            addr = poi['Address']
            flag = "{}({})".format(name, addr)
            if flag not in name_adr_poi:
                name_adr_poi[flag] = poi

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
        if OSM_REGION == True:
            aoi_dict[aoi_id]['category'] = map.get_aoi(aoi_id)['urban_land_use']
        else:
            aoi_dict[aoi_id]['category'] = map.get_aoi(aoi_id)['land_use']
        aoi_dict[aoi_id]['coord'] = eval(row.coords)[0]
        aoi_dict[aoi_id]['name'] = aoi_name
        aoi_dict[aoi_id]['address'] = address
        aoi_dict[aoi_id]['poi_ids'] = poi_ids
        aoi_dict[aoi_id]['aoi_id'] = aoi_id
    print(f"define aoi_dict done! len:{len(aoi_dict)}")
    name_adr_aoi = {}
    for id, aoi in aoi_dict.items():
        name = aoi['name']
        addr = aoi['address']
        flag = "{}({})".format(name, addr)
        if flag not in name_adr_aoi:
            name_adr_aoi[flag] = aoi

    road_dict = {}
    for row in road_message.itertuples():
        road_aois = []
        road_id = row.road_id
        length = all_roads[road_id]['length']
        road_name = row.road_name
        lane_ids = all_roads[road_id]['lane_ids']
        for lane_id in lane_ids:
            aoi_ids = all_lanes[lane_id]['aoi_ids']
            road_aois += aoi_ids
        if road_name == "未知路名" or road_name == "unknown road" or not isinstance(road_name, str):
            continue
        road_dict[road_id] = {}
        road_dict[road_id]['name'] = road_name
        road_dict[road_id]['length'] = length
        road_dict[road_id]['lane_ids'] = lane_ids
        road_dict[road_id]['aoi_ids'] = list(set(road_aois))

    train_data = []
    # EVAL_DATA=True
    with open(os.path.join(TRAIN_DATA_PATH, "citywalk-{}-mock-eval.jsonl".format(REGION_EXP)),
              'r',
              encoding="utf-8") as f:
        for line in f:
            train = json.loads(line)
            train_data.append(train)
    diags = [train_data[i]['diag'] for i in range(len(train_data))]


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
    print(f"Length of diag_records: {len(diag_records)}")
    print(f"Length of diag_routes: {len(diag_routes)}")
        
    print("Start synthesizing evaluation data")
    if OSM_REGION == False:
        print("Start task: road length & road poi")
        generate_evaluation_task_road(map, poi_dict, road_dict, all_roads, all_lanes, all_aois, TASK_FILES)

        print("Start task: road od & road link")
        generate_evaluation_task_road_junc(all_juncs, all_roads, all_lanes, REGION_EXP_POLYGON, TASK_FILES)

        print("Start task: landmark env")
        generate_evalation_task_node(map, poi_message, poi_dict, category_supported, TASK_FILES)

        print("Start task: landmark path")
        await generate_evaluation_task_landmark(map, routing_client, poi_dict, TASK_FILES)

        print("Start task: boundary road")
        generate_evaluation_task_boudary(all_roads, all_lanes, aoi_dict, REGION_EXP_POLYGON, TASK_FILES)

        print("Start task: boundary poi")
        generate_evalation_task_boundary_poi(map, aoi_dict, poi_dict, TASK_FILES)

        print("Start task: districts group")
        generate_evalation_task_districts(map, aoi_message, TASK_FILES)

        print("Start task: districts poi type & districts addr & districts type") 
        generate_districts_poi_type(map, aoi_dict, poi_dict, TASK_FILES)
        generate_evaluation_task_aoi_loc(aoi_dict, TASK_FILES)
        generate_evaluation_task_aoi_type(map, aoi_dict, poi_dict, TASK_FILES)

        print("Start task: node coor & node addr & node type & aoi poi") 
        generate_evaluation_task_poi_loc(map, poi_dict, poi_message, all_pois, category_supported, TASK_FILES)
        generate_evaluation_task_poi_type(aoi_dict, all_aois, all_pois, type_pois, TASK_FILES)
        generate_evaluation_task_poi_aoi(aoi_dict, all_pois, all_aois, poi_dict, TASK_FILES)

        print("Start task: spatial reasoning")
        generate_poi2poi_disdir(diag_records, diag_routes, poi_dict, name_adr_poi, TASK_FILES)
        generate_poi2rd_dis(map, diag_records, diag_routes, NS, EW, primary_directions, secondary_directions, name_adr_poi, poi_dict, TASK_FILES)
        generate_aoi2rd_dis(map, diag_records, diag_routes, NS, EW, primary_directions, name_adr_poi, aoi_dict, secondary_directions, TASK_FILES)
        generate_aoi2aoi_disdir(diag_records, diag_routes, aoi_dict, name_adr_poi, TASK_FILES)
        generate_aoi2poi_disdir(diag_records, diag_routes, poi_dict, aoi_dict, name_adr_poi, TASK_FILES)
        generate_poi2rd_dir(map, diag_records, diag_routes, NS, EW, primary_directions, secondary_directions, name_adr_poi, poi_dict, TASK_FILES)
        generate_aoi2rd_dir(map, diag_records, diag_routes, NS, EW, primary_directions, secondary_directions, name_adr_poi, aoi_dict, TASK_FILES)
    else:
        print("Start task: road length & road aoi")
        generate_evaluation_task_road(map, aoi_dict, road_dict, all_roads, all_lanes, all_aois, TASK_FILES)

        print("Start task: road od & road link")
        generate_evaluation_task_road_junc(all_juncs, all_roads, all_lanes, REGION_EXP_POLYGON, TASK_FILES)

        print("Start task: landmark env")
        generate_evalation_task_node(map, aoi_message, aoi_dict, category_supported, TASK_FILES)

        print("Start task: landmark path")
        await generate_evaluation_task_landmark(map, routing_client, aoi_dict, TASK_FILES)

        print("Start task: boundary road")
        generate_evaluation_task_boudary(all_roads, all_lanes, aoi_dict, REGION_EXP_POLYGON, TASK_FILES)
        
        print("Start task: aoi coor & aoi addr & aoi type & aoi near") 
        generate_evaluation_task_aoi_loc_osm(aoi_dict, TASK_FILES)
        generate_evaluation_task_aoi_type_osm(aoi_dict, category_supported, TASK_FILES)
        generate_evaluation_task_aoi_near(map, aoi_dict, road_dict, TASK_FILES)

        print("Start task: spatial reasoning")
        generate_aoi2rd_dis_osm(map, diag_records, diag_routes, NS, EW, primary_directions, name_adr_aoi, secondary_directions, TASK_FILES)
        generate_aoi2rd_dir_osm(map, diag_records, diag_routes, NS, EW, primary_directions, secondary_directions, name_adr_aoi, TASK_FILES)
        generate_aoi2aoi_disdir_osm(diag_records, diag_routes, aoi_dict, name_adr_aoi, TASK_FILES)


    print("send signal")
    process.send_signal(sig=signal.SIGTERM)
    process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate_version", type=str, default="v2")
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--port", type=str)
    args = parser.parse_args()

    asyncio.run(main(args))
    