import asyncio
import json
import jsonlines
import pandas as pd
import os
import re
import math
import signal
import random
import argparse

from simulate.utils import cal_angle, angle2dir, angle2dir_4, load_map, load_poi_category_dict
from simulate.address_system import get_center_point, get_next_road_name_pre, get_next_road_name_suc
from simulate.player import TextPlayer
from simulate.templates import *
from config import  REGION_EXP, DATA_VERSION, RESOURCE_PATH, OSM_REGION, MAP_CACHE_PATH, ROUTING_PATH, MAP_DICT, MAP_PORT_DICT
from simulate.translate import Name
LANGUAGE = Name(REGION_EXP)


def update_task_count(task_counts, task_type):
    if task_type in task_counts:
        task_counts[task_type] += 1
    else:
        task_counts[task_type] = 1

def poi_category_choose(l1_category_code):
    excluded_ids = ['11', '12', '19', '26', '80','99']
    if l1_category_code in excluded_ids:
        return None

    category_supported = {
        "10": "Cuisine", 
        "11": "Company Business",
        "12": "Organization Groups",
        "13": "Shopping", 
        "14": "Life Services", 
        "16": "Entertainment & Leisure", 
        "18": "Sports & Fitness", 
        "19": "Automotive", 
        "20": "Medical & Healthcare", 
        "21": "Hotels & Guesthouses", 
        "22": "Tourist Attractions", 
        "23": "Cultural Venues", 
        "24": "Education & Schools", 
        "25": "Banking & Finance", 
        "26": "Place Names & Addresses",
        "27": "Infrastructure", 
        "28": "Real Estate & Communities",
        "80": "Interior and Related Facilities",
        "99": "Other"
    }
    poi_category_L1 = category_supported[l1_category_code]
    return poi_category_L1

def construct_dialogues_poi(poi_file, output_file, category_dict_id_name, map, task_counts):
    """构造poi的对话"""
    data = pd.read_csv(poi_file)
    dialogues = []
    count_element = 0

    for index, row in data.iterrows():
        
        poi_name = row['name']
        poi_address = row['Address']
        poi_id = row['poi_id']

        # 判断地址、名字信息是否存在
        if pd.isnull(poi_address) or pd.isnull(poi_name):
            continue

        # poi名字到地址的对话 
        poi_name2addr_session = [
            {"role": "user", "content": poi_name2addr_choose(poi_name)},
            {"role": "assistant", "content": poi_address}
        ]
        dialogues.append(
            {
                "task": "address",
                "id": f"{poi_file}-poi_name2addr-{index}",
                "diag": poi_name2addr_session
            }
        )
        update_task_count(task_counts, "poi_name2addr")


        if OSM_REGION == False:
            # poi一级分类和地址到名字的对话
            category_code = map.pois[poi_id]['category']
            l1_category_code = category_code[:2]
            
            poi_category_L1 = poi_category_choose(l1_category_code)

            if poi_category_L1 is not None:
                category_addr2poi_session = [
                    {"role": "user", "content": category_addr2poi_choose(poi_address, poi_category_L1)},
                    {"role": "assistant", "content": poi_name}
                ]

                dialogues.append(
                    {
                        "task": "address",
                        "id": f"{poi_file}-category_addr2poi-{index}",
                        "diag": category_addr2poi_session
                    }
                )
                update_task_count(task_counts, "category_addr2poi")

            # poi三级分类和地址到名字的对话
            poi_category_L3 = None
            if category_code in category_dict_id_name["L3"]:
                poi_category_L3 = category_dict_id_name["L3"][category_code]
            else:
                print(f"警告：分类代码 {category_code} 在字典中不存在。")
                break 

            session = type_addr2poi_choose(poi_address, l1_category_code, poi_category_L3)
            if session is not None:
                type_addr2poi_session = [
                    {"role": "user", "content": session},
                    {"role": "assistant", "content": poi_name}
                ]
                dialogues.append(
                    {
                        "task": "address",
                        "id": f"{poi_file}-type_addr2poi-{index}",
                        "diag": type_addr2poi_session
                    }
                )
                update_task_count(task_counts, "type_addr2poi")
        else:
            # OSM来源数据的poi分类和地址到名字的对话
            category_string = map.pois[poi_id]['category']
            parts = category_string.split('|')
            category_name = parts[1] if len(parts) > 1 else None
            if category_name is not None:
                category_addr2poi_session = [
                    {"role": "user", "content": category_addr2poi_choose(poi_address, category_name)},
                    {"role": "assistant", "content": poi_name}
            ]

            dialogues.append(
                {
                    "task": "address",
                    "id": f"{poi_file}-category_addr2poi-{index}",
                    "diag": category_addr2poi_session
                }
            )
            update_task_count(task_counts, "category_addr2poi")
        # poi门址地址要素分析
        max_number = 10
        if count_element < max_number:
            count_element += 1
            
            pattern = r"inside (?P<aoi_name>[^,]+) on the (?P<aoi_lane_direction>\w+) side of (?P<road_name>.+?), (?P<aoi_s>(?:within )?\d+m) from the (?P<aoi_junc_direction>\w+) corner of (?P<junc_name>the junction of .+)"
            match = re.match(pattern, poi_address)
            if match:
                road_name = match.group('road_name')
                aoi_lane_direction = match.group('aoi_lane_direction')
                aoi_name = match.group('aoi_name')
                junc_name = match.group('junc_name')
                aoi_junc_direction = match.group('aoi_junc_direction')
                aoi_s = match.group('aoi_s')

                poi_addr_element_session = [
                    {"role": "user", "content": poi_addr_element_text(poi_address)},
                    {"role": "assistant", "content": json.dumps({
                        "road_name": road_name,
                        "aoi_lane_direction": aoi_lane_direction,
                        "aoi_name": aoi_name,
                        "junc_name": junc_name,
                        "aoi_junc_direction": aoi_junc_direction,
                        "aoi_s": f"{aoi_s}"
                    }, ensure_ascii=False)}
                ]
                                
                dialogues.append(
                    {
                        "task": "GeoGLUE",
                        "id": f"{poi_file}-poi_addr_element-{index}",
                        "diag": poi_addr_element_session
                    }
                )
                update_task_count(task_counts, "poi_addr_element")
            else:
                # 对于自行构建地址系统，no match的poi是由于所属aoi没有name，导致格式不符
                print("poi no match")
                print(poi_address)         
        
    with jsonlines.open(output_file, mode="w") as wid:
        for dialogue in dialogues:
            wid.write(dialogue)

def poi_categories_in_aoi(map, poi_ids):
    category_details = {}

    for poi_id in poi_ids:
        if OSM_REGION == True:
            category_string = map.pois[poi_id]['category']
            parts = category_string.split('|')
            category_name = parts[1] if len(parts) > 1 else None
            if category_name is not None:
                if category_name in category_details:
                    category_details[category_name].append(LANGUAGE.get_poi_name(poi_id, map))
                else:
                    category_details[category_name] = [LANGUAGE.get_poi_name(poi_id, map)]
        else:
            category_code = map.pois[poi_id]['category']
            l1_category_code = category_code[:2]
            
            poi_category_L1 = poi_category_choose(l1_category_code)
            if poi_category_L1 is not None:
                if poi_category_L1 in category_details:
                    category_details[poi_category_L1].append(LANGUAGE.get_poi_name(poi_id, map))
                else:
                    category_details[poi_category_L1] = [LANGUAGE.get_poi_name(poi_id, map)]
        
    if not category_details: 
        return None, []

    max_category = max(category_details, key=lambda k: len(category_details[k]))
    max_category_pois = category_details[max_category]

    return max_category, max_category_pois


def calculate_distance(point1, point2):
    distance = math.sqrt((point2.x - point1.x)**2 + (point2.y - point1.y)**2)
    return distance

def get_nearby_pois(map, aoi_info, interested_categories):
    center = get_center_point(aoi_info['shapely_xy'])
    radius = 100 
    category_pois = {}
    nearest_pois = {}

    for category_prefix in interested_categories:
        pois = map.query_pois(center, radius, category_prefix, limit=None)
        if pois:
            if OSM_REGION == False:
                poi_category_L1 = poi_category_choose(category_prefix)
                category_prefix = poi_category_L1

            category_pois[category_prefix] = pois
            nearest_pois[category_prefix] = sorted(pois, key=lambda x: x[1])[0]

    return category_pois, nearest_pois


def get_aoi_with_most_pois(aoi_data, map):
    aoi_poi_counts = {}
    for _, row in aoi_data.iterrows():
        aoi_id = row['aoi_id']
        if 'poi_ids' in map.aois[aoi_id]:
            aoi_poi_counts[aoi_id] = len(map.aois[aoi_id]['poi_ids'])
        else:
            aoi_poi_counts[aoi_id] = 0

    if not aoi_poi_counts:
        return None, 0  
     
    max_aoi_id = max(aoi_poi_counts, key=aoi_poi_counts.get)
    max_aoi_name = LANGUAGE.get_aoi_name(max_aoi_id, map)
    return max_aoi_name, aoi_poi_counts[max_aoi_id]


def construct_dialogues_aoi(aoi_file, output_file, category_dict_id_name, map, task_counts):
    """构造aoi的对话"""
    data = pd.read_csv(aoi_file)
    dialogues = []
    extra = 'nearby'
    round_demical = 4
    count_element = 0
    if OSM_REGION == False:
        interested_categories = ['amenity', 'building', 'leisure']
    else:
        interested_categories = ['10', '13', '14', '16', '18', '19', '20', '21', '22', '23', '24', '25', '27', '28']
    max_pois_aoi_name, max_poi_count = get_aoi_with_most_pois(data, map)
    max_area = -1
    max_aoi_name = None
    min_resolution=50

    # 包含最多poi的aoi对话
    if extra not in max_pois_aoi_name:
        max_pois_aoi_dialogue = {
            "task": "GeoQA",
            "id": f"{aoi_file}-max_pois_aoi",
            "diag": [
                {"role": "user", "content": max_pois_aoi_text()},
                {"role": "assistant", "content": f"{max_pois_aoi_name}"}
            ]
        }
        dialogues.append(max_pois_aoi_dialogue)
        update_task_count(task_counts, "max_pois_aoi")

    for index, row in data.iterrows():
        if extra in str(row['aoi_name']):
            continue
        aoi_name = row['aoi_name']
        aoi_address = row['Address']
        aoi_id = row['aoi_id']
        aoi_info = map.aois[aoi_id]

        if not pd.isnull(aoi_address) and not pd.isnull(aoi_name):
            # aoi名字到地址的对话
            aoi_name2addr_session = [
                {"role": "user", "content": aoi_name2addr_choose(aoi_name)},
                {"role": "assistant", "content": aoi_address}
            ]

            dialogues.append(
                {
                    "task": "address",
                    "id": f"{aoi_file}-aoi_name2addr-{index}",
                    "diag": aoi_name2addr_session
                }
            )
            update_task_count(task_counts, "aoi_name2addr")

            if 'urban_land_use' in map.aois[aoi_id] or 'landuse' in map.aois[aoi_id]:
                # aoi用地类型和地址到名字的对话
                if OSM_REGION == True:
                    landuse_code = map.aois[aoi_id]['urban_land_use']    
                    landuse_name = task_template_urban(landuse_code)
                else:
                    landuse_code = map.aois[aoi_id]['land_use']    
                    landuse_name = task_template(landuse_code)
                if landuse_name is not None:
                    landuse_addr2aoi_session = [
                        {"role": "user", "content": landuse_addr2aoi_choose(aoi_address, landuse_name)},
                        {"role": "assistant", "content": aoi_name}
                    ]
                    dialogues.append(
                        {
                            "task": "address",
                            "id": f"{aoi_file}-landuse_addr2aoi-{index}",
                            "diag": landuse_addr2aoi_session
                        }
                    )
                    update_task_count(task_counts, "landuse_addr2aoi")

            # 哪些路与当前aoi相接
            if aoi_info['driving_positions']:
                road_names = set()
                max_length = 0
                longest_road_info = None
                for driving_position in aoi_info['driving_positions']:
                    lane_id = driving_position['lane_id']
                    lane_info = map.lanes[lane_id]
                    road_info = map.roads[lane_info["parent_id"]]
                    road_name = LANGUAGE.get_road_name(road_info, map)
                    if not road_name: 
                        road_name = "unknown road"
                    road_names.add(road_name)
                    road_length = road_info['length']
                    if road_length > max_length:
                        max_length = road_length
                        longest_road_info = road_info
                road_names = ', '.join(road_names)
                aoi2connected_road_session = [
                    {"role": "user", "content": aoi2connected_road_choose(aoi_name)},
                    {"role": "assistant", "content": road_names}
                ]
                dialogues.append(
                    {
                        "task": "GeoQA",
                        "id": f"{aoi_file}-aoi2connected_road-{index}",
                        "diag": aoi2connected_road_session
                    }
                )
                update_task_count(task_counts, "aoi2connected_road")

                if longest_road_info:
                    # 与当前aoi相接的最长的道路
                    longest_road_name = LANGUAGE.get_road_name(longest_road_info, map)
                    max_length_rounded = round(max_length / min_resolution) * min_resolution
                    if not longest_road_name: 
                        # 最长的道路没有名字
                        continue
                    aoi2longest_connected_road_session = [
                        {"role": "user", "content": aoi2longest_connected_road_choose(aoi_name)},
                        {"role": "assistant", "content": f"{longest_road_name} with {max_length_rounded} meters." }
                    ]
                    dialogues.append(
                        {
                            "task": "GeoQA",
                            "id": f"{aoi_file}-aoi2longest_connected_road-{index}",
                            "diag": aoi2longest_connected_road_session
                        }
                    )
                    update_task_count(task_counts, "aoi2longest_connected_road")

            # aoi面积的对话
            if 'area' in aoi_info:
                area = aoi_info['area']
                rounded_area = round(area)
                aoi_area_session = [
                    {"role": "user", "content": aoi_area_choose(aoi_name)},
                    {"role": "assistant", "content": f"{rounded_area} square meters."}
                ]
                dialogues.append(
                    {
                        "task": "GeoQA",
                        "id": f"{aoi_file}-aoi_area-{index}",
                        "diag": aoi_area_session
                    }
                )
                update_task_count(task_counts, "aoi_area")
                if area > max_area:
                    max_area = area
                    max_aoi_name = aoi_name
        
            # 距离aoi1公里范围内的xx类型的poi
            category_pois, nearest_pois = get_nearby_pois(map, aoi_info, interested_categories)
            # 满足构造对话条件的最少特定xx类型的poi数量
            min_poi_count = 5
            if category_pois is not None:
                for category, pois in category_pois.items():
                    if len(pois) > min_poi_count:
                        poi_names = ', '.join([poi[0]['name'] for poi in pois])
                        aoi_range_category2poi_session = [
                            {"role": "user", "content": aoi_range_category2poi_choose(aoi_address, category)},
                            {"role": "assistant", "content": poi_names}
                        ]
                        dialogues.append(
                            {
                                "task": "GeoQA",
                                "id": f"{aoi_file}-aoi_range_category2poi-{index}",
                                "diag": aoi_range_category2poi_session
                            }
                        )
                        update_task_count(task_counts, "aoi_range_category2poi")
                        # 距离aoi最近的L1类型的poi
                        nearest_poi = nearest_pois[category]
                        aoi_category2nearest_poi_session = [
                            {"role": "user", "content": aoi_category2nearest_poi_choose(aoi_address, category)},
                            {"role": "assistant", "content": nearest_poi[0]['name']}
                        ]
                        dialogues.append(
                            {
                                "task": "GeoQA",
                                "id": f"{aoi_file}-aoi_category2nearest_poi-{index}",
                                "diag": aoi_category2nearest_poi_session
                            }
                        )
                        update_task_count(task_counts, f"aoi_category2nearest_poi")
            else:
                print(f"No POIs found within 1000 meters for any category at {aoi_name}.") 


        if not pd.isnull(aoi_address):   
            # aoi地址到地理坐标的对话
            aoi_coords = get_center_point(aoi_info['shapely_lnglat'])
            lng, lat = aoi_coords.x, aoi_coords.y
            lng, lat = round(lng, round_demical), round(lat, round_demical)
            aoi_addr2coords_session = [
                {"role": "user", "content": aoi_addr2coords_choose(aoi_address)},
                {"role": "assistant", "content": f"({lng}, {lat})"}
            ]
            dialogues.append(
                {
                    "task": "address",
                    "id": f"{aoi_file}-aoi_addr2coords-{index}",
                    "diag": aoi_addr2coords_session
                }
            )
            update_task_count(task_counts,"aoi_addr2coords")

            # aoi地理坐标到地址的对话
            aoi_coords2addr_session = [
                {"role": "user", "content": aoi_coords2addr_choose(lng, lat)},
                {"role": "assistant", "content": aoi_address}
            ]
            dialogues.append(
                {
                    "task": "address",
                    "id": f"{aoi_file}-aoi_coords2addr-{index}",
                    "diag": aoi_coords2addr_session
                }
            )
            update_task_count(task_counts, "aoi_coords2addr")

        # junc地址到地理坐标的对话
        if aoi_info['driving_positions']:
            flag=0
            first_driving_position = aoi_info['driving_positions'][0]
            lane_id = first_driving_position['lane_id']
            lane_info = map.lanes[lane_id]
            road_info = map.roads[lane_info["parent_id"]]
            road_name = LANGUAGE.get_road_name(road_info['id'], map)
            if not road_name: 
                road_name = "unknown road"

            next_road_name = None
            selected_next_road_name = None
            if lane_info['predecessors']:
                junc_point, next_road_names = get_next_road_name_pre(map, lane_info)
                if lane_info['successors']:
                    suc_junc_point, suc_next_road_names = get_next_road_name_suc(map, lane_info)
                    flag=1
                    
            elif lane_info['successors']:
                junc_point, next_road_names = get_next_road_name_suc(map, lane_info)
            else:
                continue

            for next_road_name in next_road_names:
                if next_road_name != road_name:
                    selected_next_road_name = next_road_name
                    break
            if selected_next_road_name is None:
                selected_next_road_name = next_road_names.pop()
            
            junc_name = f"the junction of {next_road_name} and {road_name}"
            junc_lng, junc_lat = map.xy2lnglat(junc_point.x, junc_point.y)
            junc_lng, junc_lat = round(junc_lng, round_demical), round(junc_lat, round_demical)
            junc_addr2coords_session = [
                {"role": "user", "content": junc_addr2coords_choose(junc_name)},
                {"role": "assistant", "content": f"({junc_lng}, {junc_lat})"}
            ]
            dialogues.append(
                {
                    "task": "address",
                    "id": f"{aoi_file}-junc_addr2coords-{index}",
                    "diag": junc_addr2coords_session
                }
            )
            update_task_count(task_counts, "junc_addr2coords")

            # junc地理坐标到地址的对话
            junc_coords2addr_session = [
                {"role": "user", "content": junc_coords2addr_choose(junc_lng, junc_lat)},
                {"role": "assistant", "content": junc_name}
            ]
            dialogues.append(
                {
                    "task": "address",
                    "id": f"{aoi_file}-junc_coords2addr-{index}",
                    "diag": junc_coords2addr_session
                }
            )
            update_task_count(task_counts, "junc_coords2addr")

            # 构造相邻路口距离的对话
            if flag==1:
                suc_selected_next_road_name = None
                for suc_next_road_name in suc_next_road_names:
                    if suc_next_road_name != road_name:
                        suc_selected_next_road_name = suc_next_road_name
                        break
                if suc_selected_next_road_name is None:
                    suc_selected_next_road_name = suc_next_road_names.pop()
                
                suc_junc_name = f"the junction of {road_name} and {suc_next_road_name}"               
                distance = calculate_distance(junc_point, suc_junc_point)
                distance_rounded = round(distance / min_resolution) * min_resolution
                junc_distance_session = [
                    {"role": "user", "content":junc_distance_choose(junc_name, suc_junc_name)},
                    {"role": "assistant", "content": f"{distance_rounded} meters"}
                ]
                dialogues.append(
                    {
                        "task": "address",
                        "id": f"{aoi_file}-junc_distance-{index}",
                        "diag": junc_distance_session
                    }
                )
                update_task_count(task_counts, "junc_distance")

                # 构造相邻路口方向的对话
                angle = cal_angle(junc_point, suc_junc_point)
                junc_direction = angle2dir(angle)
                junc_direction_session = [
                    {"role": "user", "content": junc_direction_choose(junc_name, suc_junc_name)},
                    {"role": "assistant", "content": f"{junc_direction}"}
                ]
                dialogues.append(
                    {
                        "task": "address",
                        "id": f"{aoi_file}-junc_direction-{index}",
                        "diag": junc_direction_session
                    }
                )
                update_task_count(task_counts, "junc_direction")

        if 'poi_ids' in aoi_info and aoi_info['poi_ids']:
            poi_ids = aoi_info['poi_ids']
            max_category, max_category_pois = poi_categories_in_aoi(map, poi_ids)
            if max_category is not None:
                if not pd.isnull(aoi_name):
                    # aoi中有哪些xx类型(出现最多)的poi的对话
                    max_category_poi_names = ', '.join(max_category_pois)
                    aoi_category2poi_session = [
                            {"role": "user", "content": aoi_category2poi_choose(aoi_name, max_category)},
                            {"role": "assistant", "content": f"They are {max_category_poi_names}."}
                        ]
                    dialogues.append(
                        {
                            "task": "GeoQA",
                            "id": f"{aoi_file}-aoi_category2poi-{index}-{max_category}",
                            "diag": aoi_category2poi_session
                        }
                    )
                    update_task_count(task_counts, "aoi_category2poi")

                    # aoi_landuse类型到所包含poi类型的对话
                    if OSM_REGION == True:
                        landuse_code = aoi_info['urban_land_use']
                        landuse_name = task_template_urban(landuse_code)
                    else:
                        landuse_code = aoi_info['land_use']
                        landuse_name = task_template(landuse_code)
                    if landuse_name is not None:
                        aoi_laneuse2poi_category_session = [
                            {"role": "user", "content": aoi_landuse2poi_category_choose(aoi_name, landuse_name)},
                            {"role": "assistant", "content": max_category}
                        ]
                        dialogues.append(
                            {
                                "task": "address",
                                "id": f"{aoi_file}-aoi_landuse2poi_category-{index}",
                                "diag": aoi_laneuse2poi_category_session
                            }
                        )
                        update_task_count(task_counts, "aoi_landuse2poi_category")

                        # 所包含poi类型到aoi_landuse类型的对话
                        poi_category2aoi_landuse_session = [
                            {"role": "user", "content": poi_category2aoi_landuse_choose(max_category, aoi_name)},
                            {"role": "assistant", "content": landuse_name}
                        ]
                        dialogues.append(
                            {
                                "task": "address",
                                "id": f"{aoi_file}-poi_category2aoi_landuse-{index}",
                                "diag": poi_category2aoi_landuse_session
                            }
                        )
                        update_task_count(task_counts, "poi_category2aoi_landuse")

                # 哪些poi与这个poi相邻
                if len(poi_ids) > 1:
                    valid_poi_ids = [poi_id for poi_id in poi_ids if pd.notna(LANGUAGE.get_poi_name(poi_id, map))]
                    chosen_poi_id = random.choice(valid_poi_ids)
                    chosen_poi_name = LANGUAGE.get_poi_name(chosen_poi_id, map)  

                    adjacent_poi_names = [LANGUAGE.get_poi_name(poi_id, map) for poi_id in valid_poi_ids if poi_id != chosen_poi_id and pd.notna(LANGUAGE.get_poi_name(poi_id, map))]
                    adjacent_poi_names_str = ', '.join(adjacent_poi_names)
                    
                    poi2adjacency_pois_session = [
                        {"role": "user", "content": poi2adjacent_pois_choose(chosen_poi_name)},
                        {"role": "assistant", "content": f"{adjacent_poi_names_str}."}
                    ]
                    
                    dialogues.append(
                        {
                            "task": "GeoQA",
                            "id": f"{aoi_file}-poi2adjacency_pois-{index}",
                            "diag": poi2adjacency_pois_session
                        }
                    )
                    update_task_count(task_counts, "poi2adjacency_pois")
            
        # aoi门址地址元素分析，仅适用于自行构建地址系统
        if not pd.isnull(aoi_address):
            max_number = 10
            if count_element < max_number:
                count_element += 1
                pattern = r"on the (?P<aoi_lane_direction>\w+) side of (?P<road_name>.+?), (?P<aoi_s>(?:within )?\d+m) from the (?P<aoi_junc_direction>\w+) corner of (?P<junc_name>the junction of .+)"
                match = re.match(pattern, aoi_address)

                if match:
                    aoi_lane_direction = match.group('aoi_lane_direction')
                    aoi_junc_direction = match.group('aoi_junc_direction')
                    aoi_s = match.group('aoi_s')
                    

                    aoi_addr_element_session = [
                        {"role": "user", "content": aoi_addr_element_text(aoi_address)},
                        {"role": "assistant", "content": json.dumps({
                            "road_name": road_name,
                            "aoi_lane_direction": aoi_lane_direction,
                            "junc_name": junc_name,
                            "aoi_junc_direction": aoi_junc_direction,
                            "aoi_s": f"{aoi_s}"
                        }, ensure_ascii=False)}
                    ]
                    dialogues.append(
                        {
                            "task": "GeoGLUE",
                            "id": f"{aoi_file}-aoi_addr_element-{index}",
                            "diag": aoi_addr_element_session
                        }
                    )
                    update_task_count(task_counts, "aoi_addr_element")
                else:
                    print("aoi no match")
                    print(aoi_address)

    # aoi最大面积的对话
    max_aoi_area_session = [
        {"role": "user", "content": max_aoi_area_text()},
        {"role": "assistant", "content": f"{max_aoi_name}"}
    ]
    dialogues.append(
        {
            "task": "GeoQA",
            "id": f"{aoi_file}-max_aoi_area-{index}",
            "diag": max_aoi_area_session
        }
    )
    update_task_count(task_counts, "max_aoi_area")

    with jsonlines.open(output_file, mode="a") as wid:
        for dialogue in dialogues:
            wid.write(dialogue)

async def construct_dialogues_routes(input_file_name, output_file_name, map, routing_client, min_road_length, task_counts):
        tasks = []
        dialogues = []
        with jsonlines.open(input_file_name) as fid:
            for i, task_info in enumerate(fid):
                tasks.append(task_info)

        for index, task_info in enumerate(tasks):
            init_aoi_id = task_info["init_aoi"]
            region_exp = task_info["region"]
            init_poi_id = task_info["init_poi"]
            dest_poi_id = task_info["poi_id"]
            dest_aoi_id = map.pois[dest_poi_id]['aoi_id']
            init_poi_name = LANGUAGE.get_poi_name(init_poi_id, map)
            dest_poi_name = LANGUAGE.get_poi_name(dest_poi_id, map)
            init_poi_addr = LANGUAGE.get_poi_address(init_poi_id)
            dest_poi_addr = LANGUAGE.get_poi_address(dest_poi_id)
            init_name = init_poi_name
            dest_name = dest_poi_name
            env = TextPlayer(map, routing_client, init_aoi_id, min_road_length, region_exp, init_poi_id)
            route = await env.get_driving_route(dest_aoi_id)

            road_list = []
            infos = []
            for road_id in route["road_ids"]:
                road_info = env._city_map.get_road(road_id)
                lane_info = env._city_map.get_lane(road_info["lane_ids"][0])
                road_list = env.road_info_collect(road_info, lane_info, road_list)

            for j in range(len(road_list)):
                current_road = road_list[j]
                infos.append("walk along {} for {} meters {}".format(current_road[0], int(current_road[1]/100)*100, current_road[3]))
                if j < len(road_list) - 1:
                    next_road = road_list[j + 1]
                    basic_direction = next_road[3].split(" ")[-1]
                    junction_name = " the junction of {} and {}".format(current_road[0], next_road[0])
                    
                    infos.append("then go {} towards{}".format(basic_direction, junction_name))
            
            if infos:
                info_text = ", ".join(infos)
                sentences = info_text.split(', ')
                # 打乱路径顺序的对话
                random.shuffle(sentences)
                shuffled_text = ', '.join(sentences)
                scramble_task_session = [
                    {"role": "user", "content": scrambled_task_route_choose(init_name, dest_name, shuffled_text)},
                    {"role": "assistant", "content": info_text}
                ]
                dialogues.append(
                        {
                            "task": "route",
                            "id": f"{input_file_name}-scrambled_task_route-{index}",
                            "diag": scramble_task_session
                        }
                )
                update_task_count(task_counts, "scrambled_task_route")

                # 缺失路径步骤的对话
                missing_index = random.randint(0, len(infos) - 1)
                missing_step = infos.pop(missing_index)
                info_text_updated = ", ".join(infos)

                lacking_task_session = [
                    {"role": "user", "content": lacking_task_route_choose(init_name, dest_name, info_text_updated)},
                    {"role": "assistant", "content": missing_step}
                ]
                dialogues.append(
                    {
                        "task": "route",
                        "id": f"{input_file_name}-lacking_task_route-{index}",
                        "diag": lacking_task_session
                    }
                )
                update_task_count(task_counts, "lacking_task_route")
            else:
                print("init_name: ", init_name)
                print("dest_name: ", dest_name)

        with jsonlines.open(output_file_name, mode="a") as wid:
            for dialogue in dialogues:
                wid.write(dialogue)


def construct_dialogues_road(road_file, output_file_name, map, task_counts):
    """构建与道路相关的对话"""
    # 某条路是否连接两个aoi_name
    dialogues = []
    data = pd.read_csv(road_file)
    count = 0
    for index, row in data.iterrows():
        road_id = row['road_id']
        road_name = row['road_name']
        if pd.notna(road_name) and road_name.strip():
            aoi_names = set()
            driving_lane_ids = map.roads[road_id]['driving_lane_ids']
            for lane_id in driving_lane_ids:
                lane_info = map.lanes[lane_id]
                aoi_ids = lane_info['aoi_ids']
                for aoi_id in aoi_ids:
                    aoi_name = LANGUAGE.get_aoi_name(aoi_id, map)
                    if not pd.isnull(aoi_name):  
                        aoi_names.add(aoi_name)
            
            # 如果连接的AOI名称数量大于等于2
            if len(aoi_names) >= 2:
                for chosen_aoi_name in aoi_names:
                    other_aoi_names = [name for name in aoi_names if name != chosen_aoi_name]
                    for other_aoi_name in other_aoi_names:
                        road2connected_2aois_session = [
                            {"role": "user", "content": road2connected_2aois_choose(chosen_aoi_name, road_name)},
                            {"role": "assistant", "content": other_aoi_name}
                        ]
                        dialogues.append(
                            {
                                "task": "GeoQA",
                                "id": f"{road_file}-road2connected_2aois-{count}",
                                "diag": road2connected_2aois_session
                            }
                        )
                        count += 1
    task_counts["road2connected_2aois"] = count
    with jsonlines.open(output_file_name, mode="a") as wid:
            for dialogue in dialogues:
                wid.write(dialogue)

def extract_task_type_from_id(id_string):
    """从id字符串中提取任务类型"""
    match = re.search(r"-(\w+)-\d+$", id_string)
    if match:
        return match.group(1)
    return None

def selected_data(input_file, output_file, max_samples_per_category=1000):
    """对数据进行随机抽取以减少数据量"""
    categories = {}
    # excluded_types中为GeoQA总结出的对话，需在评测中体现，暂时去除
    excluded_types = {'aoi_category2poi', 'max_pois_aoi', 'aoi_range_category2poi', 'aoi_category2nearest_poi', 'aoi2connected_road', 'aoi2longest_connected_road', 'poi2adjacency_pois', 'aoi_area', 'max_aoi_area', 'road2connected_2aois'}
    # 读取数据并按从ID中解析出的任务类型分组
    with jsonlines.open(input_file) as reader:
        for obj in reader:
            task_type = extract_task_type_from_id(obj['id'])
            if task_type and task_type not in excluded_types:
                if task_type not in categories:
                    categories[task_type] = []
                categories[task_type].append(obj)
    
    # 对每个类别进行随机抽样
    sampled_data = []
    for category, items in categories.items():
        if len(items) > max_samples_per_category:
            sampled_data.extend(random.sample(items, max_samples_per_category))
        else:
            sampled_data.extend(items)
    
    # 写入抽样后的数据到新文件
    with jsonlines.open(output_file, mode='w') as writer:
        writer.write_all(sampled_data)

    # 打印统计结果
    print("任务类型统计结果：")
    for category, items in categories.items():
        print(f"{category}: {len(items) if len(items) <= max_samples_per_category else max_samples_per_category}")

            
async def main(args):
    city_map = MAP_DICT[REGION_EXP]
    port = args.port if args.port is not None else MAP_PORT_DICT[REGION_EXP]
    m, process, routing_client = load_map(
        city_map=city_map, 
        cache_dir=MAP_CACHE_PATH, 
        routing_path=ROUTING_PATH, 
        port=port)

    random.seed(42)
    task_counts = {}  
    min_road_length = 100

    if args.input_pois_file == "":
        poi_file = os.path.join(RESOURCE_PATH, "{}_pois.csv".format(REGION_EXP))
    else:
        poi_file = args.input_pois_file
    if args.input_aois_file == "":
        aoi_file = os.path.join(RESOURCE_PATH, "{}_aois.csv".format(REGION_EXP))
    else:
        aoi_file = args.input_aois_file
    if args.input_roads_file == "":
        road_file = os.path.join(RESOURCE_PATH, "{}_roads.csv".format(REGION_EXP))
    else:
        road_file = args.input_roads_file
    
    category_dict_name_id, category_dict_id_name = load_poi_category_dict()
    construct_dialogues_poi(poi_file, args.output_file, category_dict_id_name, m, task_counts)
    construct_dialogues_aoi(aoi_file, args.output_file, category_dict_id_name, m, task_counts)
    await construct_dialogues_routes(args.input_file, args.output_file, m, routing_client, min_road_length, task_counts)
    construct_dialogues_road(road_file, args.output_file, m, task_counts)
    if args.random_selected:
        # 每类对话最多抽取的样本数 
        max_samples_per_category = 1000
        selected_data(args.output_file, args.selected_output_file, max_samples_per_category)

    # 输出任务统计结果
    print("任务类型统计结果：")
    for task_type, count in task_counts.items():
        print(f"{task_type}: {count}")

    print("send signal")
    process.send_signal(sig=signal.SIGTERM)
    process.wait()



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="simulate/tasks/input_citywalk_wangjing-v11.1-eng-chi.jsonl")
    parser.add_argument("--output_file", default="simulate/examples/address-wangjing-v11.1-eng-chi-makesure.jsonl")
    parser.add_argument("--selected_output_file", default="simulate/examples/address-wangjing-v11.1-eng-chi.jsonl")
    parser.add_argument("--input_pois_file", default="")
    parser.add_argument("--input_aois_file", default="")
    parser.add_argument("--input_roads_file", default="")
    parser.add_argument("--random_selected", action="store_true", help="控制是否进行随机抽取的参数，不设置即不抽取，设置即抽取")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()

    asyncio.run(main(args))
