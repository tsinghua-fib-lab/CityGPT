import os
import json
import requests
import ast
import argparse
import signal
import time
import multiprocessing
import pandas as pd
import numpy as np
from shapely import Point, Polygon, LineString

from pycitysim.map import Map
from simulate.utils import cal_angle, angle2dir, angle2dir_4, load_map
from config import REGION_EXP, SERVING_IP, RESOURCE_PATH, MAP_CACHE_PATH, ROUTING_PATH, MAP_DICT, MAP_PORT_DICT, OSM_REGION

def get_aoi_lane_direction(map, aoi_info, aoi_point):
    # 获取aoi相对lane的方向
    lane_point_dict = aoi_info['driving_gates'][0]
    lane_point = Point(lane_point_dict['x'], lane_point_dict['y'])
    angle = cal_angle(lane_point, aoi_point)
    aoi_lane_direction = angle2dir_4(angle)
    return aoi_lane_direction

def get_center_point(geometry):
    # 得到aoi的中心点坐标
    if isinstance(geometry, Point):
        return geometry
    elif isinstance(geometry, Polygon):
        return geometry.centroid
    elif isinstance(geometry, LineString):
        return geometry.centroid 
    else:
        return None

def get_next_road_name_pre(map, lane_info):
    # 通过前驱车道获得next_road_names
    center_points = []
    all_lanes = map.lanes
    all_roads = map.roads
    pre_road_names = set()
    pre_lane_id = lane_info['predecessors'][0]['id']
    pre_lane_info = all_lanes[pre_lane_id]
    junc_id = pre_lane_info['parent_id']
    junc_info = map.juncs[junc_id]
    for junc_lane_id in junc_info['lane_ids']:
        junc_lane_info = all_lanes[junc_lane_id]
        center_point = get_center_point(junc_lane_info['shapely_xy'])
        if center_point is not None:
            center_points.append((center_point.x, center_point.y))
        for predecessor in junc_lane_info['predecessors']:
            junc_pre_lane_id = predecessor['id']
            junc_pre_lane_info = all_lanes[junc_pre_lane_id]
            parent_id = junc_pre_lane_info['parent_id']
            if parent_id >= 300000000:
                continue
            if OSM_REGION == True:
                pre_road_name = all_roads[parent_id]['name']
            else:
                pre_road_name = all_roads[parent_id]['external']['name']
            if not pre_road_name: 
            # 如果road_name为空，命名为未知路名。
                pre_road_name = "unknown road"
            pre_road_names.add(pre_road_name)

    if OSM_REGION == True and center_points:
        center_points_array = np.array(center_points)
        mean_x = np.mean(center_points_array[:, 0])
        mean_y = np.mean(center_points_array[:, 1])
        junc_point = Point(mean_x, mean_y)
    elif OSM_REGION == False:
        junc_point = Point(junc_info['external']['center']['x'], junc_info['external']['center']['y'])
    else:
        return None
    
    return junc_point, pre_road_names
   

def get_next_road_name_suc(map, lane_info):
    # 通过后继车道获得next_road_names
    center_points = []
    all_lanes = map.lanes
    all_roads = map.roads
    suc_lane_id = lane_info['successors'][0]['id']
    suc_lane_info = all_lanes[suc_lane_id]
    suc_road_names = set()
    junc_id = suc_lane_info['parent_id']
    junc_info = map.juncs[junc_id]
    for junc_lane_id in junc_info['lane_ids']:
        junc_lane_info = all_lanes[junc_lane_id]
        center_point = get_center_point(junc_lane_info['shapely_xy'])
        if center_point is not None:
            center_points.append((center_point.x, center_point.y))
        for successor in junc_lane_info['successors']:
            junc_suc_lane_id = successor['id']
            junc_suc_lane_info = all_lanes[junc_suc_lane_id]
            parent_id = junc_suc_lane_info["parent_id"]
            if parent_id >= 300000000:
                continue
            if OSM_REGION == True:
                suc_road_name = all_roads[parent_id]['name']
            else:
                suc_road_name = all_roads[parent_id]['external']['name']
            if not suc_road_name: 
            # 如果road_name为空，命名为未知路名
                suc_road_name = "unknown road"
            suc_road_names.add(suc_road_name)

    if OSM_REGION == True and center_points:
        center_points_array = np.array(center_points)
        mean_x = np.mean(center_points_array[:, 0])
        mean_y = np.mean(center_points_array[:, 1])
        junc_point = Point(mean_x, mean_y)
    elif OSM_REGION == False:
        junc_point = Point(junc_info['external']['center']['x'], junc_info['external']['center']['y'])
    else:
        return None
    
    return junc_point, suc_road_names


def get_aoi_address(map, aoi_info):
    all_lanes = map.lanes
    all_roads = map.roads
    selected_next_road_name = None
    if not aoi_info['driving_positions']:
        print("driving_position_none")
        # print(aoi_info)
        return None
    for driving_position in aoi_info['driving_positions']:
        lane_id = driving_position['lane_id']
        lane_info = all_lanes[lane_id]
        if lane_info['predecessors'] or lane_info['successors']:
            aoi_s = driving_position['s']
            road_info = all_roads[lane_info["parent_id"]]
            if OSM_REGION == True:
                road_name = road_info['name']
            else:
                road_name = road_info['external']['name']
            if not road_name: 
            # 如果road_name为空，命名为unknown road
                road_name = "unknown road"
            next_road_name = None
            if lane_info['predecessors']:
                if get_next_road_name_pre(map, lane_info) is not None:
                    junc_point, next_road_names = get_next_road_name_pre(map, lane_info)
                else: 
                    print(f"point none {aoi_info}")
                    return None
                
            else:
                if get_next_road_name_suc(map, lane_info) is not None:
                    junc_point, next_road_names = get_next_road_name_suc(map, lane_info)
                    aoi_s = lane_info['length'] - aoi_s
                else:
                    print(f"point none {aoi_info}")
                    return None

            # 在set中选一个和现有road name不同的pre_road_name即可
            for next_road_name in next_road_names:
                if next_road_name != road_name:
                    selected_next_road_name = next_road_name
                    break
            if selected_next_road_name is None:
                selected_next_road_name = next_road_names.pop()

            junc_name = f"the junction of {next_road_name} and {road_name}"

            # 获得aoi的中心点
            aoi_point = get_center_point(aoi_info['shapely_xy'])
            angle = cal_angle(junc_point, aoi_point)
            aoi_junc_direction = angle2dir(angle)
            aoi_lane_direction = get_aoi_lane_direction(map, aoi_info, aoi_point)
            return aoi_s, road_name, junc_name, aoi_junc_direction, aoi_lane_direction
        
    return None
    
def construct_aoi_address(map, aoi_file_path, min_resolution=50):
    all_aois = map.aois
    aoi_df = pd.read_csv(aoi_file_path)
    try:
        for index, row in aoi_df.iterrows():
            aoi_id = row['aoi_id']
            aoi_info = all_aois[aoi_id]
            result = get_aoi_address(map, aoi_info)
            if result is None:
                print("none")
                continue
            aoi_s, road_name, junc_name, aoi_junc_direction, aoi_lane_direction = result
            aoi_s_rounded = round(aoi_s / min_resolution) * min_resolution
            if aoi_s_rounded == 0:
                aoi_s_final = f'within 50m'
            else:
                aoi_s_final = f'{aoi_s_rounded}m'

            address = f"on the {aoi_lane_direction} side of {road_name}, {aoi_s_final} from the {aoi_junc_direction} corner of {junc_name}"
            aoi_df.loc[index, 'Address'] = address
            
    except Exception as e:
        print(e)
    finally:
        aoi_df.to_csv(aoi_file_path, index=False, encoding='utf-8')

        

def construct_poi_address(map, poi_file_path, aoi_id2name, min_resolution=50):
    all_pois = map.pois
    poi_df = pd.read_csv(poi_file_path)
    try:
        for index, row in poi_df.iterrows():
            poi_id = row['poi_id']
            poi_info = all_pois[poi_id]
            aoi_id = poi_info['aoi_id']
            aoi_info = map.aois[aoi_id]
            result = get_aoi_address(map, aoi_info)
            if result is None:
                print("none")
                continue
            aoi_s, road_name, junc_name, aoi_junc_direction, aoi_lane_direction = result
            aoi_name = aoi_id2name.get(aoi_id, "")
                
            aoi_s_rounded = round(aoi_s / min_resolution) * min_resolution
            if aoi_s_rounded == 0:
                aoi_s_final = f'within 50m'
            else:
                aoi_s_final = f'{aoi_s_rounded}m'
            aoi_name_str = f"inside {aoi_name} " if not pd.isna(aoi_name) and aoi_name else ""
            address = f"{aoi_name_str}on the {aoi_lane_direction} side of {road_name}, {aoi_s_final} from the {aoi_junc_direction} corner of {junc_name}"
            poi_df.loc[index, 'Address'] = address
            
    except Exception as e:
        print(e)
    finally:
        poi_df.to_csv(poi_file_path, index=False, encoding='utf-8')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    args = parser.parse_args()

    city_map = MAP_DICT[REGION_EXP]
    port = args.port if args.port is not None else MAP_PORT_DICT[REGION_EXP]
    m, process, routing_client = load_map(
        city_map=city_map, 
        cache_dir=MAP_CACHE_PATH, 
        routing_path=ROUTING_PATH, 
        port=port)
    time.sleep(10)
    ### 自行构建地址系统
    # min_resolution参数为道路长度最小分辨率
    min_resolution=50

    aoi_file = os.path.join(RESOURCE_PATH, "{}_aois.csv".format(REGION_EXP))
    aoi_data = pd.read_csv(aoi_file)
    aoi_id2name = pd.Series(aoi_data.aoi_name.values, index=aoi_data.aoi_id).to_dict()
    if OSM_REGION == False:
        poi_file = os.path.join(RESOURCE_PATH, "{}_pois.csv".format(REGION_EXP))
        construct_poi_address(m, poi_file, aoi_id2name, min_resolution)
    
    construct_aoi_address(m, aoi_file, min_resolution)

    print("send signal")
    process.send_signal(sig=signal.SIGTERM)
    process.wait()
