import argparse
import logging
from tqdm import tqdm
from typing import Any, Dict, cast
from shapely.strtree import STRtree
import os
import numpy as np
import pymongo
from shapely.geometry import LineString, Polygon
import pyproj
import geojson
import pickle

SAVE_MAP = False  # pickle保存原地图

from map_config import MONGODB_URI

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", "-m", help="map coll", default="")
    parser.add_argument("--map_db", "-db", help="map db", default="")
    parser.add_argument("--net_path", "-net", help="road for names", default="")
    parser.add_argument(
        "--mongo_url",
        "-mongo",
        help="mongo URL",
        default=MONGODB_URI,
    )
    ## moss的："mongodb://moss:Moss202405@mgo.db.fiblab.tech:8635/"
    return parser.parse_args()


args = get_args()


def get_map(uri: str, db: str, col: str) -> Dict[str, Any]:
    client = pymongo.MongoClient(uri)
    m = list(client[db][col].find({}))
    header = None
    juncs = {}
    roads = {}
    lanes = {}
    aois = {}
    pois = {}
    for d in m:
        t = d["class"]
        data = d["data"]
        if t == "lane":
            lanes[data["id"]] = d
        elif t == "junction":
            juncs[data["id"]] = d
        elif t == "road":
            roads[data["id"]] = d
        elif t == "aoi":
            aois[data["id"]] = d
        elif t == "poi":
            pois[data["id"]] = d
        elif t == "header":
            header = d
    return {
        "header": header,
        "junctions": juncs,
        "roads": roads,
        "lanes": lanes,
        "aois": aois,
        "pois": pois,
    }


geos = []
MAP_NAME = args.map
MAP_DB = args.map_db
NET_PATH = args.net_path
MONGO_URL = args.mongo_url
print(MAP_NAME)
map_data = get_map(
    MONGO_URL,
    MAP_DB,
    MAP_NAME,
)
if SAVE_MAP:
    save_path = f"./data/temp/MAPS/{MAP_NAME}.pkl"
    if False and os.path.exists(save_path):
        print(f"{save_path} already exists!")
        import sys
        # sys.exit()
    else:
        print(f"Writing to {save_path}")
        pickle.dump(map_data, open(save_path, "wb"))
projector = pyproj.Proj(map_data["header"]["data"]["projection"])

with open(NET_PATH, "r") as f:
    net = geojson.load(f)
tree_id = 0
tree_id2name = {}
road_geoms = []
for feature in net["features"]:
    if not feature["geometry"]["type"] == "LineString":
        continue
    prop_name = feature["properties"].get("name", "")
    if not prop_name:
        continue
    coords = np.array(feature["geometry"]["coordinates"], dtype=np.float64)
    xy_coords = np.stack(projector(*coords.T[:2]), axis=1)  # (N, 2)
    road_geoms.append(LineString(xy_coords))
    tree_id2name[tree_id] = prop_name
    tree_id += 1
roads_tree = STRtree(road_geoms)

aoi_id2new_name = {}
for aoi_data in map_data["aois"].values():
    aoi_name = aoi_data["data"].get("name", "")
    if aoi_name:
        continue
    else:
        aoi_id = aoi_data["data"]["id"]
        coords_xy = np.array([(c["x"], c["y"]) for c in aoi_data["data"]["positions"]])
        tree_ids = roads_tree.query_nearest(
            geometry=Polygon(coords_xy),
            max_distance=max(Polygon(coords_xy).length, 800),
            return_distance=False,
        )
        if len(tree_ids) > 0:
            aoi_id2new_name[aoi_id] = tree_id2name[tree_ids[0]] + " nearby"

new_aoi_datas = []
for aoi_id, aoi_data in map_data["aois"].items():
    dd = aoi_data["data"]
    if not aoi_id in aoi_id2new_name:
        pass
    else:
        dd["name"] = aoi_id2new_name[aoi_id]
    new_aoi_datas.append(aoi_data)
client = pymongo.MongoClient(MONGO_URL)
map_col = client[MAP_DB][MAP_NAME]
map_col.drop()
BATCH = 1500  # 每次写入mongodb的数量 防止一次写入太多BSONSize报错
map_col.insert_one(map_data["header"])
for data_type in [
    "lanes",
    "roads",
    "junctions",
    "pois",
]:
    dd = list(map_data[data_type].values())
    print(data_type)
    for i in tqdm(range(0, len(dd), BATCH)):
        map_col.insert_many(dd[i : i + BATCH], ordered=False)
# aois在这里
print("aois")
for i in tqdm(range(0, len(new_aoi_datas), BATCH)):
    map_col.insert_many(new_aoi_datas[i : i + BATCH], ordered=False)
