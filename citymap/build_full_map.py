import argparse
import datetime
import logging
import os

import geojson
from pymongo import MongoClient

from mosstool.map.builder import Builder
from mosstool.type import Map
from mosstool.util.format_converter import dict2pb, pb2coll

from map_config import WORKERS, MONGODB_URI, DB

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers", "-ww", help="workers for multiprocessing", type=int, default=128
    )
    return parser.parse_args()


args = get_args()
workers = WORKERS
MONGO_URL = MONGODB_URI
# DB = "srt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
all_bbox = {
    "paris": {
        "min_lon": 2.224225,
        "max_lon": 2.4688,
        "min_lat": 48.8156,
        "max_lat": 48.89652,
    },
    "beijing": {
        "min_lon": 115.613909,
        "max_lon": 117.43,
        "min_lat": 39.592447,
        "max_lat": 40.31976,
    },
    "london": {
        "min_lon": -0.510375,
        "max_lon": 0.314881,
        "min_lat": 51.28676,
        "max_lat": 51.68288,
    },
    "san_francisco": {
        "min_lon": -123.173825,
        "max_lon": -122.29246797,
        "min_lat": 37.63983,
        "max_lat": 37.9134296,
    },    
    "newyork": {
        "min_lon": -74.255591,
        "max_lon": -73.72621,
        "min_lat": 40.496134,
        "max_lat": 40.914816,
    },
}
GEOJSON_PATH = "./data"
def process_city(city, bbox):
    # 加载配置
    logging.info(f"Generating map of {city}")
    lat = (bbox["max_lat"] + bbox["min_lat"]) / 2
    lon = (bbox["max_lon"] + bbox["min_lon"]) / 2
    try:
        with open(f"{GEOJSON_PATH}/roadnet_{city}.geojson", "r") as f:
            net = geojson.load(f)
        with open(f"{GEOJSON_PATH}/aois_{city}.geojson", "r") as f:
            aois = geojson.load(f)
        with open(f"{GEOJSON_PATH}/pois_{city}.geojson", "r") as f:
            pois = geojson.load(f)
        builder = Builder(
            net=net,
            proj_str=f"+proj=tmerc +lat_0={lat} +lon_0={lon}",
            aois=aois,
            pois=pois,
            gen_sidewalk_speed_limit=50 / 3.6,
            road_expand_mode="M",
            enable_tqdm=True,
            workers=workers,
        )
        m = builder.build(city)
        pb = dict2pb(m, Map())
        client = MongoClient(MONGO_URL)
        # 日期
        date_str = datetime.datetime.now().strftime("%y%m%d")
        coll = client[DB][f"map_{city}_20{date_str}"]
        with open(f"{GEOJSON_PATH}/{DB}.map_{city}_20{date_str}.pb", "wb") as f:
            f.write(pb.SerializeToString())
        pb2coll(pb, coll, drop=True)
    except Exception as e:
        print(f"{city} failed!")
        print(e)
        # continue


for city, bbox in all_bbox.items():
    process_city(city, bbox)
