# 根据from_fsq_to_raw_pois.py得到的pkl数据，替换原始地图中的poi
import os
import datetime
import logging
import pickle

import numpy as np
import pyproj
from pymongo import MongoClient
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree
from tqdm import tqdm

from map_config import MONGODB_URI, DB
from mosstool.map._map_util.const import POI_START_ID
from mosstool.type import Map
from mosstool.util.format_converter import coll2pb, dict2pb, pb2coll, pb2dict
PASS_CITIES = [
    "san_francisco",
    "beijing",
    "newyork",
    "paris",
]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# DB = "srt"
client = MongoClient(MONGODB_URI)
db = client[DB]
coll_names = list(db.list_collection_names())


def city2coll(city):
    res = []
    for nn in coll_names:
        # ATTENTION: 已经替换过fsq poi的不管
        if "_fsq" in nn:
            continue
        if city in nn and "map" in nn:
            try:
                # 后六位必须都是数字 如XX241222
                _ = int(nn[-6:])
            except:
                continue
            res.append(nn)
    if len(res) == 0:
        return ""
    else:
        # 日期最靠后的
        return sorted(res, key=lambda x: int(x[-6:]))[-1]

all_bbox = {
    "paris": {
        "min_lon": 2.224225,
        "max_lon": 2.4688,
        "min_lat": 48.8156,
        "max_lat": 48.89652,
    },
    "newyork": {
        "min_lon": -74.255591,
        "max_lon": -73.72621,
        "min_lat": 40.496134,
        "max_lat": 40.914816,
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
    }
}
FSQ_PKL_PATH = "./fsq_pkls"

def main():
    for ii, city in enumerate(
        sorted(list(all_bbox.keys()))
    ):
        if city in PASS_CITIES:
            continue
        print(f"{city} ({ii+1}/{len(all_bbox)})")
        bbox = all_bbox[city]
        lat = (bbox["max_lat"] + bbox["min_lat"]) / 2
        lon = (bbox["max_lon"] + bbox["min_lon"]) / 2
        proj_str = f"+proj=tmerc +lat_0={lat} +lon_0={lon}"
        projector = pyproj.Proj(proj_str)
        # 格式 list["latitude", "longitude", "fsq_category_labels", "name"]
        raw_pois: list = pickle.load(
            open(os.path.join(FSQ_PKL_PATH,f"{city}_raw_pois.pkl"), "rb"),
        )
        pb = Map()
        _city_coll = city2coll(city)
        print(f"fetching from {_city_coll}")
        pb = coll2pb(db[_city_coll], pb)
        # 输出poi id数量
        _orig_poi_ids = set([pid for a in pb.aois for pid in a.poi_ids])
        print("orig:", len(_orig_poi_ids))
        print(pb.header)
        orig_map_dict = pb2dict(pb)
        # 1.清除poi
        orig_map_dict["pois"] = []
        # 2.构建aoi查找树
        aois_dict = {i["id"]: i for i in orig_map_dict["aois"]}
        tree_id_2_aoi_id: dict[int, int] = {}
        aoi_id2poly: dict[int, Polygon] = {}
        for tree_id, (aoi_id, aoi) in enumerate(aois_dict.items()):
            poly = Polygon([(pos["x"], pos["y"]) for pos in aoi["positions"]])
            aoi_id2poly[aoi_id] = poly
            tree_id_2_aoi_id[tree_id] = aoi_id
        aois_tree = STRtree(list(aoi_id2poly.values()))
        # 3.查找并过滤poi
        matched_pois: list = []
        for raw_p in tqdm(raw_pois):
            lat, lon, catg_labels, name = raw_p
            try:
                catg = "|".join(list(catg_labels))
            except:
                catg = ""
            # 没有坐标 跳过
            if np.isnan(lat) or np.isnan(lon):
                continue
            x, y = projector(lon, lat)
            _point = Point(x, y)
            _indexes = aois_tree.query_nearest(_point)
            if len(_indexes) > 0:
                min_index = _indexes[0]
                aoi_id = tree_id_2_aoi_id[min_index]
                aoi_poly = aoi_id2poly[aoi_id]
                # 在内部
                if _point.within(aoi_poly):
                    matched_pois.append(
                        (
                            (
                                x,
                                y,
                                catg,
                                name,
                            ),
                            aoi_id,
                        )
                    )
        # 4.清除原始aoi的poi_ids字段
        for _, aoi in aois_dict.items():
            aoi["poi_ids"] = []
        # 5.构建新的poi
        poi_id = POI_START_ID
        output_pois = []
        for (x, y, catg, name), aoi_id in matched_pois:
            output_pois.append(
                {
                    "id": poi_id,
                    "category": catg,
                    "name": name,
                    "position": {
                        "x": x,
                        "y": y,
                    },
                    "aoi_id": aoi_id,
                }
            )
            aois_dict[aoi_id]["poi_ids"].append(poi_id)
            poi_id += 1
        # 6.替换原来的pois
        orig_map_dict["pois"] = output_pois
        new_pb = dict2pb(orig_map_dict, Map())
        # 输出poi id数量
        _new_poi_ids = set([pid for a in new_pb.aois for pid in a.poi_ids])
        print("new:", len(_new_poi_ids))
        # 输出到mongodb
        # 日期
        date_str = datetime.datetime.now().strftime("%y%m%d")
        pb2coll(new_pb,db[f"map_{city}_fsq_20{date_str}"])


if __name__ == "__main__":
    main()
