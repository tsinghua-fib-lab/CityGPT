import asyncio
import subprocess

from pymongo import MongoClient

from map_config import MONGODB_URI, DB
# DB = "srt"
client = MongoClient(MONGODB_URI)
db = client[DB]
coll_names = list(db.list_collection_names())


def city2coll(city):
    res = []
    for nn in coll_names:
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
GEOJSON_PATH = "./data"
async def main():
    for city in sorted(list(all_bbox.keys())):
        try:
            print(city)
            cmd = f"python3 aoi_name_gen.py --map {city2coll(city)} --map_db {DB} --net_path {f'{GEOJSON_PATH}/roadnet_{city}.geojson'}".split(
                " "
            )
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(city, e)
            continue


if __name__ == "__main__":
    asyncio.run(main())
