"""
打印出map的地点对应名字
"""

from pymongo import MongoClient
from map_config import MONGODB_URI, DB

# DB = "srt"
client = MongoClient(MONGODB_URI)
db = client[DB]
coll_names = list(db.list_collection_names())

# ATTENTION:逻辑是包含该城市名的地图coll名
def city2coll(city):
    res = []
    for nn in coll_names:
        if "fsq" in nn:
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
def city2fsqcoll(city):
    res = []
    for nn in coll_names:
        if "fsq" not in nn:
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


name2city = {
    "伦敦": "london",
    "旧金山": "san_francisco",
    "北京": "beijing",
    "纽约": "newyork",
    "巴黎": "paris",
}
city2name = {v: k for k, v in name2city.items()}
for city in [
    "london",
    "san_francisco",
    "beijing",
    "newyork",
    "paris",
]:
    print(f"{city2name[city]}:")
    print(f"    OSM poi: {DB}.{city2coll(city)}")
    print(f"    fsq poi: {DB}.{city2fsqcoll(city)}")
