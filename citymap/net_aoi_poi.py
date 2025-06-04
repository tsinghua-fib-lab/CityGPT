import logging
import os

from mosstool.map.osm import Building, PointOfInterest, RoadNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
from map_config import all_bbox, city_2_wiki_name, PROXIES

GEOJSON_PATH = "./data"
if not os.path.exists(GEOJSON_PATH):
    os.makedirs(GEOJSON_PATH, exist_ok=True)

for city_name, bbox in all_bbox.items():
    try:
        min_lon, min_lat = bbox["min_lon"], bbox["min_lat"]
        max_lon, max_lat = bbox["max_lon"], bbox["max_lat"]
        proj_str = f"+proj=tmerc +lat_0={(max_lat+min_lat)/2} +lon_0={(max_lon+min_lon)/2}"
        wikipedia_name=city_2_wiki_name[city_name]
        ## 路网
        rn = RoadNet(
            max_latitude=max_lat,
            min_latitude=min_lat,
            max_longitude=max_lon,
            min_longitude=min_lon,
            proj_str=proj_str,
            wikipedia_name=wikipedia_name,
            proxies=PROXIES,
        )
        rn.create_road_net(
            output_path=os.path.join(GEOJSON_PATH, f"roadnet_{city_name}.geojson")
        )
        ## AOI
        building = Building(
            max_latitude=max_lat,
            min_latitude=min_lat,
            max_longitude=max_lon,
            min_longitude=min_lon,
            proj_str=proj_str,
            wikipedia_name=wikipedia_name,
            proxies=PROXIES,
        )
        building.create_building(
            output_path=os.path.join(GEOJSON_PATH, f"aois_{city_name}.geojson")
        )
        # POI
        pois = PointOfInterest(
            max_latitude=max_lat,
            min_latitude=min_lat,
            max_longitude=max_lon,
            min_longitude=min_lon,
            wikipedia_name=wikipedia_name,
            proxies=PROXIES,
        )
        pois.create_pois(
            output_path=os.path.join(GEOJSON_PATH, f"pois_{city_name}.geojson")
        )
    except Exception as e:
        print(city_name,e)
        continue
