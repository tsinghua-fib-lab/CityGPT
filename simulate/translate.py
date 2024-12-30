import os
import pandas as pd
from tqdm import tqdm
from pycitysim.map import Map
from openai import OpenAI
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MultiLabelBinarizer

from config import REGION_EXP, RESOURCE_PATH, OSM_REGION

class Name:
    def __init__(self, region_exp="wudaokou_small"):
        POI_FILE_PATH = "{}_pois.csv".format(region_exp)
        AOI_FILE_PATH = "{}_aois.csv".format(region_exp)
        ROAD_FILE_PATH = "{}_roads.csv".format(region_exp)
        
        self.road_data = pd.read_csv(os.path.join(RESOURCE_PATH, ROAD_FILE_PATH))
        self.road_id2name = pd.Series(self.road_data.road_name.values, index = self.road_data.road_id).to_dict()
        self.aois_data = pd.read_csv(os.path.join(RESOURCE_PATH, AOI_FILE_PATH))
        self.aoi_id2name = pd.Series(self.aois_data.aoi_name.values, index=self.aois_data.aoi_id).to_dict()
        self.aoi_id2addr = pd.Series(self.aois_data.Address.values, index = self.aois_data.aoi_id).to_dict()
        if OSM_REGION == False:
            self.pois_data = pd.read_csv(os.path.join(RESOURCE_PATH, POI_FILE_PATH))
            self.poi_id2name = pd.Series(self.pois_data.name.values, index = self.pois_data.poi_id).to_dict()
            self.poi_id2addr = pd.Series(self.pois_data.Address.values, index = self.pois_data.poi_id).to_dict()

    def get_poi_name(self, poi_id, map):
        try:
            poi_name = map.pois[poi_id]['name']
        except:
            poi_name = ""
        return poi_name 
    
    def get_road_name(self, road_id, map):
        try:
            if OSM_REGION == True:
                road_name = map.roads[road_id]['name']
            else:
                road_name = map.roads[road_id]['external']['name']
        except:
            road_name = ""
        return road_name
    
    def get_aoi_name(self, aoi_id, map):
        try:
            aoi_name = map.aois[aoi_id]['name']
        except:
            aoi_name = ""
        return aoi_name

    def get_poi_address(self, poi_id):
        poi_addr = self.poi_id2addr.get(poi_id, "")
        return poi_addr

    def get_aoi_address(self, aoi_id):
        aoi_addr = self.aoi_id2addr.get(aoi_id, "")
        return aoi_addr 
