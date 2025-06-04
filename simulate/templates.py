import json
import random
random.seed(42)

from config import EVAL_DATA, REGION_EXP
from simulate.translate import Name
LANGUAGE = Name(region_exp=REGION_EXP)

def task_template_urban(land_use):
    landuse_dict = {
        "E3":"OtherNon-construction", "R":"Residential", "S4":"TrafficStation&Park", "A4":"Sports", "B31":"Entertainment", "B1":"CommercialFacilities", "U9":"OtherPublicFacilities", "A3":"Education","G1":"Park&GreenLand","B":"CommercialService&IndustryFacilities","B32":"Resort&Fitness","B13":"Restaurant&Bar","A9":"ReligiousFacilities","A5":"Hospital"
        }
    category = landuse_dict.get(land_use, None)
    return category

# address数据集
# Template for poi_name2addr
def poi_name2addr_choose(poi_name):
    templates = [
            f"Do you know where I can find {poi_name}?",
            f"I am curious about the location of {poi_name}.",
            f"Could you point out where {poi_name} is situated?"
        ]
    return random.choice(templates)

# Template for category_addr2poi
def category_addr2poi_choose(poi_address, poi_category_L1):
    templates = [
            f"I'm in {poi_address}, please help me find a nearby {poi_category_L1} POI.",
            f"I am currently located at {poi_address}, could you please assist me in finding a nearby {poi_category_L1} POI?",
            f"Could you assist me in finding a {poi_category_L1} POI in {poi_address}?"
        ]
    return random.choice(templates)


# Text for poi_addr_element
def poi_addr_element_text(poi_address):
    text = f"Could you please identify the components of this POI address: {poi_address}?"
    return text

# Text for max_pois_aoi
def max_pois_aoi_text():
    text = "Which AOI contains the highest number of POI?"
    return text

# Template for aoi_name2addr
def aoi_name2addr_choose(aoi_name):
    templates = [
            f"Can you tell me the address of {aoi_name}?",
            f"I'm curious about the location of {aoi_name}. Can you help me out?",
            f"Where is {aoi_name} located?"
        ]
    return random.choice(templates)

# Template for landuse_addr2aoi
def landuse_addr2aoi_choose(aoi_address, landuse_name):
    templates = [
            f"I'm in {aoi_address}, please help me find a nearby {landuse_name} AOI.",
            f"Hey, I'm currently in {aoi_address}. Can you help me locate a nearby {landuse_name} AOI?",
            f"Could you help me find a {landuse_name} AOI close to {aoi_address}?"
        ]
    return random.choice(templates)

# Template for aoi2connected_road
def aoi2connected_road_choose(aoi_name):
    templates = [
            f"Can you tell me the roads that {aoi_name} is connected to?",
            f"I need to know the roads that {aoi_name} is on, can you help?",
            f"Would you be able to tell me which roads {aoi_name} is adjacent to?"
        ]
    return random.choice(templates)

# Template for aoi2longest_connected_road
def aoi2longest_connected_road_choose(aoi_name):
    templates = [
        f"Which road among those connected to {aoi_name} is the longest?",
        f"Which road is the longest in the vicinity of {aoi_name}",
        f"Identify the longest road directly linked to {aoi_name}."
    ]
    return random.choice(templates)

# Template for aoi_area
def aoi_area_choose(aoi_name):
    templates = [
            f"What is the total area of {aoi_name}?",
            f"What's the size of {aoi_name} in square units?",
            f"Can you tell me the area of {aoi_name}?"
        ]
    return random.choice(templates)

# Template for aoi_range_category2poi
def aoi_range_category2poi_choose(aoi_address, category):
    templates = [
        f"Which POIs in the category {category} are located within a 100m radius of {aoi_address}?",
        f"Can you find POIs in {category} within 100m of {aoi_address}?",
        f"Give me a list of {category} POIs that are 100m or less from {aoi_address}."
    ]
    return random.choice(templates)

# Template for aoi_category2nearest_poi
def aoi_category2nearest_poi_choose(aoi_address, category):
    templates = [
            f"What is the nearest POI in the category {category} to {aoi_address}?",
            f"Can you tell me the closest {category} POI to {aoi_address}?",
            f"Can you identify the proximal {category} POI relative to {aoi_address}?"
        ]
    return random.choice(templates)

# Template for aoi_addr2coords
def aoi_addr2coords_choose(aoi_address):
    templates = [
            f"Here is the address: {aoi_address}, please tell me the geographic coordinates.",
            f"I need the geographic coordinates for the place {aoi_address}. Can you help?",
            f"What are the geographic coordinates for the place {aoi_address}?"
        ]
    return random.choice(templates)

# Template for aoi_coords2addr
def aoi_coords2addr_choose(lng, lat):
    templates = [
        f"Given the longitude and latitude: ({lng}, {lat}), can you determine the corresponding AOI address?",
        f"Can you please provide the AOI address associated with the longitude and latitude ({lng}, {lat})?",
        f"What is the AOI address near the location ({lng}, {lat})?"
    ]
    return random.choice(templates)

# Template for junc_addr2coords
def junc_addr2coords_choose(junc_name):
    templates = [
            f"For the address provided: {junc_name}, could you supply the corresponding geographic coordinates?",
            f"What are the geographic coordinates for {junc_name}?",
            f"Could you provide the geographic coordinates associated with {junc_name}?"
        ]
    return random.choice(templates)

# Template for junc_coords2addr
def junc_coords2addr_choose(junc_lng, junc_lat):
    templates = [
        f"Given the specified longitude and latitude: ({junc_lng}, {junc_lat}), can you identify the associated junction address?",
        f"Can you determine the junction address at the location ({junc_lng}, {junc_lat})?",
        f"What is the address of the junction at ({junc_lng}, {junc_lat})?"
        ]
    return random.choice(templates)

# Template for junc_distance
def junc_distance_choose(junc_name, suc_junc_name):
    templates = [
        f"Can you calculate the distance between {junc_name} and {suc_junc_name}?",
        f"What's the distance from {junc_name} to {suc_junc_name}?",
        f"How far apart are {junc_name} and {suc_junc_name}?"
    ]
    return random.choice(templates)

# Template for junc_direction
def junc_direction_choose(junc_name, suc_junc_name):
    templates = [
        f"Can you describe the relative position of {suc_junc_name} in relation to {junc_name}?",
        f"Can you explain {suc_junc_name}'s location in relation to {junc_name}?",
        f"Please tell me {suc_junc_name}'s orientation in relation to {junc_name}."
        ]    
    return random.choice(templates)

# Template for aoi_category2poi
def aoi_category2poi_choose(aoi_name, max_category):
    templates = [
            f"Can you list all the POIs in the {max_category} category within the area at {aoi_name}?",
            f"Could you provide a list of {max_category} POIs within the {aoi_name} area?",
            f"What POIs are there in {max_category} category within the boundaries of {aoi_name}?"
        ]
    return random.choice(templates)

# Template for aoi_landuse2poi_category
def aoi_landuse2poi_category_choose(aoi_name, landuse_name):
    templates = [
            f"Given the land_use type {landuse_name} of this {aoi_name} AOI, what is the primary category of POIs found here?",
            f"What's the most common category of POIs in the {aoi_name} area designated as {landuse_name}?",
            f"In a {aoi_name} zone of {landuse_name}, what kind of POIs dominate the landscape?"
        ]
    return random.choice(templates)

# Template for poi_category2aoi_landuse
def poi_category2aoi_landuse_choose(max_category, aoi_name):
    templates = [
        f"Considering the most common category {max_category} of POIs in this area, what land_use type might be inferred for this {aoi_name} AOI?",
        f"Given the predominance of {max_category} POIs, what type of land_use does this suggest for AOI {aoi_name}?",
        f"Based on the prevalence of {max_category} POIs, what kind of land_use could be inferred for AOI {aoi_name}?"
        ]
    return random.choice(templates)

# Template for poi2adjacent_pois
def poi2adjacent_pois_choose(chosen_poi_name):
    templates = [
            f"Which POIs are adjacent to {chosen_poi_name} in this area?",
            f"Can you tell me which POIs are next to {chosen_poi_name}?",
            f"What are the surrounding POIs to {chosen_poi_name}?"
        ]
    return random.choice(templates)

# Text for aoi_addr_element
def aoi_addr_element_text(aoi_address):
    text = f"Can you identify the elements in this AOI address: {aoi_address}?"
    return text

# Text for max_aoi_area
def max_aoi_area_text():
    text = "Which AOI has the largest area?"
    return text


# Template for scrambled_task_route
def scrambled_task_route_choose(init_poi_name, dest_poi_name, shuffled_text):
    templates = [
            f"This is a scrambled route from the start point {init_poi_name} to the destination point {dest_poi_name}: {shuffled_text}. Can you provide me with the correct route?",
            f"I've got a route from {init_poi_name} to {dest_poi_name}, but it's all mixed up: {shuffled_text}. Can you put it in order?",
            f"I've got a jumbled route from {init_poi_name} to {dest_poi_name}: {shuffled_text}. Can you help me figure out the correct order?"
        ]
    return random.choice(templates)

# Template for lacking_task_route
def lacking_task_route_choose(init_poi_name, dest_poi_name, info_text):
    templates = [
            f"Here is an incomplete route from the starting point {init_poi_name} to the destination {dest_poi_name}: {info_text}. Can you identify the missing step in the sequence?",
            f"Given the partial route from {init_poi_name} to {dest_poi_name} detailed here: {info_text}, what step appears to be absent?",
            f"In the route provided from {init_poi_name} to {dest_poi_name}: {info_text}, which part of the journey is missing?"
        ]
    return random.choice(templates)

# Template for road2connected_2aois
def road2connected_2aois_choose(chosen_aoi_name, road_name):
    templates = [
            f"Which AOI is connected to {chosen_aoi_name} by {road_name}?",
            f"What's the AOI linked to {chosen_aoi_name} via {road_name}?",
            f"What's the neighboring AOI of {chosen_aoi_name} connected by {road_name}?"
        ]
    return random.choice(templates)


# 外部数据集
# Text for geoeta_addr_element
def geoeta_addr_element_text(elements_requested, full_address):
    text = f"Could you break down the components of this address into {elements_requested}: {full_address}?"
    return text

# Text for geoeag_addr_match
def geoeag_addr_match_text(sentence1):
    text = f"Can you rephrase '{sentence1}' in a different way?"
    return text

# citywalk数据集
# Template for navigation
def navigation_choose(start_location, destination):
    templates = [
            f"Please guide me with navigation from {start_location} to {destination}.",
            f"Can you show me the way from {start_location} to {destination}?",
            f"Could you help me find the route to {destination} from {start_location}?",
            f"Could you indicate the way to go from {start_location} to {destination}?",
            f"I'm trying to reach {destination} from {start_location}. Can you direct me?",
            f"Can you outline the path from {start_location} to {destination}?",
            f"Please assist me in navigating from {start_location} to {destination}.",
            f"I need guidance to {destination} starting from {start_location}, can you help?",
            f"How do I get to {destination} from {start_location}?",
            f"Could you provide directions from {start_location} to {destination}?"
        ]
    return random.choice(templates)


# 在player.py中
# 当前位置描述
def current_aoi_position(junction_name, longitude, latitude, current_aoi_name):
    if EVAL_DATA == True:
        if current_aoi_name:
            return {
                "type": "position",
                "junction_name": junction_name,
                "longitude": longitude,
                "latitude": latitude,
                "current_aoi_name": current_aoi_name
            }
        else:
            return {
                "type": "position",
                "junction_name": junction_name,
                "longitude": longitude,
                "latitude": latitude
            }
    else:
        position_text= "Your current position is{}, longitude:{} latitude:{}".format(junction_name, longitude, latitude)

    if current_aoi_name:
        position_text = position_text + " ({})".format(current_aoi_name)
    # position_text= "Your current position is longitude:{} latitude:{}\n".format(
    #     round(obs["position"]["longlat_position"]["longitude"], 4), 
    #     round(obs["position"]["longlat_position"]["latitude"], 4)
    #     )
    return position_text

def junc_name_text(road_name, next_road_name):
    text = " the junction of {} and {}".format(road_name, next_road_name)
    return text


# 起终点描述模板
def start_dest_text(start_name, start_addr_str, dest_name, dest_addr_str):
    if EVAL_DATA == True:
        return {
            "type": "start_position",
            "start_name": start_name,
            "start_addr": start_addr_str,
            "dest_name": dest_name,
            "dest_addr": dest_addr_str,
            "routes": []
        }
    else:
        templates = [
            "starting from {}{}, you can arrive at the destination {}{} by following the navigation instruction below:\n".format(start_name, start_addr_str, dest_name, dest_addr_str),
            "beginning at {}{}, you can reach your target {}{} by adhering to the following navigation guidelines:\n".format(start_name, start_addr_str, dest_name, dest_addr_str),
            "starting at {}{}, the route to {}{} will be guided by the navigation steps listed below:\n".format(start_name, start_addr_str, dest_name, dest_addr_str),
            "departing from {}{}, you can reach {}{} by following the navigation guide below:\n".format(start_name, start_addr_str, dest_name, dest_addr_str),
            "leaving {}{}, you can get to {}{} by using the route described below:\n".format(start_name, start_addr_str, dest_name, dest_addr_str),
            "embarking from {}{}, you can arrive at {}{} following the instructions below:\n".format(start_name, start_addr_str, dest_name, dest_addr_str),
            "initiating your journey from {}{}, navigate to {}{} with the instructions below:\n".format(start_name, start_addr_str, dest_name, dest_addr_str),
            "setting out from {}{}, you will arrive at {}{} by adhering to the steps outlined below:\n".format(start_name, start_addr_str, dest_name, dest_addr_str),
            "beginning your travel from {}{}, proceed to {}{} by following these directions:\n".format(start_name, start_addr_str, dest_name, dest_addr_str),
            "exiting {}{}, make your way to {}{} by following the detailed path below:\n".format(start_name, start_addr_str, dest_name, dest_addr_str),
            "starting your trip from {}{}, find your way to {}{} using the following guide:\n".format(start_name, start_addr_str, dest_name, dest_addr_str),
            "your departure from {}{}, head towards {}{} as directed below:\n".format(start_name, start_addr_str, dest_name, dest_addr_str)
            ]
        # text = "starting from {}{}, you can arrive at the destination {}{} by following the navigation instruction below:\n".format(start_name, start_addr_str, dest_name, dest_addr_str)
        # return text
        
    return random.choice(templates)

# 路口处行走描述模板
def junc_walk_text(basic_direction, junction_name):
    if EVAL_DATA:
        return {
            "type": "junc",
            "junc_name": junction_name,
            "direction": basic_direction
        }

    else:
        templates = [
            "at{}, proceed {}".format(junction_name, basic_direction),
            "upon reaching{}, head {}".format(junction_name, basic_direction),
            "when you arrive at{}, continue {}".format(junction_name, basic_direction),
            "at{}, go {}".format(junction_name, basic_direction),
            "once at{}, move {}".format(junction_name, basic_direction),
            "navigate through{} by going {}".format(junction_name, basic_direction),
            "while at{}, follow the path leading {}".format(junction_name, basic_direction),
            "on reaching{}, your direction should be {}".format(junction_name, basic_direction),
            "enter{} and proceed {}".format(junction_name, basic_direction),
            "reach{} and then take the {} direction".format(junction_name, basic_direction)
        ]
        # text = "then go {} towards{}".format(basic_direction, junction_name)
        # return text
    return random.choice(templates)

def step_interests_text(interests_side):
    """描述每一步看到的POI信息，用于citywalk-基于landmark的改进"""
    descriptions = []
            
    for side, interests in interests_side.items():
        names_with_categories = []
        if interests:  # 如果列表不为空
            for interest in interests:
                interest_name = interest['name']
                interest_category = interest['category']
                description = f"{interest_name}({interest_category})"
                names_with_categories.append(description)
            joined_interests = ", ".join(names_with_categories)
            descriptions.append(f"{joined_interests} on the {side}")

    if descriptions == []:
        return None
    # 连接所有描述
    all_description = ", ".join(descriptions)
    templates = [
        "you can see {}".format(all_description),
        "with {} in sight".format(all_description),
        "noticing {}".format(all_description)
    ]
    return random.choice(templates)

# 附近POI信息描述
# TODO 考虑增加关于方位的表述，将当前位置的POI按照分为4个象限，以丰富信息量，根据评测设计进行调整
def describe_pois_via_text(map, pois_info, radius, has_category, detail_interest=False):
    """将当前位置所在的POI信息进行文字描述"""
    count = 0
    info = []
    poi_data = {"pois": []}
    for key in pois_info:
        count += 1
        poi_descriptions = [] 
        if EVAL_DATA == True:
            if len(pois_info[key])==0:
                continue
            if has_category:
                poi_descriptions = [
                    {
                        "name": LANGUAGE.get_poi_name(poi["id"], map),
                        "category": poi["category"]
                    }
                    for poi in pois_info[key]
                ]

            else:
                poi_descriptions = [
                        {
                            "name": LANGUAGE.get_poi_name(poi["id"], map)
                        }
                        for poi in pois_info[key]
                    ]

            poi_data["pois"].extend(poi_descriptions)
            
        # poi_descriptions = [LANGUAGE.poi_name_choose(poi,poi["id"],USE_ENGLISH) + "(" + category_id_name["L3"][poi["category"]] +")" for poi in pois_info[key]]
        else:
            if len(pois_info[key])==0:
                continue
            if detail_interest:
                info_item = "There are {} {} POIs, they are ".format(len(pois_info[key]), key)
                if has_category:
                    info_item += ";".join([LANGUAGE.get_poi_name([poi["id"]], map) + "(" + poi["category"] +")" for poi in pois_info[key]])
                else:
                    info_item += ";".join([LANGUAGE.get_poi_name(poi["id"], map) for poi in pois_info[key]])
                info.append(info_item)

            else:
                info_item = "{} {} POIs".format(len(pois_info[key]), key)
                info.append(info_item)
        
    if EVAL_DATA == True:
        return poi_data
    
    if detail_interest:
        init_context = "There are {} kinds of POIs within a {}-meter radius.".format(count, radius)
        if count > 0:
            init_context += " Detailed information of them are as follows: "

        info = [init_context] + info

        return "\n".join(info)
    else:
        if count > 0:
            init_context = "There are "
        else:
            init_context = "There are no POIs within a {}-meter radius.".format(radius)
        return init_context + ", ".join(info)


# 单步移动描述
def describe_one_step_text(road_length, road_name, direction):
    """根据当前步骤描述移动"""
    if EVAL_DATA == True:
        return {
            "type": "road",
            "road_name": road_name,
            "road_length": road_length,
            "direction": direction
        }

    else:
        templates = [
            "move {} meters along {} {}".format(road_length, road_name, direction),
            "walk along {} for {} meters {}".format(road_name, road_length, direction),
            "head down {}, traveling {} meters {}".format(road_name, road_length, direction),
            "follow {} for about {} meters, moving {}".format(road_name, road_length, direction),
            "march along {} for a total of {} meters, proceeding {}".format(road_name, road_length, direction),
            "amble along {}, covering {} meters {}".format(road_name, road_length, direction),
            "navigate your way through {} for {} meters, heading {}".format(road_name, road_length, direction),
            "saunter down {} for roughly {} meters {}".format(road_name, road_length, direction),
            "make your way along {} for {} meters {}".format(road_name, road_length, direction),
            "proceed directly along {} and cover {} meters {}".format(road_name, road_length, direction)
        ]
        # text = "walk along {} for {} meters {}".format(road_name, road_length, direction)
        # return text
    return random.choice(templates)

# 到达终点文本描述
def end_point_text(dest_poi_name):
    end_point = "move to the destination {}".format(dest_poi_name)
    
    return end_point

# 最终位置描述
def final_position(name, longitude, latitude, current_aoi_name):
    if current_aoi_name:
        return {
            "type": "dest_position",
            "dest_name": name, 
            "current_aoi_name": current_aoi_name,
            "longitude": longitude, 
            "latitude": latitude
        }
    else:
        return {
            "type": "dest_position",
            "dest_name": name, 
            "longitude": longitude, 
            "latitude": latitude
        }


# 在train_task.py中
def task_description_text(start_poi_name, start_poi_addr_str, dest_poi_name, dest_poi_addr_str, lng, lat):
    templates = [
        "You are in {}{} and you need to go to {}{}. Your current position is longitude:{:.4f} latitude:{:.4f}.".format(start_poi_name, start_poi_addr_str, dest_poi_name, dest_poi_addr_str, lng, lat),
        "You're at {}{}, and you need to get to {}{}. Present coordinates are Longitude: {:.4f}, Latitude: {:.4f}.".format(start_poi_name, start_poi_addr_str, dest_poi_name, dest_poi_addr_str, lng, lat),
        "Right now, you are located at {}{} and your goal is to reach {}{}. Your longitude and latitude are {:.4f} and {:.4f} respectively.".format(start_poi_name, start_poi_addr_str, dest_poi_name, dest_poi_addr_str, lng, lat),
        "Here's where you are: {}{}. Here's where you're going: {}{}. You are currently at Longitude: {:.4f}, Latitude: {:.4f}.".format(start_poi_name, start_poi_addr_str, dest_poi_name, dest_poi_addr_str, lng, lat),
        "You have started from {}{} and need to proceed to {}{}. Your exact position is Longitude: {:.4f}, Latitude: {:.4f}.".format(start_poi_name, start_poi_addr_str, dest_poi_name, dest_poi_addr_str, lng, lat),
        "Your current position is at {}{}, and you need to head towards {}{}. Coordinates are Longitude: {:.4f} and Latitude: {:.4f}.".format(start_poi_name, start_poi_addr_str, dest_poi_name, dest_poi_addr_str, lng, lat),
        "Now at {}{}; next stop {}{}. Your current longitude and latitude read {:.4f} and {:.4f}.".format(start_poi_name, start_poi_addr_str, dest_poi_name, dest_poi_addr_str, lng, lat),
        "You are positioned at {}{} and must travel to {}{}. Your coordinates are Longitude: {:.4f}, Latitude: {:.4f}.".format(start_poi_name, start_poi_addr_str, dest_poi_name, dest_poi_addr_str, lng, lat),
    ]
        # text = "You are in {}{} and you need to go to {}{}. Your current position is longitude:{:.4f} latitude:{:.4f}.".format(start_poi_name, start_poi_addr_str, dest_poi_name, dest_poi_addr_str, lng, lat)
        # return text
  
    return random.choice(templates)

# 在agent.py中
# init_prompt文本的构建
def init_prompt_text(current_task, available_actions_text):
    init_prompt = "Here is your task. "+ current_task["task"][:-1] + available_actions_text
    return init_prompt
    
