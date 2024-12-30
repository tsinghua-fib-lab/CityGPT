from simulate.utils import Action, RunMode

def get_system_instruction(run_mode=RunMode.NORMAL):
    if run_mode == RunMode.CITY_WALK:
        return "You are a tourist who want to explore <Beijing> by walking or driving. Imagine you are an intelligent agent in the city and your target is to go to the designated locations and records the surroundings in the way. You can obtain the route to the destination with action <navigate> and then use aciton <walk> or <drive> to got to it. In first round, you can only choose action <navigate>. And in following rounds, you can only choose action <walk> or <drive>. You should choose from two actions: \"THOUGHT\" or \"ACTION\". If you choose \"THOUGHT\", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:\"THOUGHT: your thoughts.\n ACTION: your next action\n\"; If you choose \"ACTION\", you should directly output the action in this turn. Your output must strictly follow this format:\"ACTION: your next action\n\". After your each turn, the environment will give you immediate feedback based on which you plan your next few steps."
    elif run_mode == RunMode.NORMAL:
        # adapated from AgentBench
        return "Interact with a city to solve a task. Imagine you are an intelligent agent in a city environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the simple description of the current position and your goal to accomplish. You can explore the city with search and navigate service when necessary. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. You should choose from two actions: \"THOUGHT\" or \"ACTION\". If you choose \"THOUGHT\", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:\"THOUGHT: your thoughts.\n ACTION: your next action\n\"; If you choose \"ACTION\", you should directly output the action in this turn. Your output must strictly follow this format:\"ACTION: your next action\n\". After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the environment output \"Nothing happened\", that means the previous action is invalid and you should try more options.\n Reminder: \n1. the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. \n2. Think when necessary, try to act directly more in the process.\n\n"
    else:
        raise NotImplementedError


def get_prompts(action: str):
    """"构造1-shot示例，注意该示例不要用于SFT训练"""
    example_prompts = {
        "buy": [
            "You are in a residential area. Your task is to: buy some potatoes and bring them back. \n",
            "THOUGHT: The task is to buy potatoes and bring them back, so first i need to searh the city for shops. After finding the shops, i need to go to it. I will start my search on shops.\n ACTION: {} convenience store\n".format(Action.SEARCH.value),
            "There are 5 shops around you. They are: shop-1, shop-2, shop-3, jewelry store-1, market-1",
            "THOUGHT: Potatoes are very common and available in almost every store. I only need to go to the nearest store from above ones. Thus, i need to obtain the routes to them.\n ACTION: {} shop-1,shop-2,shop-3".format(Action.NAVIGATE.value),
            "Go to shop-1: {'walk':{'route':[...], 'price': 0, 'eta':10}, 'drive':{'route':[...], 'price': 10, 'eta':5},}.\n Go to shop-2: {'walk':{'route':[...], 'price': 0, 'eta':20}, 'drive':{'route':[...], 'price': 15, 'eta':7},}.\n Go to shop-3: {'walk':{'route':[...], 'price': 0, 'eta':30}, 'drive':{'route':[...], 'price': 20, 'eta':10},}.",
            "THOUGHT: We should select the route with minimum time cost and price cost. \n ACTION: {} shop-1".format(Action.WALK.value),
            "You have walk to the nearyby region of shop-1.",
            "THOUGHT: I need to look around the surrondings and find the specific location of shop-1.\n ACTION: {}".format(Action.LOOK_IN_DOOR.value),
            "There are 3 shops in the vicinity of your current location. They are: restaurant-1, jewelry store-2, shop-4.",
            "THOUGHT: shop-1 is not here. I should walk around like move to shop-4 to find wheather shop-1 is around it.\n ACTION: {} shop-4".format(Action.MOVE_IN_DOOR.value),
            "Now you are in the shop-4.",
            "ACTION: {}".format(Action.LOOK_IN_DOOR.value),
            "There are 3 shops in the vicinity of your current location. They are: shop-1, shop-4, shop-10.",
            "ACTION: {} shop-1.".format(Action.MOVE_IN_DOOR.value),
            "You are in the shop-1.",
            "ACTION: {} potatoes".format(Action.BUY.value),
            "You buy potatoes by paying $1",
            "THOUGHT: I've already made the purchase, it's time to go home.\n ACTION: {} home".format(Action.NAVIGATE.value),
            "There are two ways to go home. They are: {'walk':{'route':[...], 'price': 0, 'eta':10}, 'drive':{'route':[...], 'price': 10, 'eta': 5},}",
            "ACTION: {} home".format(Action.WALK.value),
            "Your are back to home.",
            "ACTION: Task is completed."
        ],
        "citywalk": [
            "You are in the U-Center. Your task is to: go to Tsinghua University.",
            "THOUGHT: I am in U-Center, I want to go to Tsinghua University, I need to navigate to Tsinghua University first. ACTION: {} Tsinghua University".format(Action.NAVIGATE.value),
            "You can arrive at the destination Tsinghua University by following the route below.\n UCenter-成府路(500m)-成府路(500m)-成府路和中关村东路交叉口-中关村东路(200m)-Tsinghua University",
            "THOUGHT: OK, I will follow the instructions and walk step by step. ACTION: {} Tsinghua University".format(Action.WALK.value),
            "You walk along 成府路 with 100 merters. There are 5 kinds of POIs within 100-merter raidus. Detailed information of them are as follows.\n 3 Food POIs: 局气(北京菜),云海肴(云南菜),米村拌饭(快餐).\n 3 Entertainment POIs: xxx",
            "ACTION: {} Tsinghua University".format(Action.WALK.value),
            "You continue to walk along 成府路 with 100 meters. There are xxx....",
            "ACTION: {} Tsinghua University".format(Action.WALK.value),
            "You pass through the intersection of 成府路 and 中关村东路. You continue to walk along 中关村东路 with 100 meters. There are xxx",
            "ACTION: {} Tsinghua University".format(Action.WALK.value),
            "You have arrived at the destination Tsinghua University.",
            "ACTION: Task is completed."
        ],
        "eat": []
    }
    if action in example_prompts:
        return example_prompts[action]
    else:
        return ""

def get_available_actions(actions):
    actions = "\n".join(actions)
    return " AVAILABLE ACTIONS: " + actions + "\n"
