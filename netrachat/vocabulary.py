"""
Vocabulary shared by every part of NetraChat.

The object detector (YOLOv8, trained on the COCO dataset) knows 80 object
classes. People do not always use the exact class name, so this module maps
the words a user might type ("people", "kids", "phone", "sofa") to the
detector's class names ("person", "cell phone", "couch").

It also stores the correct singular and plural form of every class, so the
response generator never writes things like "2 person" or "1 skis".
"""

# --- The 80 COCO classes: class name -> (singular, plural) -------------------
COCO_FORMS = {
    "person": ("person", "people"),
    "bicycle": ("bicycle", "bicycles"),
    "car": ("car", "cars"),
    "motorcycle": ("motorcycle", "motorcycles"),
    "airplane": ("airplane", "airplanes"),
    "bus": ("bus", "buses"),
    "train": ("train", "trains"),
    "truck": ("truck", "trucks"),
    "boat": ("boat", "boats"),
    "traffic light": ("traffic light", "traffic lights"),
    "fire hydrant": ("fire hydrant", "fire hydrants"),
    "stop sign": ("stop sign", "stop signs"),
    "parking meter": ("parking meter", "parking meters"),
    "bench": ("bench", "benches"),
    "bird": ("bird", "birds"),
    "cat": ("cat", "cats"),
    "dog": ("dog", "dogs"),
    "horse": ("horse", "horses"),
    "sheep": ("sheep", "sheep"),
    "cow": ("cow", "cows"),
    "elephant": ("elephant", "elephants"),
    "bear": ("bear", "bears"),
    "zebra": ("zebra", "zebras"),
    "giraffe": ("giraffe", "giraffes"),
    "backpack": ("backpack", "backpacks"),
    "umbrella": ("umbrella", "umbrellas"),
    "handbag": ("handbag", "handbags"),
    "tie": ("tie", "ties"),
    "suitcase": ("suitcase", "suitcases"),
    "frisbee": ("frisbee", "frisbees"),
    "skis": ("pair of skis", "pairs of skis"),
    "snowboard": ("snowboard", "snowboards"),
    "sports ball": ("ball", "balls"),
    "kite": ("kite", "kites"),
    "baseball bat": ("baseball bat", "baseball bats"),
    "baseball glove": ("baseball glove", "baseball gloves"),
    "skateboard": ("skateboard", "skateboards"),
    "surfboard": ("surfboard", "surfboards"),
    "tennis racket": ("tennis racket", "tennis rackets"),
    "bottle": ("bottle", "bottles"),
    "wine glass": ("wine glass", "wine glasses"),
    "cup": ("cup", "cups"),
    "fork": ("fork", "forks"),
    "knife": ("knife", "knives"),
    "spoon": ("spoon", "spoons"),
    "bowl": ("bowl", "bowls"),
    "banana": ("banana", "bananas"),
    "apple": ("apple", "apples"),
    "sandwich": ("sandwich", "sandwiches"),
    "orange": ("orange", "oranges"),
    "broccoli": ("piece of broccoli", "pieces of broccoli"),
    "carrot": ("carrot", "carrots"),
    "hot dog": ("hot dog", "hot dogs"),
    "pizza": ("pizza", "pizzas"),
    "donut": ("donut", "donuts"),
    "cake": ("cake", "cakes"),
    "chair": ("chair", "chairs"),
    "couch": ("couch", "couches"),
    "potted plant": ("potted plant", "potted plants"),
    "bed": ("bed", "beds"),
    "dining table": ("dining table", "dining tables"),
    "toilet": ("toilet", "toilets"),
    "tv": ("TV", "TVs"),
    "laptop": ("laptop", "laptops"),
    "mouse": ("computer mouse", "computer mice"),
    "remote": ("remote control", "remote controls"),
    "keyboard": ("keyboard", "keyboards"),
    "cell phone": ("mobile phone", "mobile phones"),
    "microwave": ("microwave", "microwaves"),
    "oven": ("oven", "ovens"),
    "toaster": ("toaster", "toasters"),
    "sink": ("sink", "sinks"),
    "refrigerator": ("refrigerator", "refrigerators"),
    "book": ("book", "books"),
    "clock": ("clock", "clocks"),
    "vase": ("vase", "vases"),
    "scissors": ("pair of scissors", "pairs of scissors"),
    "teddy bear": ("teddy bear", "teddy bears"),
    "hair drier": ("hair dryer", "hair dryers"),
    "toothbrush": ("toothbrush", "toothbrushes"),
}

# --- Extra words users type -> detector class name ---------------------------
SYNONYMS = {
    "people": "person", "persons": "person", "human": "person", "humans": "person",
    "man": "person", "men": "person", "woman": "person", "women": "person",
    "boy": "person", "boys": "person", "girl": "person", "girls": "person",
    "child": "person", "children": "person", "kid": "person", "kids": "person",
    "guy": "person", "guys": "person", "lady": "person", "ladies": "person",
    "player": "person", "players": "person", "pedestrian": "person",
    "pedestrians": "person", "someone": "person", "somebody": "person",
    "anyone": "person", "anybody": "person",
    "bike": "bicycle", "bikes": "bicycle", "cycle": "bicycle", "cycles": "bicycle",
    "motorbike": "motorcycle", "motorbikes": "motorcycle", "scooter": "motorcycle",
    "scooters": "motorcycle",
    "plane": "airplane", "planes": "airplane", "aeroplane": "airplane",
    "aeroplanes": "airplane", "aircraft": "airplane", "jet": "airplane",
    "automobile": "car", "taxi": "car", "taxis": "car",
    "lorry": "truck", "lorries": "truck",
    "ship": "boat", "ships": "boat",
    "puppy": "dog", "puppies": "dog", "kitten": "cat", "kittens": "cat",
    "kitty": "cat", "pony": "horse", "lamb": "sheep",
    "bag": "handbag", "bags": "handbag", "purse": "handbag", "purses": "handbag",
    "luggage": "suitcase", "ball": "sports ball", "balls": "sports ball",
    "football": "sports ball", "racket": "tennis racket", "racquet": "tennis racket",
    "glass": "wine glass", "glasses": "wine glass", "mug": "cup", "mugs": "cup",
    "doughnut": "donut", "doughnuts": "donut",
    "sofa": "couch", "sofas": "couch", "plant": "potted plant", "plants": "potted plant",
    "table": "dining table", "tables": "dining table",
    "television": "tv", "televisions": "tv", "tvs": "tv", "monitor": "tv",
    "screen": "tv", "computer": "laptop", "notebook": "laptop",
    "phone": "cell phone", "phones": "cell phone", "mobile": "cell phone",
    "mobiles": "cell phone", "smartphone": "cell phone", "smartphones": "cell phone",
    "cellphone": "cell phone", "fridge": "refrigerator", "fridges": "refrigerator",
    "remote control": "remote", "remotes": "remote", "teddy": "teddy bear",
    "hair dryer": "hair drier", "dryer": "hair drier", "mice": "mouse",
}

# Category words: a question like "are there any animals?" covers many classes.
CATEGORIES = {
    "animal": {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
               "zebra", "giraffe"},
    "vehicle": {"bicycle", "car", "motorcycle", "airplane", "bus", "train",
                "truck", "boat"},
    "food": {"banana", "apple", "sandwich", "orange", "broccoli", "carrot",
             "hot dog", "pizza", "donut", "cake"},
    "fruit": {"banana", "apple", "orange"},
    "furniture": {"chair", "couch", "bed", "dining table", "bench"},
    "electronic": {"tv", "laptop", "mouse", "remote", "keyboard", "cell phone"},
    "kitchenware": {"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"},
    "appliance": {"microwave", "oven", "toaster", "sink", "refrigerator"},
    "sports equipment": {"frisbee", "skis", "snowboard", "sports ball", "kite",
                         "baseball bat", "baseball glove", "skateboard",
                         "surfboard", "tennis racket"},
}
CATEGORY_WORDS = {
    "animal": "animal", "animals": "animal", "pet": "animal", "pets": "animal",
    "vehicle": "vehicle", "vehicles": "vehicle", "transport": "vehicle",
    "food": "food", "foods": "food", "meal": "food", "fruit": "fruit",
    "fruits": "fruit", "furniture": "furniture", "electronics": "electronic",
    "electronic": "electronic", "gadget": "electronic", "gadgets": "electronic",
    "device": "electronic", "devices": "electronic", "utensil": "kitchenware",
    "utensils": "kitchenware", "kitchenware": "kitchenware",
    "appliance": "appliance", "appliances": "appliance",
    "sports equipment": "sports equipment", "sport": "sports equipment",
    "sports": "sports equipment",
}
CATEGORY_FORMS = {
    "animal": ("animal", "animals"), "vehicle": ("vehicle", "vehicles"),
    "food": ("food item", "food items"), "fruit": ("fruit", "fruits"),
    "furniture": ("piece of furniture", "pieces of furniture"),
    "electronic": ("electronic device", "electronic devices"),
    "kitchenware": ("kitchen item", "kitchen items"),
    "appliance": ("appliance", "appliances"),
    "sports equipment": ("piece of sports equipment", "pieces of sports equipment"),
}

# Plain colour names the colour estimator can return.
COLOR_WORDS = ["red", "orange", "yellow", "green", "blue", "purple", "pink",
               "brown", "black", "white", "grey"]


def singular(label: str) -> str:
    """Return the readable singular name of a detector class."""
    return COCO_FORMS.get(label, (label, label + "s"))[0]


def plural(label: str) -> str:
    """Return the readable plural name of a detector class."""
    return COCO_FORMS.get(label, (label, label + "s"))[1]


def _build_lookup():
    """Map every phrase a user might type to a detector class name."""
    lookup = {}
    for label, (sing, plur) in COCO_FORMS.items():
        lookup[label] = label
        lookup[sing.lower()] = label
        lookup[plur.lower()] = label
    lookup.update(SYNONYMS)
    return lookup


PHRASE_TO_LABEL = _build_lookup()
