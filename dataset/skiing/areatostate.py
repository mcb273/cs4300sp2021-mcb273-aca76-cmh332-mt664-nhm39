import json


with open("dataset/skiing/reviews.json", "r") as f:
    data = json.load(f)
with open("dataset/skiing/area_name_to_state.json", "w") as f2:
    processed = []
    area_name_to_state = {}
    for review in data["reviews"]:
        area_name = review['area_name']
        if area_name not in processed:
            processed.append(area_name)
            area_name_to_state[area_name] = review['state']
    json.dump(area_name_to_state, f2)
