import json
from collections import defaultdict

with open("dataset/skiing/area_name_to_state.json", "r") as f:
    area_name_to_state = json.load(f)

with open("dataset/skiing/area_name_to_rating_and_sentiment.json", "r") as f:
    area_name_to_rating_and_sentiment = json.load(f)

with open("dataset/skiing/area_name_to_top_sentiment.json", "r") as f:
    area_name_to_top_sentiment = json.load(f)

with open("dataset/skiing/reviews.json") as f:
    dataset = json.load(f)

with open("dataset/skiing/area_name_data.json", "w") as f:
    result = defaultdict(dict)
    for area_name, state in area_name_to_state.items():
        result[area_name]['state'] = state
        result[area_name]['average_rating'] = area_name_to_rating_and_sentiment[area_name]["average_rating"]
        result[area_name]['average_sentiment'] = area_name_to_rating_and_sentiment[area_name]["average_sentiment"]
        result[area_name]['top_10_positive'] = [{"sentiment_score": d['sentiment']['compound'],
                                                 'review':dataset[d['row_number']]}
                                                for d in area_name_to_top_sentiment[area_name]['top_10_positive']]
        result[area_name]['top_10_negative'] = [{"sentiment_score": d['sentiment']['compound'],
                                                 'review':dataset[d['row_number']]}
                                                for d in area_name_to_top_sentiment[area_name]['top_10_negative']]
    json.dump(result, f)
