import json
from collections import defaultdict

with open("dataset/skiing/sentiment.json", "r") as f:
    row_num_to_sentiment = json.load(f)

with open("dataset/skiing/reviews.json", "r") as f:
    dataset = json.load(f)

with open("dataset/skiing/area_name_to_rating_and_sentiment.json", "w") as f:
    result = {}
    area_name_to_total_rating = defaultdict(int)
    area_name_to_review_count = defaultdict(int)
    area_name_to_total_sentiment = defaultdict(float)
    for row_num, sentiment_dict in row_num_to_sentiment.items():
        area_name = dataset[row_num]['area_name']
        area_name_to_total_rating[area_name] += int(
            dataset[row_num]['rating'])
        area_name_to_review_count[area_name] += 1
        area_name_to_total_sentiment[area_name] += sentiment_dict['compound']
    for area_name in area_name_to_total_rating:
        result[area_name] = {"average_rating":
                             area_name_to_total_rating[area_name] /
                             area_name_to_review_count[area_name],
                             "average_sentiment":
                             area_name_to_total_sentiment[area_name] /
                             area_name_to_review_count[area_name]}
    json.dump(result, f)
