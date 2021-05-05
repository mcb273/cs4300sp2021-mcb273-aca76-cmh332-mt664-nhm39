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

with open("dataset/mldata/review_to_emotion.json") as f:
    review_to_emotion = json.load(f)

area_to_emotion_count = defaultdict(dict)
area_to_review_count = defaultdict(int)
for i in range(len(dataset)):
    str_i = str(i)
    if str_i not in dataset:
        continue
    emotion = review_to_emotion[str_i]['emotion']
    text = review_to_emotion[str_i]['text']
    word_count = len(text.split())
    area_name = dataset[str_i]['area_name']
    area_to_review_count[area_name] += 1

    if emotion != 'fear' and word_count > 100:
        if area_name in area_to_emotion_count:
            area_to_emotion_count[area_name][emotion] += 1
        else:
            area_to_emotion_count[area_name] = {
                'love': 0,
                'surprise': 0,
                'sadness': 0,
                'anger': 0,
                'joy': 0,
            }

        area_to_emotion_count[area_name][emotion] += 1

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
        result[area_name]['emotions'] = area_to_emotion_count[area_name]
        result[area_name]['number_of_reviews'] = area_to_review_count[area_name]
    json.dump(result, f)
