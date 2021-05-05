from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
from collections import defaultdict

sia = SentimentIntensityAnalyzer()


# for each review do sia.polarity_scores(review)
def calculateSentimentForReviews():
    with open("dataset/skiing/sentiment.json", "w") as f:
        with open("dataset/skiing/reviews.json", "r") as f2:
            dataset = json.load(f2)
            data = {}
            # for review in dataset['reviews']:
            for row_num, review in dataset.items():
                text = review['text']
                scores = sia.polarity_scores(text)
                data[row_num] = scores
            json.dump(data, f)


def get_area_name_to_sentiments():
    result = defaultdict(list)
    with open("dataset/skiing/sentiment.json", "r") as f:
        sentiment = json.load(f)
    with open("dataset/skiing/reviews.json", "r") as f2:
        dataset = json.load(f2)
    for row_num, sent in sentiment.items():
        area_name = dataset[row_num]["area_name"]
        d = {"row_number": row_num, "sentiment": sent}
        result[area_name].append(d)
    return result


def get_top_10_for_area(sentiments, isPositive):
    # key = "pos" if isPositive else "neg"
    # top = sorted(sentiments, key=lambda x: x["sentiment"][key], reverse=True)
    # return top[:min(10, len(top))]

    # if positive, rank by compound descending (reverse is True), else rank by compound ascending
    # so reverse = isPositive
    top = sorted(
        sentiments, key=lambda x: x['sentiment']['compound'], reverse=isPositive)
    if isPositive:
        top = [x for x in top if x['sentiment']['compound'] > 0]
    else:
        top = [x for x in top if x['sentiment']['compound'] < 0]
    return top[:min(10, len(top))]


def getSentimentForReviewsByAreaName():
    area_name_to_sentiments = get_area_name_to_sentiments()
    result = {}
    for area_name, lst in area_name_to_sentiments.items():
        top10positive, top10negative = get_top_10_for_area(
            lst, True), get_top_10_for_area(lst, False)
        result[area_name] = {
            "top_10_positive": top10positive,
            "top_10_negative": top10negative
        }
    with open("dataset/skiing/area_name_to_top_sentiment.json", "w") as f:
        json.dump(result, f)


# print(sia.polarity_scores("some really awesome skiing here, loved it"))
calculateSentimentForReviews()
getSentimentForReviewsByAreaName()
# with open("dataset/skiing/area_name_to_top_sentiment.json", "r") as f:
#     x = json.load(f)
#     print(x['killington-resort'])
