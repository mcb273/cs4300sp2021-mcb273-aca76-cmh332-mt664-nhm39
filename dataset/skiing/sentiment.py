from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json

sia = SentimentIntensityAnalyzer()


# for each review do sia.polarity_scores(review)
with open("dataset/skiing/sentiment.json", "w") as f:
    with open("dataset/skiing/reviews.json", "r") as f2:
        dataset = json.load(f2)
        data = {}
        for review in dataset['reviews']:
            text = review['text']
            scores = sia.polarity_scores(text)
            data[review['row_number']] = scores
        json.dump(data, f)
