import json

with open("dataset/skiing/reviews.json", "r") as f:
    dataset = json.load(f)

with open("dataset/skiing/reviews-deduped.json", "w") as f:
    result = {}
    processed = []
    for row_num, review in dataset.items():
        if review['text'] not in processed:
            processed.append(review['text'])
            result[row_num] = review

    json.dump(result, f)
