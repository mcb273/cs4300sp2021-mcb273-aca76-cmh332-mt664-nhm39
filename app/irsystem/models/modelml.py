import numpy as np
import pickle
import pandas as pd
import csv
import re
import nltk
import json
# import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load data from train.txt file
# Each row in train.txt stores one feature and it's label seperated by semi-colon
# ex. I hate apple; angry
# Returns pd dataframe with coloumns features and labels


def load_training_data(path='dataset/mldata/train.txt'):
    df = pd.read_csv(path, sep=";", header=None)
    df.columns = ["features", "labels"]
    return df

# Process training dataset
# Takes pd dataframe returned by load_training_data()
# Minimally processes features


def extract_training_data(df, hasLabels=True):
    features = df["features"].values
    labels = df["labels"].values if hasLabels else None

    processed_features = []

    for sentence in range(0, len(features)):
        # Remove all the special characters
        processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

        # Remove all single characters
        processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

        # Substituting multiple spaces with single space
        processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

        # Convert to lowercase
        processed_feature = processed_feature.lower()

        processed_features.append(processed_feature)

    return processed_features, labels

# Turns processed_features into a Tfidf matrix


def vectorize_feats(processed_features):
    vectorizer = TfidfVectorizer(
        max_features=2000, min_df=2, max_df=0.8, stop_words=stopwords.words('english'))
    processed_features = vectorizer.fit_transform(processed_features).toarray()

    feature_names_list = vectorizer.get_feature_names()
    return processed_features, feature_names_list

# Splits dataset into a training dataset and a testing dataset where the training set is 80% of the total dataset and testing 20%
# Returns X_train, X_test, y_train, y_test where X_train and y_train are the features and labels for training respectively


def split(processed_features, labels, test_size=0.2, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(
        processed_features, labels, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Use RandomForestClassifier on training dataset to train model
# Returns a RandomForestClassifier object fit to training dataset


def train(X_train, y_train):
    text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    text_classifier.fit(X_train, y_train)
    return text_classifier


def save_model(classifier, pkl_filename="dataset/mldata/pickle_model.pkl"):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(classifier, file)


def load_model(pkl_filename="dataset/mldata/pickle_model.pkl"):
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
        return pickle_model


def predict(text_classifier, X_test, y_test):
    predictions = text_classifier.predict(X_test)
    if y_test is not None:
        return predictions, confusion_matrix(y_test, predictions), classification_report(y_test, predictions), accuracy_score(y_test, predictions)
    else:
        return predictions, None, None, None


def buildModel():
    dataframe = load_training_data()
    processed_features, labels = extract_training_data(dataframe)
    tfidf_processed_features, feature_names_list = vectorize_feats(
        processed_features)
    X_train, X_test, y_train, y_test = split(tfidf_processed_features, labels)
    classifier = train(X_train, y_train)
    save_model(classifier)
    return classifier, X_test, y_test


def testModel(classifier, X_test, y_test):
    predictions, confusion, classification, accuracy = predict(
        classifier, X_test, y_test)
    print("Confusion Matrix:\n", confusion)
    print("Classification Report:\n", classification)
    print("Accuracy:\n", accuracy)


def get_feature_names_list():
    dataframe = load_training_data()
    processed_features, labels = extract_training_data(dataframe)
    tfidf_processed_features, feature_names_list = vectorize_feats(
        processed_features)
    with open("dataset/mldata/feature_names.csv", 'w') as myfile:
        wr = csv.writer(myfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for item in feature_names_list:
            wr.writerow([item])
    return feature_names_list


def processReviews(reviews_path="dataset/skiing/reviews.json"):
    with open(reviews_path, "r") as f:
        reviews = json.load(f)
    data_flat = {"features": [review['text'] for _, review in reviews.items()]}
    df = pd.DataFrame.from_dict(data_flat)
    features, _ = extract_training_data(df, hasLabels=False)
    with open("dataset/mldata/feature_names.csv", newline='') as f:
        reader = csv.reader(f)
        feature_names_list = list(reader)
    # print(feature_names_list[0])
    flat_feature_names_list = [
        item for sublist in feature_names_list for item in sublist]
    # print(flat_feature_names_list[0])
    num_feats = len(feature_names_list)
    # print (num_feats)
    num_revs = len(features)
    vectorize = np.zeros((num_revs, num_feats))
    for rev_num in (range(num_revs)):
        processed_feat = features[rev_num]
        # print (processed_feat)
        tokens = processed_feat.split()
        for tok in tokens:
            if tok in flat_feature_names_list:
                # print (tok)
                index = flat_feature_names_list.index(tok)
                vectorize[rev_num, index] += 1.0
    # print (np.shape(vectorize))
    return vectorize, features


def loadModelAndRunOnReviews(model_path="dataset/mldata/pickle_model.pkl", reviews_path="dataset/skiing/reviews.json", save_path="dataset/mldata/review_to_emotion.json"):
    model = load_model(pkl_filename=model_path)
    dataset, text = processReviews(reviews_path=reviews_path)
    with open(reviews_path, "r") as f:
        reviews = json.load(f)
    predictions, _, _, _ = predict(model, dataset, y_test=None)
    with open(save_path, "w") as f:
        result = {}
        for i in range(len(predictions)):
            result[i] = {"emotion": predictions[i],
                         "text": reviews[str(i)]['text']}
        json.dump(result, f)

    # d = defaultdict(int)
    # count = 0
    # for i in range(len(predictions)):
    #     if count < 5:
    #         print(predictions[i])
    #         print(np.sum(dataset[i]))
    #         print(text[i])
    #         print(reviews[str(i)])
    #         print("\n\n\n")
    #         count += 1
    #     d[predictions[i]] += 1

    # print(d)


##model = loadModelAndRunOnReviews()

# feature_names_list = get_feature_names_list()
# classifier, X_test, y_test= buildModel()
# review_vect, features = processReviews()
# loadModelAndRunOnReviews ()

dataframe = load_training_data()
processed_features, labels = extract_training_data(dataframe)
print(set(labels))
