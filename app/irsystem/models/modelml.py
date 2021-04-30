import numpy as np
import pickle
import pandas as pd
import csv
import re
import nltk 
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#TODO: change to relative file path

# Load data from train.txt file
# Each row in train.txt stores one feature and it's label seperated by semi-colon
    # ex. I hate apple; angry
# Returns pd dataframe with coloumns features and labels 
def load_training_data (path ='/Users/nicholasmohan/Desktop/cs4300sp2021-mcb273-aca76-cmh332-mt664-nhm39/dataset/mldata/train.txt'):

    df = pd.read_csv(path,sep = ";", header=None)
    df.columns = ["features", "labels"]
    return df

# Process training dataset
# Takes pd dataframe returned by load_training_data()
# Minimally processes features
def extract_training_data(df):
    features = df["features"].values
    labels = df["labels"].values

    processed_features = []

    for sentence in range(0, len(features)):
        # Remove all the special characters
        processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

        # Remove all single characters
        processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

        # Substituting multiple spaces with single space
        processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

        # Convert to lowercase
        processed_feature = processed_feature.lower()

        processed_features.append(processed_feature)

    return processed_features, labels

# Turns processed_features into a Tfidf matrix  
def vectorize_feats(processed_features):
    vectorizer = TfidfVectorizer (max_features=2000, min_df=2, max_df=0.8, stop_words=stopwords.words('english'))
    processed_features = vectorizer.fit_transform(processed_features).toarray()

    return processed_features

# Splits dataset into a training dataset and a testing dataset where the training set is 80% of the total dataset and testing 20%
# Returns X_train, X_test, y_train, y_test where X_train and y_train are the features and labels for training respectively 
def split (processed_features, labels, text_size = 0.2, random_state =0):
    X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test

# Use RandomForestClassifier on training dataset to train model
# Returns a RandomForestClassifier object fit to training dataset
def train (X_train,y_train):
    text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    text_classifier.fit(X_train, y_train)

    return text_classifier

def save_model (classifier):
    pkl_filename = "pickle_model.pkl"

    with open(pkl_filename, 'wb') as file:
        pickle.dump(classifier, file)

## def load_model (Xtest, Ytest): 
##    Load from file
##    with open(pkl_filename, 'rb') as file:
##        pickle_model = pickle.load(file)
##        
##    Calculate the accuracy score and predict target values
##    score = pickle_model.score(Xtest, Ytest)
##    print("Test score: {0:.2f} %".format(100 * score))
##    Ypredict = pickle_model.predict(Xtest)

def predict (text_classifier, X_test,y_test):
    predictions = text_classifier.predict(X_test)
    
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))
    print(accuracy_score(y_test, predictions))


dataframe = load_training_data()

processed_features, labels = extract_training_data(dataframe)

tfidf_processed_features = vectorize_feats (processed_features)

X_train, X_test, y_train, y_test = split (tfidf_processed_features, labels, text_size = 0.2, random_state =0)

classifier = train (X_train,y_train)

save_model(classifier)

predict (classifier, X_test,y_test)





