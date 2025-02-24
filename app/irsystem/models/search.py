import pandas as pd
import statistics
import re
import numpy as np
import nltk
import csv
import json
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import time
import math
import requests
import app.irsystem.models.distance as dist
from nltk.tokenize import TreebankWordTokenizer
# import distance as dist


# Citation note: CS4300 Assignments were used as a reference when writing code
# for this project, such as our computation of the cosine similarity matrix.
# SKI RESORT REVIEW DATA SET FROM: https://www.kaggle.com/fredkellner/onthesnow-ski-area-reviews


def write_csv_to_json(csv_path="dataset/skiing/reviews.csv", json_path="dataset/skiing/reviews.json"):
    with open(csv_path, "r") as f:
        with open(json_path, "w") as j:
            reader = csv.reader(f)
            data = {}
            reviews = []
            isFirstLine = True
            for line in reader:
                if isFirstLine:
                    isFirstLine = False
                    continue
                row_num, state, area, reviewer, date, rating, text = line[
                    0], line[1], line[2], line[3], line[4], line[5], line[6]
                review = {
                    "state": state,
                    "area_name": area,
                    "reviewer_name": reviewer,
                    "review_date": date,
                    "rating": rating,
                    "text": text
                }
                data[row_num] = review
            json.dump(data, j)
    return


def load_json(file):
    with open(file, "r") as f:
        data = json.load(f)
        return data


def write_json(data, file):
    with open(file, "w") as f:
        json.dump(data, f)
    return


def build_vectorizer(max_n_terms=5000, max_prop_docs=0.8, min_n_docs=10):
    vectorizer = TfidfVectorizer(
        min_df=min_n_docs, max_df=max_prop_docs, max_features=max_n_terms, stop_words="english")
    return vectorizer


with open('dataset/skiing/area_name_data.json', 'r') as f:
    area_name_data = json.load(f)


class Model1:
    def __init__(self):
        self.ski_dict = {}
        return

    def load_data(self):
        df = pd.read_csv('app/irsystem/models/ski_reviews.csv', header=1)
        dataframe = pd.DataFrame(df)
        nump_data = dataframe.to_numpy()

        dict_ski = {}
        num_states = 0
        num_locs = 0

        for data in nump_data:
            state = str(data[1])
            ski_area = str(data[2])
            reviewer = str(data[3])
            date = data[4]
            rating = data[5]
            review = str(data[6])

            if state in dict_ski:
                if ski_area in dict_ski[state]:
                    dict_ski[state][ski_area].append(
                        (date, reviewer, rating, review, state))
                else:
                    dict_ski[state][ski_area] = [
                        (date, reviewer, rating, review, state)]
                    num_locs += 1
            else:
                num_states += 1
                num_locs += 1
                dict_ski[state] = {ski_area: [
                    (date, reviewer, rating, review, state)]}

        self.ski_dict = dict_ski

    def pre_vectorize(self, query):
        dict_ski = self.ski_dict
        # list of dicts where dict[ski_area] = [review1, review2, ...]
        state_locs = list(dict_ski.values())

        # list of state names
        state_names = list(dict_ski.keys())
        b = True
        location_names = []  # this accumulates the names of all ski areas
        reviews_by_loc = [('query', query)]
        for loc in state_locs:
            # loc is dictionary with keys=area names, values= list of reviews for each area
            # loc.keys() gives all the ski areas in this state
            # location_names.extend(list(loc.keys()))
            location_names += list(loc.keys())
            for site in loc:
                # site is the area name, loc[site] is the list of reviews
                site_revs = []
                site_str = " "
                # site_revs.extend([tup[3] for tup in loc[site]])
                # for each review, extract the text part of the review and add it to site_revs
                # site_revs is a list of all the text reviews for this area name
                site_revs += [tup[3] for tup in loc[site]]
                # site_str.join... is the concatenated string of all reviews' text
                # so we append a tuple (areaname, string of all reviews, dictionary)
                reviews_by_loc.append((site, site_str.join(site_revs), loc))

        # returns a list of tuples, one tuple for each ski area
        return reviews_by_loc

    def ski_site_to_index(self, reviews_by_loc):
        return {site: index for index, site in enumerate([site[0] for site in reviews_by_loc])}

    def ski_index_to_site(self, reviews_by_loc):
        return {index: site for index, site in enumerate([site[0] for site in reviews_by_loc])}

    def vectorize(self, reviews_by_loc, vectorizer=build_vectorizer(max_n_terms=5000, max_prop_docs=0.8, min_n_docs=10)):
        tfidf_vec = vectorizer
        # param is a list: [string of all reviews concatenated for each ski area]
        tfidf_mat = tfidf_vec.fit_transform(
            [site[1] for site in reviews_by_loc]).toarray()
        index_to_vocab = {i: v for i, v in enumerate(
            tfidf_vec.get_feature_names())}
        vocab_to_index = {v: i for i, v in enumerate(
            tfidf_vec.get_feature_names())}
        return tfidf_mat, index_to_vocab, vocab_to_index

    def get_cos_sim(self, loc1, loc2, input_mat, site_to_index=ski_site_to_index):
        """Returns the cosine similarity of reviews from two locations
        """
        loc1_tf = input_mat[site_to_index[loc1]]
        loc2_tf = input_mat[site_to_index[loc2]]
        cossim = np.dot(loc1_tf, loc2_tf) / \
            (np.linalg.norm(loc1_tf) * np.linalg.norm(loc2_tf))

        return cossim

    def build_sims_cos(self, tfidf_mat):
        trans_input = np.transpose(tfidf_mat)
        numer = np.dot(tfidf_mat, trans_input)
        denom = np.linalg.norm(tfidf_mat, axis=1)
        denom_new = denom[:, np.newaxis]
        denom_final = np.multiply(denom, denom_new)
        fin = np.divide(numer, denom_final)
        return fin

    def most_sim(self, sim_mat, ski_index_to_site, rev_by_loc):
        # the first row is the similarity between the query and each concatenated string by area
        most_sim = sim_mat[0]
        sim = []
        count = 0
        for i in most_sim:
            sim.append((i, ski_index_to_site[count]))
            count += 1
        # sim is a list of tuples: (similarity score, area name)
        x = sorted(sim, key=lambda x: x[0], reverse=True)
        # top_3_rankings = x[1:4]
        # top_3_locs = [a[1] for a in top_3_rankings]
        # return (top_3_locs)

        # return a list of (score, area name) in descending order of similarity
        # the first element of x is the query, so eliminate it
        return x[1:]

    def search(self, query, location=None, distance=None):
        vectorizer = build_vectorizer()
        self.load_data()
        reviews_by_loc = self.pre_vectorize(query)
        ski_site_to_index1 = self.ski_site_to_index(reviews_by_loc)
        ski_index_to_site1 = self.ski_index_to_site(reviews_by_loc)
        tfidf_mat, index_to_vocab, vocab_to_index = self.vectorize(
            reviews_by_loc, vectorizer)
        sim_mat = self.build_sims_cos(tfidf_mat)
        results = self.most_sim(sim_mat, ski_index_to_site1, self.ski_dict)
        return results


class Model2:
    def __init__(self):
        self.data = {}
        self.review_num_to_idx = {}
        return

    def load_data_from_json(self, file="dataset/skiing/reviews.json"):
        self.data = load_json(file)

    def get_tfidf_mat_and_idx(self, query, vectorizer=build_vectorizer()):
        reviews = [review['text'] for row_num, review in self.data.items()]
        index_to_area_name = {i: self.data[key]["area_name"]
                              for i, key in enumerate(self.data)}
        self.review_num_to_idx = {row_num: i for i,
                                  row_num in enumerate(self.data)}
        # index_to_area_name = { for row_num, _ in self.data}
        corpus = [query] + reviews
        # mat = vectorizer.fit_transform(corpus).toarray()
        mat = vectorizer.fit_transform(corpus)
        self.index_to_vocab = {i: v for i, v in enumerate(
            vectorizer.get_feature_names())}
        self.tfidf = mat
        return mat, index_to_area_name

    def build_cos_sim_vect(self, tfidf):
        query_row = tfidf[0]
        trans = np.transpose(tfidf)
        # print("query row", query_row.shape)
        # print("trans", trans.shape)

        # num = np.matmul(query_row, trans)
        num = query_row.dot(trans)
        # print("num", num.shape)

        # norm = np.sqrt(np.sum(np.square(tfidf), axis=1))
        norm = np.sqrt(tfidf.power(2).sum(axis=1))
        # print("norm", norm.shape)

        # replace norms=0 with 1, since the numerator is zero anyway
        norm[norm == 0] = 1
        # print("norm2", norm.shape)
        norm = norm.transpose()
        # print("norm3", norm.shape)
        result = (num/norm)
        result = result.transpose()
        result = result[1:]
        # print("result", result.shape, type(result))
        result = np.squeeze(np.asarray(result))
        # print("result2", result.shape, type(result))
        return result

    def intermediate(self, tuples):
        result = defaultdict(list)
        i = 0
        for score, area_name in tuples:
            result[area_name].append((i, score))
            i += 1
        for area_name in result:
            result[area_name].sort(key=lambda x: x[1], reverse=True)
        return result

    def average_sim_vect_by_area(self, sim_vect, index_to_area_name):
        # returns a sorted list of tuples (area_name, avg score)
        tuples = [(sim, index_to_area_name[i])
                  for i, sim in enumerate(sim_vect)]
        area_to_total, area_to_count = defaultdict(float), defaultdict(int)
        for score, area in tuples:
            area_to_total[area] += score
            area_to_count[area] += 1
        unsorted = [(area, area_to_total[area]/area_to_count[area])
                    for area in area_to_total]
        area_name_to_score_tup_lst = self.intermediate(tuples)

        def add_row_num_back(row_num, review):
            review["row_number"] = row_num
            return review
        area_name_to_reviews = {
            area_name: [add_row_num_back(
                tup[0], self.data[str(tup[0])]) for tup in tup_lst if str(tup[0]) in self.data]  # TODO figure out what to do with reviews that no longer exist
            for area_name, tup_lst in area_name_to_score_tup_lst.items()}
        return sorted(unsorted, key=lambda x: x[1], reverse=True), area_name_to_reviews

    def search(self, query, location, distance):
        self.load_data_from_json()
        self.mat, self.index_to_area_name = self.get_tfidf_mat_and_idx(query)
        self.cos_sim_vect = self.build_cos_sim_vect(self.mat)
        return self.average_sim_vect_by_area(self.cos_sim_vect, self.index_to_area_name)


class Model3(Model2):
    def __init__(self):
        super().__init__()

    def rank_weighted(self, lst, alpha=0.85, beta=0.1, gamma=0.05):
        # takes in a list of tuples (area_name, avg sim score)
        # returns a list of tuples (area_name, weighted score)
        # cosine similarity is [0,1], sentiment is [-1,1]
        # but rating is [0,5] so have to normalize it
        x = [(sim*100, (area_name_data[area_name]['average_rating']/5),
              area_name_data[area_name]['average_sentiment']) for area_name, sim in lst]
        y = [(alpha*sim, beta*(area_name_data[area_name]['average_rating']/5),
              gamma*area_name_data[area_name]['average_sentiment']) for area_name, sim in lst]
        result = [(area_name, alpha*sim + beta*(area_name_data[area_name]['average_rating']/5) +
                   gamma*area_name_data[area_name]['average_sentiment']) for area_name, sim in lst]
        for i in range(5):
            print(lst[i][0])
            print(lst[i][1], "-->", result[i][1])
            print("normalized:", x[i])
            print("weighted:", y[i], "\n\n")
        return lst

    def intermediate(self, tuples):
        # list of pairs (score, area_name)
        # returns { area_name: sorted list of pairs (review_id, score) }
        result = defaultdict(list)
        for score, area_name, idx in tuples:
            result[area_name].append((list(self.data.keys())[idx], score))
            # i += 1
        for area_name in result:
            result[area_name].sort(key=lambda x: x[1], reverse=True)
        return result

    def average_sim_vect_by_area(self, sim_vect, index_to_area_name):
        # returns a sorted list of tuples (area_name, avg score)
        tuples = [(sim, index_to_area_name[i], i)
                  for i, sim in enumerate(sim_vect)]
        area_to_total, area_to_count = defaultdict(float), defaultdict(int)
        for score, area, idx in tuples:
            area_to_total[area] += score
            area_to_count[area] += 1
        unsorted = [(area, area_to_total[area]/area_to_count[area])
                    for area in area_to_total]
        # unsorted = self.rank_weighted(unsorted)
        area_name_to_score_tup_lst = self.intermediate(tuples)

        def add_row_num_back(row_num, review):
            review["row_number"] = row_num
            return review
        area_name_to_reviews = {
            area_name: [add_row_num_back(
                tup[0], self.data[str(tup[0])]) for tup in tup_lst if str(tup[0]) in self.data]
            for area_name, tup_lst in area_name_to_score_tup_lst.items()}
        return sorted(unsorted, key=lambda x: x[1], reverse=True), area_name_to_reviews

    def search(self, query, location, distance):
        self.load_data_from_json()
        self.mat, self.index_to_area_name = self.get_tfidf_mat_and_idx(query)
        self.cos_sim_vect = self.build_cos_sim_vect(self.mat)
        return self.average_sim_vect_by_area(self.cos_sim_vect, self.index_to_area_name)

    def add_important_similarity_words(self, results, query, k=10):
        # compute the big terms in the dot product of this review and query
        # get idx from results, idx of query is 0
        for result in results:
            review = result['reviews'][0]
            # we have to add one because the query is now index zero
            # idx = int(review['row_number']) + 1
            # TODO: this is not the idx anymore

            idx = self.review_num_to_idx[review["row_number"]] + 1
            row = self.tfidf[idx]

            # print(row.shape, self.tfidf[0].shape)

            # row = np.squeeze(row.toarray())
            # x = np.squeeze(self.tfidf[0].toarray())
            # print(row.shape, x.shape)
            # elem_wise_prod = np.multiply(row, self.tfidf[0])
            x = self.tfidf[0]
            elem_wise_prod = row.multiply(x)
            elem_wise_prod = np.squeeze(elem_wise_prod.transpose().toarray())
            # print(elem_wise_prod.shape)
            # elem_wise_prod = np.multiply(row, self.tfidf[0])
            words = [(elem_wise_prod[i], self.index_to_vocab[i])
                     for i in range(len(elem_wise_prod)) if elem_wise_prod[i] > 0]
            words = sorted(words, key=lambda x: x[0], reverse=True)[:k]
            important_word_set = set([word for _, word in words])
            # print(important_word_set)
            out, i = [], 0

            def tokenize_word(s):
                x = re.sub(r'\W', ' ', s)
                x = re.sub(r'\s+', ' ', x, flags=re.I)
                return x.lower()
            tokenized = review['text'].split(' ')
            isImportant, idx = None, 0
            for x in tokenized:
                text_word = x.lower()
                if ((text_word in important_word_set) and (isImportant == True)):
                    # continue
                    idx += 1
                elif ((text_word not in important_word_set) and (isImportant == False)):
                    # also continue
                    idx += 1
                else:
                    if isImportant != None:
                        out.append((isImportant, idx))
                    isImportant = (text_word) in important_word_set
                    idx = i
                i += 1
            out.append((isImportant, idx))
            result["important_words"] = out
        return


def search_q(query, version, location=None, distance=None):
    if version == 1:
        model = Model1()
        data = model.search(query)
        data = [tup[1] for tup in data]
        return data[:3]
    elif version == 2:
        # assert (location is not None and distance is not None)
        model = Model2()
    else:
        model = Model3()

    scores, area_name_to_sorted_reviews = model.search(
        query, location, distance)
    # area_to_distance = dist.getDistanceForAreas(location)
    # if "error" in area_to_distance.keys():
    #     return area_to_distance
    results = [{
        "version": version,
        "area_name": area_name,
        "state": area_name_data[area_name]["state"],
        # "distance":round(area_to_distance[area_name]),
        "distance": 0,
        "score": round(score*100, 2),
        "reviews": area_name_to_sorted_reviews[area_name][:3],
        "sentiment": round(float(area_name_data[area_name]['average_sentiment']), 2),
        "rating": round(float(area_name_data[area_name]['average_rating']), 2),
        "most_positive_reviews": area_name_data[area_name]['top_10_positive'],
        "most_negative_reviews": area_name_data[area_name]['top_10_negative'],
        "emotion_numbers": area_name_data[area_name]['emotions'],
        "important_words": [],
        "number_of_reviews": area_name_data[area_name]['number_of_reviews'],
        "query": query
        # } for area_name, score in scores if area_to_distance[area_name] <= distance or distance == 1000]
    } for area_name, score in scores]
    results = results[:min(5, len(results))]
    if version == 3:
        model.add_important_similarity_words(results, query)
    if results == []:
        x = {"error": "Your search did not return any results. Try expanding your location range or changing your query."}
        return x
    return results
