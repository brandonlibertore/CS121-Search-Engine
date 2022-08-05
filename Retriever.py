from nltk.stem import PorterStemmer
from ast import literal_eval as make_tuple
from collections import defaultdict
from math import log
from math import sqrt
from time import time
import os
import json
import re


def retriever():
    # Initialize porter stemmer to be used on query.
    ps = PorterStemmer()

    # Open final inverted index to be used to find query searches
    final = open('final.txt', 'r')

    # Dictionary to store inverted index to memory
    final_dict = dict()

    # Document ID
    doc_count = 1

    # Dictionary to store document IDs to Urls
    url_dict = dict()

    # Store a dictionary into memory before any query begins that holds document ID to url for faster look up time
    current_dir = os.path.join(str(os.getcwd()), "DEV")
    for root, dirs, files in os.walk(current_dir):
        for file in files:
            f = open(os.path.join(root, file))
            data = json.load(f)
            url = data["url"]
            url_dict[doc_count] = url
            doc_count += 1

    # Store final inverted index into memory before any query begins for faster look up time
    for line in final:
        line = line.rstrip().split(":")
        final_dict[line[0]] = [make_tuple(x.strip()) for x in line[1].rstrip().split(", ")]
    final.close()

    # Prompt for Query
    query = input("Please enter your query: ")
    while query != "":
        start = time()  # Initialize start timer for when query begins

        qv = defaultdict(float)  # dictionary that holds tfidf scores for each query token (similar to a vector)

        doc_scores = defaultdict(float)  # {document id: final/retrieval score through dot product}

        intersect = defaultdict(dict)  # {doc_id: {query token: tf_idf, query token: tf_idf}}

        query_dict = dict()  # {query token: list of postings}

        # query_components = [ps.stem(x.lower()) for x in query.rstrip().split(" ")]  # list of query tokens
        query_components = [ps.stem(x.strip().rstrip().lower()) for x in re.split(r'(\s|\d+|\W+|_)', query) if x.strip().rstrip().lower() != '' and x.isalnum() is True]

        # For each token in the query, check if it exist in the final inverted index and grab its line
        # RETRIEVAL
        for tokens in query_components:
            try:
                query_dict[tokens] = final_dict[tokens]  # If Token exist, assign its list of postings
            except KeyError:
                pass

        query_dict = dict(sorted(query_dict.items(), key=lambda x: len(x[1])))  # sort dict by len of posting list

        key_list = [x for x in query_dict.keys()]  # list of the sorted keys by len of posting

        if len(query_dict) == 0:
            print("NO DOCUMENTS WERE FOUND")

        elif len(query_dict) >= 1:  # Handles ALL query cases with >= 1 word and at least 1 word exist in the final index
            # transform query_dict values of posting tuple to just the doc_id
            # The intersect dict contains all documents that have a combination of the query terms
            for token, list_of_postings in query_dict.items():
                for post in list_of_postings:
                    doc_id = post[0]
                    tf_idf = post[1]
                    intersect[doc_id][token] = tf_idf

            # Append tf_idf for tokens in the query into the vector query
            for token in key_list:
                tf_idf = (1 + log(query_components.count(token), 10)) * (
                    log(len(url_dict) / len(query_dict[token]), 10))
                qv[token] = tf_idf

            # {doc_id: {query token: tf_idf, query token: tf_idf}}
            # Loop over intersect to get score for every single document that contains some or all the query tokens
            for doc_id, token_tfidf_dict in intersect.items():
                score = 0
                q_i_sq_sum = 0
                d_i_sq_sum = 0
                for token, tf_idf in qv.items():
                    q_i_sq_sum += tf_idf**2
                    try:
                        d_i_sq_sum += token_tfidf_dict[token]**2
                        score += token_tfidf_dict[token] * tf_idf
                    except KeyError:
                        pass
                normalizing_const = sqrt(q_i_sq_sum) * sqrt(d_i_sq_sum)
                doc_scores[doc_id] = score / normalizing_const

        doc_scores = sorted(doc_scores.items(), key=lambda x: -x[1])

        #  Print the top 10 Urls based on retrieval score pertaining to the search query
        for i in range(10):
            try:
                print("Document ID: {},".format(doc_scores[i][0]), "Document URL: {},".format(url_dict[doc_scores[i][0]]), "Retrieval Score: {}".format(doc_scores[i][1]))
            except IndexError:
                pass
        end = time()
        total_time = end-start
        print("Total Time For Search: {}".format(total_time))

        query = input("Please enter your query: ")


if __name__ == "__main__":
    retriever()
