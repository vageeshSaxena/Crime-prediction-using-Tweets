#!/usr/bin/python3

import sys
import pickle
import time

import pandas as pd

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from utils.consts import LDA_PARAMS

LDA_PARAMS['n_jobs'] = -1


def coalesce(token):
    """
    Klaues: why this function?
    """
    new_tokens = []
    for char in token:
        if len(new_tokens) < 2 or char != new_tokens[-1] or char != new_tokens[-2]:
            new_tokens.append(char)
    return ''.join(new_tokens)


def preprocess_tweet_for_LDA(raw_tokens):
    """
    text input is one string
    output is tokenized and preprocessed(as defined below) text

    lowercase
    no hashtags or mentions
    any url converted to "url"
    replace multiple repeated chars with 2 of them. eg paaaarty -> paarty
    """

    processed_tokens = []
    for token in raw_tokens:
        if token.startswith("@") or token.startswith("#"):
            continue
        elif token.startswith("https://") or token.startswith("http://"):
            processed_tokens.append("url")
        else:
            processed_tokens.append(coalesce(token))

    return processed_tokens


def train_LDA_model(docs, params=LDA_PARAMS, preprocessor=preprocess_tweet_for_LDA):
    print(params)
    vectorizer = CountVectorizer(stop_words="english",
                                 preprocessor=preprocessor,
                                 tokenizer=lambda x: x)

    lda_train_data = vectorizer.fit_transform(docs)

    lda_model = LatentDirichletAllocation(**params)

    lda_model.fit(lda_train_data)

    doc_topics = lda_model.transform(lda_train_data)

    vocabulary = vectorizer.get_feature_names()

    return lda_model, doc_topics, vocabulary


if __name__ == '__main__':
    docs_path = sys.argv[1]
    result_path = sys.argv[2]
    print('Load', time.time())
    docs = pickle.load(open(docs_path, 'rb'))
    print('Train', time.time())
    result = train_LDA_model(docs)
    print('Save', time.time())
    pickle.dump(result, open(result_path, 'wb'))
    print('Done', time.time())

"""
import pickle

def train_LDA_model2(docs):
    pickle.dump(docs, open('./docs.pickle', 'wb'))
    !python3 ./utils/lda.py ./docs.pickle ./result.pickle
    result = pickle.load(open('./result.pickle', 'rb'))
    return result
"""
