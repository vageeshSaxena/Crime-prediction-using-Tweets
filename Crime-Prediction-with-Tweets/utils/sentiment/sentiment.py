"""
Guidelines :

1) Call sentiment_of_document for finding the sentiment of some particular documents.
2) The following files work in support with tweet_feature_extractor.py file

"""

import random
import os


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

import pandas as pd

from utils.sentiment.tweets_feature_extractor import build_pipeline_steps

current_directory = os.path.dirname(__file__)

bigram_lexicon = current_directory + "/" + "bigrams-pmilexicon.txt"
unigram_lexicon = current_directory + "/" + "unigrams-pmilexicon.txt"


def read(path, tag):
    """
    Input :
        Path of the file along with the sentiment to be tagged
    Output :
        A list of tagged tweets.
    """
    with open(path, "r") as f:
        tweets = f.readlines()
    tweet_tag = [[tweet, tag] for tweet in tweets]
    return tweet_tag


def test(corpus):
    """
    Input : 
        Corpus with tagged positive and negative tweets.
    Output :
        Test the passed corpus against the logistic regression classifiers
        """
    random.seed(42)
    random.shuffle(corpus)
    tweets, labels = zip(*corpus)
    vectorizer = build_pipeline_steps(do_bigram_sent=True, do_unigram_sent=True,
                                      bigram_sent_file=bigram_lexicon, unigram_sent_file=unigram_lexicon)
    X = vectorizer.fit_transform(tweets)
    clf = LogisticRegression()
    scoring = ["f1_micro", "f1_macro", "precision_micro",
               "precision_macro", "recall_micro", "recall_macro"]
    f1_scores = cross_validate(clf, X, labels, cv=10, scoring=scoring, return_train_score=False)
    for score_name, scores in f1_scores.items():
        print("average {} : {}".format(score_name, sum(scores)/len(scores)))


def train(corpus):
    """
    Input : 
        Corpus with tagged positive and negative tweets.
    Output :
        Train the passed corpus against the logistic regression classifiers
        """
    random.shuffle(corpus)
    tweets, labels = zip(*corpus)
    vectorizer = build_pipeline_steps(do_bigram_sent=True, do_unigram_sent=True,
                                      bigram_sent_file=bigram_lexicon, unigram_sent_file=unigram_lexicon)
    X = vectorizer.fit_transform(tweets)
    clf = LogisticRegression()
    clf.fit(X, labels)
    return TweetClf(clf, vectorizer)


class TweetClf:
    """
    The following class vectorizes the tweets and return sentiment and confidence values for a single tweet/document of tweets.
    """
    def __init__(self, clf, vectorizer):
        self.classifier = clf
        self.vectorizer = vectorizer

    @property
    def clf(self):
        return self.classifier

    def vectorize(self, tweets):
        return self.vectorizer.transform(tweets)

    def predict(self, tweets):
        X = self.vectorize(tweets)
        return self.clf.predict(X)

    def predict_proba(self, tweets):
        X = self.vectorize(tweets)
        return self.clf.predict_proba(X)

    def score_document(self, tweets):
        probs = self.predict_proba(tweets)
        neg_score, pos_score = 0, 0
        for neg, pos in probs:
            neg_score += neg
            pos_score += pos
        return (pos_score - neg_score) / len(probs)


def calculate_sentiment_tweet(tweets):
    """input :
        pandas dataseries"""
    # return data.predict_proba([tweets])
    return data.score_document([tweets])

# For finding the sentiment of all the documents in all the grids


pos = current_directory + "/" + "positive.txt"
neg = current_directory + "/" + "negative.txt"
corpus = read(pos, 1) + read(neg, -1)
print("Length of the testing Corpus :",len(corpus))
print("Adding unigrams and bigrams sentiment scores \n")
testing = test(corpus)
data = train(corpus)
