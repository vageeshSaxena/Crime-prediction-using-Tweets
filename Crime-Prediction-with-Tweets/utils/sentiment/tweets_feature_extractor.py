import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer


def build_pipeline_steps(do_bigram_sent, do_unigram_sent, bigram_sent_file, unigram_sent_file):
    """
    The function uses FunctionTransformer method from sklearn to provide precision,recall and f1-score values
    for unigrams and bigrams.
        """
    features = []
    #print("Adding ngram features : ngram_range 2")
    text_pipeline = ('text_pipeline', Pipeline([('ngrams', CountVectorizer(
        stop_words="english", ngram_range=(1, 2), preprocessor=str.split, tokenizer=lambda x:x))]))
    features.append(text_pipeline)
    if do_bigram_sent:
        #print("Add bigram sentiment scores")
        bigram_sent_score_lookup = get_bigram_sentiments(bigram_sent_file)
        features.append(("bigram sentiment score", FunctionTransformer(
            score_document_bigrams, kw_args={'score_lookup': bigram_sent_score_lookup}, validate=False)))
    if do_unigram_sent:
        #print("Add unigram sentiment scores")
        unigram_sent_score_lookup = get_unigram_sentiments(unigram_sent_file)
        features.append(("unigram sentiment score", FunctionTransformer(
            score_document, kw_args={'score_lookup': unigram_sent_score_lookup}, validate=False)))
    pipeline_steps = Pipeline([("features", FeatureUnion(features))])
    return pipeline_steps


def tweet_score(tweet, score_lookup):
    """
    Input:
        tweets and score_lookup

    Output:
        count the score of the tweets"""
    score = 0
    for word in tweet:
        if word in score_lookup:
            score += score_lookup[word]
    return score


def score_document(tweets, score_lookup):
    return np.array([tweet_score(tw, score_lookup) for tw in tweets]).reshape(-1, 1)


def score_document_bigrams(tweets, score_lookup):
    return np.array([tweet_score(bigrams(tw), score_lookup) for tw in tweets]).reshape(-1, 1)


def bigrams(tokens):
    return [bg for bg in zip(tokens[:-1], tokens[1:])]


def get_bigram_sentiments(bigrams_path):
    """
    Input :
        Bigram lexical file path
    Output:
        Sentiment value for bigram model."""
    bigram_sentiments = {}
    # also doesnt work on windows without the encoding parameter
    with open(bigrams_path, encoding="utf-8") as infile:
        for line in infile:
            w1, w2, score, pos, neg = line.split()
            bigram_sentiments[w1, w2] = float(score)
    return bigram_sentiments


def get_unigram_sentiments(unigrams_path):
    """
    Input :
        Unigram lexical file path
    Output:
        Sentiment value for unigram model."""
    unigram_sentiments = {}
    # also doesnt work on windows without the encoding parameter
    with open(unigrams_path, encoding="utf-8") as infile:
        for line in infile:
            word, score, pos, neg = line.split()
            unigram_sentiments[word] = float(score)
    return unigram_sentiments
