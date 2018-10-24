#!/usr/bin/python3

import pandas as pd

from utils.consts import LDA_PARAMS, LDA_TOPICS, PROCESSED_TWEETS_DATA_PATH, CSV_DATE_FORMART
from utils.datasets_generation import generate_tweets_docs
from utils.lda import train_LDA_model


def main():
    tweets_data = pd.read_csv(PROCESSED_TWEETS_DATA_PATH)
    tweets_data['timestamp'] = pd.to_datetime(
        tweets_data['timestamp'], format=CSV_DATE_FORMART).dt.normalize()
    tweets_data['tokens'] = tweets_data['tokens'].apply(lambda x: eval(x))

    tweets_docs, _ = generate_tweets_docs(tweets_data)

    lda_params = LDA_PARAMS.copy()
    lda_params['n_components'] = 500

    print('Training LDA with {} topics...'.format(lda_params['n_components']))
    tweets_lda_model, doc_topics, tweets_vocabulary = train_LDA_model((tweets_docs
                                                                       .apply(lambda r: sum(r, []))
                                                                       .tolist()),
                                                                      params=lda_params)
    topics_words = [tweets_lda_model.components_[topic_index]
                    for topic_index in range(LDA_PARAMS['n_components'])]

    n_trivial_topics = [len(set(topic_words)) for topic_words in topics_words].count(1)

    print('#topics: ', lda_params['n_components'])
    print('#effective topics (i.e. withoug unifrom distribution on words): ',
          lda_params['n_components']-n_trivial_topics)


if __name__ == '__main__':
    main()
