import functools

import pandas as pd

from utils.geo import latlng2LDA_sentiment_chicago, latlng2LDA_topics_chicago, \
    enrich_with_chicago_grid_200, generate_chicago_threat_grid_list, \
    CHICAGO_THREAT_GRID_LIST
from utils.kde import train_KDE_model
from utils.lda import train_LDA_model
from utils.consts import LDA_TOPICS


def generate_tweets_docs(tweets_data):
    """
    Gropby tweets by spatial location into documents.
    """

    tweet_docs_groupby = tweets_data.groupby(('latitude_index', 'longitude_index'))
    tweet_docs = tweet_docs_groupby['tokens'].apply(lambda r: list(r))
    tweet_docs = tweet_docs.sort_index()
    return tweet_docs, tweet_docs_groupby


def filter_dataset_by_date_window(dataset, start_date, end_date):
    """
    Filtering given dataset based on adate window: start --> end.
    Inclusive in both sides.
    """

    return dataset[(dataset['timestamp'] >= start_date) &
                   ((dataset['timestamp'] <= end_date))]


def build_datasets_by_date_window(crimes_data, tweets_data, start_train_date, n_train_days):
    """
    Building the training and evaluation datasets given a date window,
    seperatly for crimes and tweets.
    """

    start_train_date_dt = pd.to_datetime(start_train_date)
    end_train_date_dt = start_train_date_dt + pd.DateOffset(n_train_days)
    evaluation_date_dt = end_train_date_dt + pd.DateOffset(1)

    crimes_train_dataset = filter_dataset_by_date_window(crimes_data,
                                                         start_train_date_dt,
                                                         end_train_date_dt)

    tweets_train_dataset = filter_dataset_by_date_window(tweets_data,
                                                         start_train_date_dt,
                                                         end_train_date_dt)

    crimes_evaluation_dataset = filter_dataset_by_date_window(crimes_data,
                                                              evaluation_date_dt,
                                                              evaluation_date_dt)

    return crimes_train_dataset, tweets_train_dataset, crimes_evaluation_dataset


def generate_one_step_train_dataset(crimes_dataset, tweets_dataset):
    """
    Generating the training dataset with all the features and thier models.
    """

    crimes_kde_model = train_KDE_model(crimes_dataset)

    tweets_docs, tweet_docs_groupby = generate_tweets_docs(tweets_dataset)

    average_sentiment_docs = tweet_docs_groupby['sentiment'].mean()

    latlng2LDA_tweet_sentiment_chicago = functools.partial(latlng2LDA_sentiment_chicago,
                                                           average_sentiment_docs=average_sentiment_docs)

    tweets_lda_model, doc_topics, tweets_vocabulary = train_LDA_model((tweets_docs
                                                                       .apply(lambda r: sum(r, []))
                                                                       .tolist()))

    latlng2LDA_tweet_topics_chicago = functools.partial(latlng2LDA_topics_chicago,
                                                        doc_topics=doc_topics,
                                                        docs=tweets_docs)

    train_dataset = pd.concat([enrich_with_chicago_grid_200(crimes_dataset[['latitude', 'longitude']]).assign(crime=True),
                               CHICAGO_THREAT_GRID_LIST.assign(crime=False)],
                              axis=0)

    train_dataset = train_dataset[['latitude', 'longitude',
                                   'latitude_index', 'longitude_index', 'crime']]

    train_dataset['KDE'] = crimes_kde_model.score_samples(
        train_dataset[['latitude', 'longitude']].as_matrix()
    )

    train_dataset['SENTIMENT'] = train_dataset.apply(lambda row: latlng2LDA_tweet_sentiment_chicago(
        row['latitude'],
        row['longitude']),
        axis=1)

    train_dataset[LDA_TOPICS] = train_dataset.apply(lambda row: pd.Series(latlng2LDA_tweet_topics_chicago(
        row['latitude'],
        row['longitude'])),
        axis=1)

    features_cols = ['KDE', 'SENTIMENT'] + LDA_TOPICS

    train_dataset = {
        'X': train_dataset[['latitude', 'longitude', 'latitude_index', 'longitude_index'] + features_cols],
        'Y': train_dataset['crime'],
        'KDE': crimes_kde_model,
        'SENTIMENT': average_sentiment_docs,
        'LDA': {'model': tweets_lda_model, 'vocabulary': tweets_vocabulary, 'docs': tweets_docs}
    }

    return train_dataset


def generate_one_step_evaluation_dataset(crimes_evaluation_dataset):
    """
    Generating the evaluation dataset (actual crime incidents).
    """

    evaluation_dataset = enrich_with_chicago_grid_200(crimes_evaluation_dataset)
    evaluation_dataset = evaluation_dataset[['latitude_index', 'longitude_index']]
    return evaluation_dataset


def generate_one_step_datasets(crimes_data, tweets_data, start_train_date, n_train_days):
    """
    Generating training and evaluation datasets for given date frame.
    """

    crimes_train_dataset, tweets_train_dataset, crimes_evaluation_dataset = build_datasets_by_date_window(crimes_data,
                                                                                                          tweets_data,
                                                                                                          start_train_date,
                                                                                                          n_train_days)
    train_dataset = generate_one_step_train_dataset(crimes_train_dataset, tweets_train_dataset)
    evaluation_dataset = generate_one_step_evaluation_dataset(crimes_evaluation_dataset)

    return train_dataset, evaluation_dataset
