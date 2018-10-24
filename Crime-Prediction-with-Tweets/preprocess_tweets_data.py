#!/usr/bin/python3

import argparse
import logging
import json
import glob

import pandas as pd
from tqdm import tqdm

from utils.twokenize3 import tokenizeRawTweetText

from utils.consts import START_DATE, END_DATE

from utils.geo import enrich_with_chicago_grid_1000, \
    filter_by_chicago_coord

# adding one day to END_DATE because Pandas
# dates comparison is exlusive for less then (<=)
END_DATE = pd.to_datetime(END_DATE) + pd.DateOffset(1)


TWEETS_DATA_TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'

logging.basicConfig(level=logging.INFO)


def _extract_tweets_latlng(tweet):
    if tweet['geo']:
        return _extract_tweets_geo_latlng(tweet)
    elif tweet['place']['place_type'] == 'poi':
        return _extract_tweets_place_latlng(tweet)
    else:
        return pd.Series({'latitude': None,
                          'longitude': None})


def _extract_tweets_geo_latlng(tweet):
    return pd.Series({'latitude': tweet['geo']['coordinates'][0],
                      'longitude': tweet['geo']['coordinates'][1]})


def _extract_tweets_place_latlng(tweet):
    return pd.Series({'latitude': tweet['place']['bounding_box']['coordinates'][0][0][0],
                      'longitude': tweet['place']['bounding_box']['coordinates'][0][0][1]})


def process_json(json_path):

    logging.debug('Loading raw twitter data <{}>...'.format(json_path))
    tweets_data = pd.read_json(json_path)

    raw_tweets_count = len(tweets_data)

    logging.debug('Removing tweets with NaN id...')
    tweets_data = tweets_data.dropna(subset=['id'])
    logging.debug('#: {}'.format(len(tweets_data)))

    logging.debug('Removing duplicated tweets...')
    tweets_data = tweets_data.drop_duplicates(subset=['id'])
    logging.debug('#: {}'.format(len(tweets_data)))

    tweets_data['has_geo'] = ~tweets_data['geo'].isna()
    logging.debug('Countint geo-tag tweets...')
    geo_counts = sum(tweets_data['has_geo'])
    try:
        no_geo_place_counts = (tweets_data[~tweets_data['has_geo']]['place']
                               .apply(lambda x: x['place_type'])
                               .value_counts())
    except:
        print(json_path)
        sa
    logging.debug('Extracting tweets latitude & longitude...')
    tweets_data = pd.concat([tweets_data,
                             tweets_data.apply(
                                 lambda tweet: _extract_tweets_latlng(tweet),
                                 axis=1)],
                            axis=1)
    logging.debug('#: {}'.format(len(tweets_data)))

    logging.debug('Removing tweets without geo-data...')
    tweets_data = tweets_data.dropna(subset=['latitude'])
    logging.debug('#: {}'.format(len(tweets_data)))

    logging.debug('Arranging columns...')
    tweets_data = tweets_data[['id',
                               'created_at',
                               'latitude',
                               'longitude',
                               'text']]

    tweets_data = tweets_data.rename(index=str,
                                     columns={'created_at': 'timestamp'})
    logging.debug('#: {}'.format(len(tweets_data)))

    logging.debug('Filter by location...')
    tweets_data = filter_by_chicago_coord(tweets_data)
    logging.debug('#: {}'.format(len(tweets_data)))

    logging.debug('Filtering by time...')
    tweets_data['timestamp'] = pd.to_datetime(
        tweets_data['timestamp'],
        format=TWEETS_DATA_TIMESTAMP_FORMAT
    )
    logging.debug('#: {}'.format(len(tweets_data)))

    tweets_data = tweets_data[(tweets_data['timestamp'] >= START_DATE) &
                              (tweets_data['timestamp'] <= END_DATE)]
    logging.debug('#: {}'.format(len(tweets_data)))

    logging.debug('Enriching with Chicago 1000m x 1000m grid data...')
    tweets_data = enrich_with_chicago_grid_1000(tweets_data)
    logging.debug('#: {}'.format(len(tweets_data)))

    logging.debug('Tokenizing tweets...')
    tweets_data['tokens'] = tweets_data['text'].apply(tokenizeRawTweetText)
    logging.debug('#: {}'.format(len(tweets_data)))

    logging.debug('Sorting by time...')
    tweets_data = tweets_data.sort_values('timestamp')
    logging.debug('#: {}'.format(len(tweets_data)))

    return tweets_data, raw_tweets_count, geo_counts, no_geo_place_counts


def main(raw_tweets_data_wildcard_path, processed_tweets_data_path):
    tweets_data = pd.DataFrame()
    no_geo_place_counts = pd.Series(data=[0]*5,
                                    index=['admin', 'city', 'poi', 'neighborhood', 'country'])
    raw_tweets_count = 0
    geo_counts = 0

    logging.info('Processing tweets jsons...')
    for path in tqdm(glob.glob(raw_tweets_data_wildcard_path)):
        (single_tweets_data,
         single_raw_tweets_count,
         single_geo_counts,
         single_no_geo_place_counts) = process_json(path)

        tweets_data = pd.concat((tweets_data, single_tweets_data))

        raw_tweets_count += single_raw_tweets_count
        geo_counts += single_geo_counts
        no_geo_place_counts = no_geo_place_counts.add(
            single_no_geo_place_counts,
            fill_value=0)

    logging.info('# raw tweets: {}'.format(raw_tweets_count))
    logging.info('# final tweets: {}'.format(len(tweets_data)))
    logging.info('% final tweets: {}'.format(100 * len(tweets_data)
                                             / raw_tweets_count))

    logging.info('% with geo key: {}'.format(
        100 * geo_counts / raw_tweets_count)
    )

    logging.info('% with no geo key but with place:\n{}'.format(
        100 * no_geo_place_counts / raw_tweets_count)
    )

    logging.info('Saving processed tweets data...')
    tweets_data.to_csv(processed_tweets_data_path)

    logging.info('Done!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess Tweets Data.')
    parser.add_argument('raw', help='raw tweets data wildcard path')
    parser.add_argument('processed', help='processed tweets data path')

    args = parser.parse_args()

    main(args.raw, args.processed)
