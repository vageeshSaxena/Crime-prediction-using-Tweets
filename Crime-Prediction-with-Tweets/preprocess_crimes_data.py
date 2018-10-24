#!/usr/bin/python3

import argparse
import logging
import sys

import pandas as pd

from utils.consts import CRIME_TYPE, \
    START_DATE, END_DATE


from utils.geo import filter_by_chicago_coord

# adding one day to END_DATE because Pandas
# dates comparison is exlusive for less then (<=)
END_DATE = pd.to_datetime(END_DATE) + pd.DateOffset(1)

CHICAGO_CRIMES_DATA_TIMESTAMP_FORMAT = '%m/%d/%Y %H:%M:%S %p'

logging.basicConfig(level=logging.INFO)


def main(raw_crimes_data_path, processed_crimes_data_path):
    logging.info('Loading raw crime data...')
    crimes_data = pd.read_csv(raw_crimes_data_path)
    logging.info('#: {}'.format(len(crimes_data)))

    logging.info('Arranging columns...')
    crimes_data = (crimes_data[['Date',
                                'Primary Type',
                                'Latitude',
                                'Longitude']])

    crimes_data = crimes_data.rename(index=str,
                                     columns={'Date': 'timestamp',
                                              'Primary Type': 'type',
                                              'Latitude': 'latitude',
                                              'Longitude': 'longitude'})

    logging.info('Filtering by time...')
    crimes_data['timestamp'] = pd.to_datetime(
        crimes_data['timestamp'],
        format=CHICAGO_CRIMES_DATA_TIMESTAMP_FORMAT
    )
    logging.info('#: {}'.format(len(crimes_data)))

    crimes_data = crimes_data[(crimes_data['timestamp'] >= START_DATE) &
                              (crimes_data['timestamp'] <= END_DATE)]
    logging.info('#: {}'.format(len(crimes_data)))

    logging.info('Filtering by location...')
    crimes_data = filter_by_chicago_coord(crimes_data)
    logging.info('#: {}'.format(len(crimes_data)))

    """
    crime_types_probs = crimes_data['type']
                        .value_counts(normalize=True, ascending=True)
    crime_types_precents = crime_types_probs * 100
    crime_types_precents = crime_types_precents[crime_types_precents > 1]
    crime_types_precents.astype(int)
    crime_types_precents.plot(kind='barh', color='blue')
    plt.title('Crimes Types Precentages')
    plt.xlabel('%')
    plt.ylabel('Crime Types')
    plt.xticks(range(0, 30, 5))
    """

    logging.info('Filtering by crime type...')
    crimes_data = crimes_data[crimes_data['type'] == CRIME_TYPE]
    logging.info('#: {}'.format(len(crimes_data)))

    logging.info('Selecting columns...')
    crimes_data = crimes_data[['timestamp', 'latitude', 'longitude']]
    logging.info('#: {}'.format(len(crimes_data)))

    """
    (pd.Series(np.ones(len(crimes_data)), index=crimes_data['timestamp'])
     .resample('D')
     .sum()
     .plot())


    plt.title('Number of Theft Crime Incidents per Date')
    plt.xlabel('Date')
    plt.ylabel('Number of Theft Crime Incidents')
    """

#    logging.info('Enriching with Chicago grid data...')
#    crimes_data = enrich_with_chicago_grid_1000(crimes_data)

    logging.info('Sorting by time...')
    crimes_data = crimes_data.sort_values('timestamp')
    logging.info('#: {}'.format(len(crimes_data)))

    logging.info('Saving processed crime data...')
    crimes_data.to_csv(processed_crimes_data_path)

    logging.info('Done!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess Crimes Data.')
    parser.add_argument('raw', help='raw crime data path')
    parser.add_argument('processed', help='processed crimes data path')

    args = parser.parse_args()

    main(args.raw, args.processed)
