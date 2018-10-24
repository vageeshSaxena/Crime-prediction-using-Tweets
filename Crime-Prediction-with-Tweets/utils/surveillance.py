import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.geo import N_CHICAGO_THREAT_GRID_LIST
from utils.consts import START_DATE, END_DATE
from utils.threat_models import generate_threat_datasets
from utils.datasets_generation import generate_one_step_datasets


def generate_surveillance_data(train_dataset, evaluation_dataset):
    """
    Return the surveillance data and threat datasets for a given
    training & evluation time frame datasets.
    """

    surveillance_data = np.zeros((5, N_CHICAGO_THREAT_GRID_LIST))

    threat_datasets = generate_threat_datasets(train_dataset)

    crime_counts = evaluation_dataset.groupby(['latitude_index', 'longitude_index']).size()
    crime_counts = crime_counts.sort_values(ascending=False)

    # real crime occurence is our ORACLE dataset
    threat_datasets['ORACLE'] = {'cells': list(crime_counts.index)}
    threat_datasets.move_to_end('ORACLE')

    for threat_model_index, (threat_model_name, threat_dataset) in enumerate(threat_datasets.items()):
        for cell_index, (latitude_index, longitude_index) in enumerate(threat_dataset['cells']):
            surveillance_data[threat_model_index][cell_index] = crime_counts.get(
                (latitude_index, longitude_index), 0)

    return surveillance_data, threat_datasets


def generate_one_step_surveillance_data(crimes_data, tweets_data, start_train_date, n_train_days):
    """
    Return the surveillance data and threat datasets for a given
    training & evluation time frame.
    """

    train_dataset, evaluation_dataset = generate_one_step_datasets(crimes_data,
                                                                   tweets_data,
                                                                   start_train_date,
                                                                   n_train_days)

    surveillance_data, threat_datasets = generate_surveillance_data(train_dataset,
                                                                    evaluation_dataset)

    return surveillance_data, threat_datasets


def generate_all_data_surveillance_data(crimes_data, tweets_data, n_train_days):
    """
    Return the aggregated surveillance data and threat datasets for a given
    training & evluation time frame.
    """

    agg_surveillance_data = np.zeros((5, N_CHICAGO_THREAT_GRID_LIST))
    all_threat_datasets = []

    start_train_dates = pd.date_range(START_DATE, END_DATE)[:-(n_train_days+1)]

    for start_train_date in tqdm(start_train_dates):

        surveillance_data, threat_datasets = generate_one_step_surveillance_data(crimes_data,
                                                                                 tweets_data,
                                                                                 start_train_date,
                                                                                 n_train_days)

        agg_surveillance_data += surveillance_data
        all_threat_datasets.append((start_train_date, threat_datasets))

    agg_surveillance_data = agg_surveillance_data.cumsum(
        axis=1) / agg_surveillance_data.sum(axis=1)[:, None]

    return agg_surveillance_data, all_threat_datasets


def calc_AUCs(agg_surveillance_data, model_names):
    """
    Calculate the Area Under the Curve (AUC) for all the pairs for models.
    """

    model_names_list = list(model_names)
    aucs = pd.DataFrame(columns=model_names_list[:-1], index=model_names_list[1:], dtype=float)

    for (index1, name1), (index2, name2) in itertools.combinations(
        enumerate(
            reversed(model_names),  start=1
        ),
            2):
        auc = (agg_surveillance_data[-index1] - agg_surveillance_data[-index2]
               ).sum() / agg_surveillance_data.shape[1]
        aucs.loc[name1, name2] = '{:.6f}'.format(auc)

    aucs = aucs.fillna('')

    return aucs
