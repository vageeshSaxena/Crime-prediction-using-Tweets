import numpy as np
from sklearn.neighbors.kde import KernelDensity

from utils.consts import KDE_BANDWITH


def train_KDE_model(train_df, bandwith=KDE_BANDWITH):
    """
    Train KDE model based on coordinates of incidents.
    """

    kde = KernelDensity(bandwidth=bandwith,
                        metric='haversine',
                        kernel='gaussian',
                        algorithm='ball_tree')

    kde.fit(train_df[['latitude', 'longitude']] * np.pi / 180)

    return kde
