import os
import math
import functools
import itertools

import numpy as np
import pandas as pd
import utm
import shapefile as shp
from shapely.geometry import Point, Polygon


from utils.consts import CHICAGO_COORDS, DOCS_GEO_CELL_SIZE, \
    UTM_ZONE_NUMBER, UTM_ZONE_LETTER, \
    FALSE_LABLE_DATASET_CELL_SIZE, LDA_PARAMS, CHICAGO_BOUNDARY, \
    CHICAGO_GRID_THREAT_PATH, CHICAGO_DOCS_GRID_PATH


CHICAGO_SHAPEFILE_PATH = os.path.join(os.path.dirname(__file__), 'blalbashapefile')


def filter_by_geo_coord(df, bounderies):
    """
    Filtering Dataframe rows by latitude and longitude.
    """

    return df[(df['latitude'] >= bounderies['ll']['latitude']) &
              (df['latitude'] <= bounderies['ur']['latitude']) &
              (df['longitude'] >= bounderies['ll']['longitude']) &
              (df['longitude'] <= bounderies['ur']['longitude'])]


def _latlng2utm(coords):
    utm_coord = utm.from_latlon(coords['latitude'], coords['longitude'])[:2]
    return dict(
        zip(
            ('latitude', 'longitude'),
            utm_coord
        )
    )


def _utm2latlng(coords):
    utm_coord = utm.to_latlon(coords['latitude'], coords['longitude'],
                              UTM_ZONE_NUMBER, UTM_ZONE_LETTER)
    return dict(
        zip(
            ('latitude', 'longitude'),
            utm_coord
        )
    )


def _generate_utm_columns(row):
    return pd.Series(_latlng2utm(row))


def latlng2grid_cords(latitude, longitude, bounderies_utm, cell_size):
    utm_cords = _latlng2utm({'latitude': latitude,
                             'longitude': longitude})

    latitude_index = int(((utm_cords['latitude'] - bounderies_utm['ll']['latitude'])
                          / cell_size))

    longitude_index = int(((utm_cords['longitude'] - bounderies_utm['ll']['longitude'])
                           / cell_size))

    return latitude_index, longitude_index


def bounderis_latlng2utm(bounderies):
    return {'ll': _latlng2utm(bounderies['ll']),
            'ur': _latlng2utm(bounderies['ur'])}


def enrich_with_grid_coords(df, bounderies, cell_size):
    '''
    The accepts a data frame which has atleast 2 columns with names 'Latitude' and
    'Longitude'. It will be converted into UTM(Universal Transverse Mercator) co-odrinates for obtaining
    grid of a locality.

    input:
    Data frame with 'Latitude' and 'Longitude'

    output:
    Data frame with additional column which represents Grid Numbers
    '''

    bounderies_utm_cords = bounderis_latlng2utm(bounderies)

    # n_latitude_cells = int(math.ceil((bounderies_utm_cords['ur']['latitude'] -
    #                                  bounderies_utm_cords['ll']['latitude'])
    #                                 / cell_size))

    utm_coords = df[['latitude', 'longitude']].apply(lambda row: _generate_utm_columns(row), axis=1)

    df.loc[:, 'latitude_index'] = (((utm_coords['latitude'] - bounderies_utm_cords['ll']['latitude'])
                                    / cell_size)
                                   .astype(int))

    df.loc[:, 'longitude_index'] = (((utm_coords['longitude'] - bounderies_utm_cords['ll']['longitude'])
                                     / cell_size)
                                    .astype(int))

    # df['cell_index'] = df['longitude_index'] * n_latitude_cells + df['latitude_index']

    return df


def generate_grid_list(cell_size):
    """
    Generate list of all the cells of the grid in the box bounderis of Chicago.
    """

    utm_latitude_dim = np.arange(CHICAGO_UTM_COORDS['ll']['latitude'],
                                 CHICAGO_UTM_COORDS['ur']['latitude'],
                                 cell_size)

    utm_longitude_dim = np.arange(CHICAGO_UTM_COORDS['ll']['longitude'],
                                  CHICAGO_UTM_COORDS['ur']['longitude'],
                                  cell_size)

    grid_list = []
    for (lat_ind, lat), (lng_ind, lng) in itertools.product(enumerate(utm_latitude_dim),
                                                            enumerate(utm_longitude_dim)):

        grid_cord = _utm2latlng({'latitude': lat, 'longitude': lng})
        grid_cord['latitude_index'] = lat_ind
        grid_cord['longitude_index'] = lng_ind

        grid_list.append(grid_cord)

    return pd.DataFrame(grid_list)


def utm_city_boundary():
    """
    Accepts only BOUNDARY SHAPEFILES!!
    """
    shpfile = shp.Reader(CHICAGO_BOUNDARY)
    chicago = shpfile.shapeRecords()[0].shape.points
    chicago_utm = []
    for i in range(0, len(chicago)):
        chicago_utm.append(utm.from_latlon(chicago[i][1], chicago[i][0])[0:2])
    return chicago_utm


def generate_grid(distance_cell_size):
    """
    Getting the grids of 1000 meters square.

    parameters: minx, miny, maxx, maxy

    returns: grid of the
    """
    dx = distance_cell_size
    dy = distance_cell_size

    nx = int(
        math.ceil(abs(CHICAGO_UTM_COORDS['ur']['latitude'] - CHICAGO_UTM_COORDS['ll']['latitude'])/dx))
    ny = int(
        math.ceil(abs(CHICAGO_UTM_COORDS['ur']['longitude'] - CHICAGO_UTM_COORDS['ll']['longitude'])/dy))
    grid = []
    lat_long_index = []

    for i in range(ny):
        for j in range(nx):
            vertices = []
            vertices.append([CHICAGO_UTM_COORDS['ll']['latitude']+dx*j,
                             CHICAGO_UTM_COORDS['ll']['longitude']+dy*i])
            vertices.append([CHICAGO_UTM_COORDS['ll']['latitude']+dx*(j+1),
                             CHICAGO_UTM_COORDS['ll']['longitude']+dy*i])
            vertices.append([CHICAGO_UTM_COORDS['ll']['latitude']+dx*(j+1),
                             CHICAGO_UTM_COORDS['ll']['longitude']+dy*(i+1)])
            vertices.append([CHICAGO_UTM_COORDS['ll']['latitude']+dx*j,
                             CHICAGO_UTM_COORDS['ll']['longitude']+dy*(i+1)])
            grid.append(vertices)
            lat_long_index.append([i, j])
    return grid, lat_long_index


def generate_grid_list2(cell_size):
    """

    """
    green_grid = []
    grids, ll_index = generate_grid(cell_size)

    chicago_utm = utm_city_boundary()

    poly = Polygon(chicago_utm)
    for i in range(0, len(grids)):
        for g in grids[i]:
            if(poly.contains(Point(g))):
                cords = _utm2latlng({'latitude': grids[i][0][0],
                                     'longitude': grids[i][0][1]})
                cords['latitude_index'] = ll_index[i][0]
                cords['longitude_index'] = ll_index[i][0]
                green_grid.append(cords)
                break

    return pd.DataFrame(green_grid)  # .drop_duplicates()


def latlng2LDA_topics_chicago(latitude, longitude, doc_topics, docs):
    """
    Return the topic vector in a given latitude and longtitude,
    by the geo document groupby.
    """

    latitude_index, longitude_index = latlng2grid_docs_cords_chicago(latitude, longitude)

    if (latitude_index, longitude_index) in docs.index:
        doc_index = docs.index.get_loc((latitude_index, longitude_index))
        return doc_topics[doc_index]
    else:
        #raise KeyError
        return np.zeros(LDA_PARAMS['n_components'])


def latlng2LDA_sentiment_chicago(latitude, longitude, average_sentiment_docs):
    """
    Return the sentiment value in a given latitude and longtitude,
    by the geo document groupby.
    """

    latitude_index, longitude_index = latlng2grid_docs_cords_chicago(latitude,
                                                                     longitude)

    if (latitude_index, longitude_index) in average_sentiment_docs.index:
        return average_sentiment_docs[(latitude_index, longitude_index)]
    else:
        # raise KeyError
        return 0.


CHICAGO_UTM_COORDS = bounderis_latlng2utm(CHICAGO_COORDS)

enrich_with_chicago_grid_1000 = functools.partial(enrich_with_grid_coords,
                                                  bounderies=CHICAGO_COORDS,
                                                  cell_size=DOCS_GEO_CELL_SIZE)

enrich_with_chicago_grid_200 = functools.partial(enrich_with_grid_coords,
                                                 bounderies=CHICAGO_COORDS,
                                                 cell_size=FALSE_LABLE_DATASET_CELL_SIZE)

filter_by_chicago_coord = functools.partial(filter_by_geo_coord,
                                            bounderies=CHICAGO_COORDS)

latlng2grid_docs_cords_chicago = functools.partial(latlng2grid_cords,
                                                   bounderies_utm=CHICAGO_UTM_COORDS,
                                                   cell_size=DOCS_GEO_CELL_SIZE)

generate_chicago_docs_grid_list = functools.partial(generate_grid_list,
                                                    # bounderies_utm=CHICAGO_UTM_COORDS,
                                                    cell_size=DOCS_GEO_CELL_SIZE)

generate_chicago_threat_grid_list = functools.partial(generate_grid_list,
                                                      # bounderies_utm=CHICAGO_UTM_COORDS,
                                                      cell_size=FALSE_LABLE_DATASET_CELL_SIZE)

if not os.path.exists(CHICAGO_GRID_THREAT_PATH):
    print('Generating threat grid...')
    CHICAGO_THREAT_GRID_LIST = generate_chicago_threat_grid_list()
    CHICAGO_THREAT_GRID_LIST.to_pickle(CHICAGO_GRID_THREAT_PATH)
else:
    CHICAGO_THREAT_GRID_LIST = pd.read_pickle(CHICAGO_GRID_THREAT_PATH)

N_CHICAGO_THREAT_GRID_LIST = len(CHICAGO_THREAT_GRID_LIST)


if not os.path.exists(CHICAGO_DOCS_GRID_PATH):
    print('Generating docs grid...')
    CHICAGO_DOCS_GRID_LIST = generate_chicago_docs_grid_list()
    CHICAGO_DOCS_GRID_LIST.to_pickle(CHICAGO_DOCS_GRID_PATH)
else:
    CHICAGO_DOCS_GRID_LIST = pd.read_pickle(CHICAGO_DOCS_GRID_PATH)

N_CHICAGO_DOCS_GRID_LIST = len(CHICAGO_DOCS_GRID_LIST)
