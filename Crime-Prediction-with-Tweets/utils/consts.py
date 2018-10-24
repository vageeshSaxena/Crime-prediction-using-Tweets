import os

RAW_CRIMES_DATA_PATH = os.path.join('data', 'raw', 'Crimes_-_2001_to_present.csv')
PROCESSED_CRIMES_DATA_PATH = os.path.join('data', 'processed', 'crime_data.csv')

RAW_TWEETS_DATA_WILDCARD_PATH = ('"' +
                                 os.path.join('data', 'raw', 'tweets', '*.json') +
                                 '"')
PROCESSED_TWEETS_DATA_PATH = os.path.join('data', 'processed', 'tweets_data.csv')

CHICAGO_COORDS = {'ll': {'longitude': -87.94011,
                         'latitude': 41.64454},  # Lower Left Corner
                  'ur': {'longitude': -87.52413,
                         'latitude': 42.02303}}  # Upper Right Corner

START_DATE = '2017-12-08'
END_DATE = '2018-02-19'

CRIME_TYPE = 'THEFT'
DOCS_GEO_CELL_SIZE = 1000  # meter
FALSE_LABLE_DATASET_CELL_SIZE = 200


KDE_BANDWITH = 0.0005
KDE_LEVELS = 40

UTM_ZONE_NUMBER = 16
UTM_ZONE_LETTER = 'T'

LDA_PARAMS = {
    'n_components': 350,
    'verbose': 0,
    # 'max_iter': 10,
    'learning_method': 'batch',
    # 'learning_offset': 50.,
    # 'random_state': 42,
}

LDA_TOPICS = ['T{:03}'.format(i) for i in range(LDA_PARAMS['n_components'])]

CSV_DATE_FORMART = "%Y-%m-%d %H:%M:%S"

CHICAGO_GRID_THREAT_PATH = os.path.join('data', 'chicago_grid_200.pickle')
CHICAGO_DOCS_GRID_PATH = os.path.join('data', 'chicago_grid_1000.pickle')

# shapefile with neighborhood boundaries.
CHICAGO_NEIGHBORHOOD = 'data/shapefiles/neighborhood/geo_export_0d18d288-fb07-4743-8d14-780bc1108034.shp'

# shapefile with just the city boundary.
CHICAGO_BOUNDARY = 'data/shapefiles/boundary/geo_export_72bbf21c-442f-41e2-bd40-433d7fb7928d.shp'


# Visualization Constants
FIGURE_SIZE = (13, 15)

SCATTER_SIZE_OF_CHICAGO_CITY = 0.75
SCATTER_SIZE_OF_CRIME_POINTS = 1.5

CITY_MAP_ORDER = 2

CITY_MAP_COLOR = 'black'
CONTOUR_PLOT_COLOUR = 'Reds'
CRIME_POINTS_COLOR = 'red'

CHICAGO_THREAT_GRID_SIZE = (209,175)
CHICAGO_DOCS_GRID_SIZE = (42,35)