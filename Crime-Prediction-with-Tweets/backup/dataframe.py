"""
Conventions to follow :

1) Place the downloaded tweets(JSON files) on your desktop screen inside a folder named as "tweets_json".
2) Call final_dataset_function to get the processed dataframe and dictionary with assigned sentiment values to each 
   document and LSA dictionary with their associated topics.
3) The following files work in support with the sentiment.py,tweet_feature_extractor.py and twokenize3.py file

Note : The execution of the program takes some time(like 15 min approximately), have patience.
   
"""


# Installing dependencies

""" 
! pip install numpy
! pip install pandas
! pip install utm
! pip install tqdm
! pip install shapely
! pip install sklearn
"""

# Importing libraries
import glob,os, sys
import pandas as pd
from tqdm import tqdm_notebook
import numpy as np

import matplotlib.pyplot as plt
import utm, math
from shapely.geometry import Point, Polygon

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from twokenize3 import tokenizeRawTweetText

sent_directory = os.getcwd() + "/" + "sentiment"
sys.path.insert(0, sent_directory)
from sentiment import sentiment_of_document,find_sentiment_doc,calculate_sentiment_tweet


# Geospatial Grid
def generate_grid_number(df,offset,coors,xGrid):
    """
    The accepts a data frame which has atleast 2 columns with names 'Latitude' and 'Longitude'. 
    It will be converted into UTM(Universal Transverse Mercator) co-odrinates for obtaining grid of a locality.
        
    input:
       	Data frame with 'Latitude' and 'Longitude'
    
    output:
        Data frame with additional column which represents Grid Numbers
        
    requirements:
        import utm
        import pandas as pd
    """
    df = pd.concat([df, 
            df[['Latitude', 'Longitude']].apply(lambda r: pd.Series(dict(zip(('UTM Lat', 'UTM Long'), utm.from_latlon(r['Latitude'], r['Longitude'])[:2]))), axis=1)],
            axis=1)
    df['Lat_ind'] = ((df['UTM Lat'] -  coors['low_left_x']) / offset).astype(int)
    df['Long_ind'] = ((df['UTM Long'] -  coors['low_left_y']) / offset).astype(int)
    df['Grid Number'] = df['Long_ind'] * xGrid + df['Lat_ind']
    del df['UTM Lat']
    del df['UTM Long']
    return df

# Creates hdf5 documents on the basis of c2v values
def make_documents(c2v_values,dataset):
    desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
    os.chdir(desktop_path)
    directory = "hdf5_Doc"
    if not os.path.exists(directory):
        os.makedirs(directory)
    os.chdir(desktop_path + "/" + directory)
    for values in c2v_values:
        doc = dataset[dataset["Grid Number"] == values]
        filename = str(values) + ".hdf5"
        doc.reset_index(drop=True)
        doc.to_hdf(filename,'key')

# Processing dataframe
def create_dataframe():
    desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
    tweets_path = desktop_path + "/tweets_json/*.json"
    tweets = pd.concat(map(pd.read_json, tqdm_notebook(glob.glob(tweets_path))))
    return tweets

# Removing the null values from the input dataframe
def filter_dataframe(data):
    # coordinates = longitude
    # geo = lattitude
    headers = ["geo", "text","lang","timestamp_ms"]
    df = data[headers]
    for columns in headers:
        df = df[df[columns].notnull()]
    # Considering the tweets only in english language
    return df[df["lang"] == "en"].reset_index(drop=True)

# Converts the coordinates to their respective grid index
def c2v(dataset,offset,coors,xGrid):
    """ Below are the set of constants that are being used in the Project includes :
    - offset for generating the grid
    - lower left and upper right co-ordinates of a locality(in this case, Chicago)
    """
    latitude,longitude = ([] for i in range(2))
    for values in dataset["geo"]:
        latitude.append(values["coordinates"][0])
        longitude.append(values["coordinates"][1])
    dataset["Latitude"] = latitude
    dataset["Longitude"] = longitude
    dataset = generate_grid_number(dataset,offset,coors,xGrid)
    del dataset['geo']
    return dataset

# Initialising dataframe
def initialise_dataframe():
    df = create_dataframe()
    df = filter_dataframe(df)
    return df

# Process the dataframe to get sentiment and tokenized values
def process_dataframe(dataset,offset,coors,xGrid):
    myDic1,myDic2 = ({} for i in range(2))
    tweetList = []
    dataset = c2v(dataset,offset,coors,xGrid)
    dataset["sentiment_text"] = dataset["text"].apply(calculate_sentiment_tweet)
    dataset["twokenized_text"] = dataset["text"].apply(tokenizeRawTweetText)
    dataset.dropna(inplace=True)
    dataset.sort_values(['Grid Number'],inplace=True)
    dataset["Grid Number"] = dataset["Grid Number"].astype(int)
    c2v_list = dataset["Grid Number"].unique()
    make_documents(c2v_list,dataset)
    desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
    directory = desktop_path + "/" + "hdf5_Doc"
    for filename in os.listdir(os.getcwd()):
        if filename.endswith(".hdf5"):
            temp = []
            doc = pd.read_hdf(filename)
            for values in doc["text"]:
                temp.append(values)
            tweetList.append(temp)
            myDic1[str(filename)] = find_sentiment_doc(tweetList[-1])
            myDic2[str(filename)] = tweetList[-1]
        else:
            pass
    return dataset,myDic1,myDic2,tweetList

# Vectorization of the documents
def vectorizer(document,vectorizer):
    matrix = vectorizer.fit_transform(document)
    return matrix.shape[0], matrix, vectorizer

# Decompostion of the matrix
def decomposition(shape,matrix,vectorizer,dic_keys,lsa_dict):
    lsa = TruncatedSVD(n_components=shape, n_iter=100)
    lsa.fit(matrix)
    terms = vectorizer.get_feature_names()
    for i,comp in enumerate(lsa.components_):
        termList = []
        termsInComp = zip(terms,comp)
        sortedterms = sorted(termsInComp, key=lambda x: x[1],reverse=True)[:10]
        mykey = "Topic :" + str(i)
        for term in sortedterms:
            termList.append(term[0])
        lsa_dict[mykey] = termList
    return lsa_dict

def final_dataset_function():
    decompostion_dict,lsa,lsa_dict = ({} for i in range(3))
    offset = 1000
    coors = {'low_left_x' : 421710.112401581, 'low_left_y' : 4610737.961457818, 'up_right_x' : 456608.39121255605, 'up_right_y' : 4652466.087380382}
    xGrid = int((coors['up_right_x'] - coors['low_left_x']) / offset) + 1
    df = initialise_dataframe()
    df,doc_sentiment,myDict2, textData = process_dataframe(df,offset,coors,xGrid)
    my_vectorizer = TfidfVectorizer(use_idf=True,ngram_range=(1,3))
    for keys,values in myDict2.items():
        shape,matrix,my_vectorizer = vectorizer(values,my_vectorizer)
        myDict = decomposition(shape,matrix,my_vectorizer,keys,lsa_dict)
        lsa[keys] = myDict
    return df,doc_sentiment,lsa