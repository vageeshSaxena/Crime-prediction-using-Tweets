#!/usr/bin/python3

import os
import glob
import time
import math

import pandas as pd

from tqdm import tqdm

TWEETS_DIR_JSON_PATH = '/Users/shlomi/Dropbox/Apps/file-saver'
TWEETS_JSON_PATHS = glob.glob(os.path.join(TWEETS_DIR_JSON_PATH, '*-*-*.json'))
TWEETS_JSON_PATHS.sort()

N_JSONS_IN_CSV = 100

N_JSONS_BLOCKS = int(math.ceil(len(TWEETS_JSON_PATHS)/N_JSONS_IN_CSV))

def main():
    timestamp = int(time.time())
    
    
    for i in tqdm(range(N_JSONS_BLOCKS)):
        tweets_jsons_paths_block = TWEETS_JSON_PATHS[i*N_JSONS_IN_CSV:(i+1)*N_JSONS_IN_CSV]
        
        unified_csv_path = os.path.join(
                    TWEETS_DIR_JSON_PATH, 'tweets-{}-{}.csv'.format(timestamp, i))

        with open(unified_csv_path, 'w') as unified_file:
            unified_df = pd.concat(
                (pd.read_json(path) for path in tqdm(tweets_jsons_paths_block))
            )
            unified_df.to_csv(unified_file)

            """
            header = True
            for tweets_json_path in tqdm(tweets_jsons_paths_block):
                single_tweets_df = pd.read_json(tweets_json_path)
                single_tweets_df.to_csv(unified_file, mode='a', header=header)
                header = False
           """
    
if __name__ == '__main__':
    main()