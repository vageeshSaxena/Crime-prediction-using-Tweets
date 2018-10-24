import logging
import queue
import threading
import time
import argparse

import tweepy
import dropbox

TWITTER_CONSUMER_KEY = ''  # to fill
TWITTER_CONSUMER_SECRET = ''  # to fill
TWITTER_ACCESS_TOKEN = ''  # to fill
TWITTER_ACCESS_SECRET = ''  # to fill

DROPBOX_ACCESS_TOKEN = ''  # to fill

DEFAULT_SAVE_SIZE = 4 * 1024 ** 2  # 4 MB

LIMIT_SLEEP = 10

LOCATION_CHICAGO = [-87.94011, 41.64454, -87.52413, 42.02303]

LOGGING_FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)


class JSONDropboxSaver(threading.Thread):

    def __init__(self, queue, access_token, save_size):
        super().__init__()
        self.queue = queue

        self.access_token = access_token
        self.dbx = dropbox.Dropbox(self.access_token)

        self.save_size = save_size

    def run(self):
        data = []
        accumulated_data_length = 0
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    logging.info('Got None in Queue, saving and ending thread...')
                    if data:
                        self._save_json(data)
                    break
                else:
                    data.append(item)
                    accumulated_data_length += len(item)
                    logging.info('Accumulated Data Length: {}'.format(accumulated_data_length))
                if accumulated_data_length > self.save_size:
                    self._save_json(data)
                    data = []
                    accumulated_data_length = 0

        except Exception:
            logging.exception('Exception {}!'.format(e))

    def _save_json(self, data):
        json_data = '[' + ','.join(data) + ']'
        path = '/' + time.strftime('%Y-%m-%d %H:%M:%S') + '.json'
        logging.info('Saving to file <{}>. {} bytes...'.format(path, len(json_data)))
        self.dbx.files_upload(json_data.encode('utf8'), path)


class TweetsQueueListner(tweepy.streaming.StreamListener):
    def __init__(self, queue):
        self.queue = queue

    def on_connect(self):
        logging.info('StreamListner connected succefully to Twitter!')

    def on_data(self, data):
        logging.info('Data arrived! len={}'.format(len(data)))
        if data:
            self.queue.put(data)

    def on_error(self, status_code):
        logging.error('Stream Error! status code: {}'.format(status_code))
        if status_code != 420:
            return False

    # TODO - better sleep mechanisem
    def on_limit(self, track):
        logging.warning('Limit reached! <{}>'.format(str(track)))
        logging.warning('Sleeping for {} seconds...'.LIMIT_SLEEP)
        time.sleep(LIMIT_SLEEP)
        logging.warning('Woke up!')


def main(save_size):
    logging.info('Authonticating...')
    auth = tweepy.OAuthHandler(TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET)
    auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
    logging.info('Authontication succeed!')

    tweets_queue = queue.Queue()

    tweets_listener = TweetsQueueListner(tweets_queue)

    dropbox_saver = JSONDropboxSaver(tweets_queue, DROPBOX_ACCESS_TOKEN, save_size)
    logging.info('Starting Dropbox Save thread...')
    dropbox_saver.start()

    tweets_stream = tweepy.Stream(auth, tweets_listener)

    logging.info('Filtering Tweets by locaion in {}...'.format(LOCATION_CHICAGO))
    tweets_stream.filter(locations=LOCATION_CHICAGO, async=False)

    try:
        pass  # dropbox_saver.join()

    # BaseException for catching also KeyboardInterrupt
    except BaseException as e:
        logging.exception('Exception {}!'.format(e))

        logging.info('Disconnect Tweets Stream...')
        tweets_stream.disconnect()

        logging.info('Put None in Queue...')
        tweets_queue.put(None)

        logging.info('Wait Dropbox Saver to finish...')
        dropbox_saver.join()

    finally:
        logging.info('Bye Bye...')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Tweets saver to Dropbox.')

    parser.add_argument('-s', '--save-size', type=int, default=DEFAULT_SAVE_SIZE,
                        help='save size in bytes')

    parser.add_argument('-l', '--log', type=str,
                        help='log file path')

    args = parser.parse_args()

    if args.log:
        log_file = logging.FileHandler(args.log)
        log_file.setLevel(logging.DEBUG)
        formatter = logging.Formatter(LOGGING_FORMAT)
        log_file.setFormatter(formatter)
        logging.getLogger('').addHandler(log_file)

    main(args.save_size)
