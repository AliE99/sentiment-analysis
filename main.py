import re

import pandas as pd


class SentimentAnalysis:
    def __init__(self, file_path: str = 'data/sentiment.csv'):
        self.data = pd.read_csv(file_path, on_bad_lines='skip', engine='python')

    def pre_process(self):
        self.make_all_tweets_lowercase()
        self.data['text'] = self.data['text'].apply(self.clean_tweet)
        print(self.data.head(10))

    def make_all_tweets_lowercase(self):
        """
        Make all text lowercase in the 'text' column
        :return:
        """
        self.data['text'] = self.data['text'].str.lower()

    def clean_tweet(self, text):
        """
        Only process if text is a string
        Remove URLs, mentions and hashtags
        :param text:
        :return:
        """
        if isinstance(text, str):
            text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
            text = re.sub(r"@\w+", '', text)
            text = re.sub(r"#\w+", '', text)
        return text


SentimentAnalysis().pre_process()
