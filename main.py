import re
import string
import pandas as pd


class SentimentAnalysis:
    def __init__(self, file_path: str = "data/sentiment.csv"):
        self.data = pd.read_csv(file_path, on_bad_lines="skip", engine="python")
        print(self.data.head(10))

    def pre_process(self):
        self.make_all_tweets_lowercase()
        self.data["text"] = self.data["text"].apply(self.remove_url_mentions_hashtags)
        self.data["text"] = self.data["text"].apply(self.remove_punctuation)

        print(self.data.head(10))

    def make_all_tweets_lowercase(self):
        """
        Make all text lowercase in the 'text' column
        :return:
        """
        self.data["text"] = self.data["text"].str.lower()

    def remove_url_mentions_hashtags(self, text):
        """
        Remove URLs, mentions and hashtags
        :param text:
        :return:
        """
        if isinstance(text, str):
            text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
            text = re.sub(r"@\w+", "", text)
            text = re.sub(r"#\w+", "", text)
        return text

    def remove_punctuation(self, text):
        """
        Remove punctuation marks and special characters from a string.
        :param text:
        :return:
        """
        if isinstance(text, str):
            # Create a translation table that maps each punctuation character to None
            translator = str.maketrans("", "", string.punctuation)
            # Use translate method to remove punctuation
            return text.translate(translator)
        return text


SentimentAnalysis().pre_process()
