import re
import string

import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


# spacy.cli.download("en_core_web_sm")


class SentimentAnalysis:
    def __init__(self, file_path: str = "data/sentiment.csv"):
        self.data = pd.read_csv(file_path, on_bad_lines="skip", engine="python")
        self.stop_words: set = set(stopwords.words("english"))
        self.nlp = spacy.load("en_core_web_sm")

    def pre_process(self):
        self.make_all_tweets_lowercase()
        self.data["text"] = self.data["text"].apply(self.remove_url_mentions_hashtags)
        self.data["text"] = self.data["text"].apply(self.remove_punctuation)
        self.data["text"] = self.data["text"].apply(self.remove_emojis)
        self.data["text"] = self.data["text"].apply(self.tokenize)
        # self.data["text"] = self.data["text"].apply(self.handle_negation)
        self.data["text"] = self.data["text"].apply(self.remove_stopwords)
        self.data["text"] = self.data["text"].apply(self.lemmatize_text)

        print(self.data.head(10))

    def make_all_tweets_lowercase(self):
        """
        Make all text lowercase in the 'text' column
        :return: list of all text lowercase
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
        if not isinstance(text, str):
            return text

        translator = str.maketrans("", "", string.punctuation)
        return text.translate(translator)

    def remove_emojis(self, text):
        """Removes emojis from a string.

        Args:
            text: The input string.

        Returns:
            The string with emojis removed.
        """
        if not isinstance(text, str):
            return text

        emoji_pattern = re.compile("[\U00010000-\U0010FFFF]", flags=re.UNICODE)
        return emoji_pattern.sub("", text)

    def tokenize(self, text):
        """
        Tokenize a string.
        :param text: The input string.
        :return: tokenized string.
        """
        if not isinstance(text, str):
            return text

        tokens = word_tokenize(text)
        return " ".join(tokens)

    def handle_negation(self, text):
        if not isinstance(text, str):
            return text

        doc = self.nlp(text)
        tokens = []

        for token in doc:
            if token.dep_ == "neg":
                tokens.append("NEG_" + token.head.text)
            else:
                tokens.append(token.text)

        return " ".join(tokens)

    def remove_stopwords(self, text):
        """
        Remove stopwords from a string.
        :param text: The input string.
        :return: filtered string.
        """
        if not isinstance(text, str):
            return text

        filtered_words = [
            word for word in text.split() if word.lower() not in self.stop_words
        ]
        return " ".join(filtered_words)

    def lemmatize_text(self, text):
        """
        Lemmatize the input text using spaCy.

        Parameters:
            text (str): The text to lemmatize.

        Returns:
            str: The lemmatized text.
        """
        if not isinstance(text, str):
            return text

        doc = self.nlp(text)
        lemmatized_text = " ".join([token.lemma_ for token in doc])
        return lemmatized_text


SentimentAnalysis(file_path="data/sentiment.csv").pre_process()
