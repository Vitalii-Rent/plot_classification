import string
import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np


def clean_text(text):
    text = text.lower()
    html_pattern = re.compile('<.*?>')
    text = html_pattern.sub(r' ', text)
    url_pattern = re.compile(r'http?://\S+|www\.\S+')
    text = url_pattern.sub(r' ', text)
    dots_pattern = re.compile(r'\.+')
    text = dots_pattern.sub(r' ', text)
    text = text.translate(text.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    text = re.sub(r'\d+', ' ', text)
    text = " ".join(text.split())
    return text


def filter_rare(data):
    value_counts = data['Genre'].value_counts()
    mask = data['Genre'].isin(value_counts[value_counts >= 10].index)

    # Apply the mask to filter out the rows
    df_filtered = data[mask].copy()
    return df_filtered


def lemming(text):
    lemmatizer = WordNetLemmatizer()
    result = []
    for word in text:
        result.append(lemmatizer.lemmatize(word))
    return result


def remove_stopwords(text, en_stopwords):
    result = []
    for token in text:
        if token not in en_stopwords:
            result.append(token)
    return result


def replace_labels(labels, replacements):
    for old, new in replacements:
        labels = labels.replace(old, new)
    return labels


def partial_clean_text(text):
    html_pattern = re.compile('<.*?>')
    text = html_pattern.sub(r' ', text)
    url_pattern = re.compile(r'http?://\S+|www\.\S+')
    text = url_pattern.sub(r' ', text)
    text = " ".join(text.split())
    return text

def process_script(df):
    en_stopwords = stopwords.words('english')
    df = df.apply(clean_text)
    df = df.apply(lambda X: word_tokenize(X))
    df=df.apply(lambda x: remove_stopwords(x, en_stopwords))
    df=df.apply(lemming)
    return df

def dummy(x):
    return x

replacements = [
    ('animated', 'animation'),
    ('biography', 'biographical'),
    ('biopic', 'biographical'),
    ('com', 'com'),
    ('com', 'comedy'),
    ('docudrama', 'documentary drama'),
    ('dramedy', 'drama comedy'),
    ('sci fi', 'sci_fi'),
    ('science fiction', 'sci_fi'),
    ('film', ''),
    ('world war ii', 'world_ii war'),
    ('rom ', 'romantic '),
    ('romance', 'romantic'),
    ('comedyedy', 'comedy')
]

best_classes = ['spy', 'short', 'fantasy', 'mystery', 'war', 'animation', 'adventure', 'sci_fi', 'musical', 'western', 'horror', 'crime', 'thriller', 'action', 'romantic', 'comedy', 'drama']