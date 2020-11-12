import tweepy
import nltk
import sys
sys.path.append('/Users/apple/bhavik/COS80023/Project/Emotion detector/')
from collections import Counter
import re
import string
from nltk.corpus import stopwords
import vincent
from collections import defaultdict
from textblob import TextBlob
import text2emotion as te
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import model
from . import twitter_credentials
import pandas as pd

import sys
from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS



try:
    import json
except ImportError:
    import simplejson as json

# get path to script's directory
currdir = path.dirname(__file__)

auth = tweepy.OAuthHandler(twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_SECRET)
auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)


emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)


def tokenize(s):
    return tokens_re.findall(s)


def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def tweet_prediction(text):
    # tokenize the string and convert into matrix
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)

    X = tokenizer.texts_to_sequences(text)
    X = pad_sequences(X)

    type_result = model.predict(X)
    return type_result


# Function to convert
def listToString(s):
    # initialize an empty string
    str1 = " "

    # return string
    return (str1.join(s))




punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['RT', 'via']


common_word_count = Counter()
emotion_count = Counter()

com = defaultdict(lambda : defaultdict(int))

positive = 0
negative = 0
neutral = 0
polarity = 0

labels_dict={0:'Sadness',1:'Worry',2:'Joy',3:'Surprise',4:'Happiness',5:'Love'}



def data(search):
    tweets = []
    common_word = []
    sentiment = []
    Tweet_analysis = []
    tweet_emotion = []

    cp = 0
    cn = 0
    cneg = 0

    for search in tweepy.Cursor(api.search, q=search, lang='en').items(10):

        terms_only = [term for term in preprocess(search._json['text']) if term not in stop and not term.startswith(('#','@','"','…','’'))]

        tweets.append(search._json['text'])

        clean_tweet = listToString(terms_only)


        predict = tweet_prediction(clean_tweet)

        label = np.argmax(predict, axis=1)[0]
        tweet_emotion.append(labels_dict[label])

        analysis = TextBlob(clean_tweet)

        if analysis.sentiment[0] > 0:
            cp += 1
            sentiment.append("Positive")
        elif analysis.sentiment[0] < 0:
            cneg += 1
            sentiment.append("Negative")
        else:
            cn += 1
            sentiment.append("Neutral")

        common_word_count.update(terms_only)
        word_freq = common_word_count.most_common(10)
        labels, freq = zip(*word_freq)
        data = {'data': freq, 'x': labels}

        bar = vincent.Bar(data, iter_idx='x')
        bar.to_json('search.json')

        Tweet_analysis = list(zip(tweets,sentiment,tweet_emotion))
        print()

    common_word.append(common_word_count.most_common(10))
    emotion_count.update(tweet_emotion)
    emotion_freq = emotion_count.most_common(10)

    labels, freq = zip(*emotion_freq)
    data1 = {'data': freq, 'x': labels}

    bar = vincent.Pie(data1, iter_idx='x')
    bar.legend('Emotion')
    bar.to_json('emotion.json')

    # create numpy araay for wordcloud mask image
    mask = np.array(Image.open(path.join(currdir, "cloud.png")))

    # create set of stopwords
    stopwords = set(STOPWORDS)

    # create wordcloud object
    wc = WordCloud(background_color="black",
                   max_words=200,
                   mask=mask,
                   stopwords=stopwords)


    # Create the Pandas dataFrame.
    wc_tweet = pd.DataFrame(tweets, columns=['tweet'])

    wc_tweet['tweet'] = wc_tweet['tweet'].str.replace('[^A-Za-z0-9\s]+', '')
    wc_tweet['tweet'] = wc_tweet['tweet'].str.replace('http\S+|www.\S+', '', case=False)
    wc_tweet['tweet'] = wc_tweet['tweet'].str.replace('RT', '')


    wc.generate(str(wc_tweet['tweet']))

    # save wordcloud
    wc.to_file(path.join('/Users/apple/bhavik/COS80023/Project/twitter_sentiment_analysis-2 copy 5/static/image/', "wc.png"))

    common_word_count.clear()
    emotion_count.clear()

    return tweets,common_word,sentiment,Tweet_analysis,cp,cneg,cn

