#!/usr/bin/env python3

from gensim.models import Word2Vec
from datetime import datetime, timedelta
import boto3
from io import BytesIO
import io
import re
import emoji
import pandas as pd
from tqdm import tqdm
import json
import pickle

# Set todays date
today = datetime.now()
yesterday = today - timedelta(days=1)

today = today.strftime('%Y-%m-%d')
yesterday = yesterday.strftime('%Y-%m-%d')

# Get Terms
termlist = s3.Object('tweet-vectors', 'terms/termslist.txt').get()['Body'].read().decode('utf-8')
termlist = termlist.split('\n')

# Get Trends and add terms to list
obj = s3.Object("socialtrendingtweets-v1", 'trends/trends_{}.json'.format(today))
file_content = obj.get()['Body'].read().decode('utf-8')
json_content = json.loads(file_content)
names = [a['name'].lower() for a in json_content['trends'] if len(a['name'].split()) > 1]
termlist += names
s3.Object('tweet-vectors', 'terms/termslist.txt').put(Body="\n".join(termlist))
    

def clean_tweet(x):    
    out = re.sub(r"(https?.+\s|[\w\.]+\.(com|co\.uk|net|co))"," ", x)   
    out = re.sub(r"(\.|\?|\!|\…|\,)" ,r" \1 ", out)
    for name in termlist:
        out = re.sub(r'{}'.format(name), name.replace(' ', '_'), out.lower())
    return re.sub(emoji.get_emoji_regexp(), r' \1 ', out.lower()).split()

# Set Session
session = boto3.Session()
s3 = session.resource('s3')
my_bucket = s3.Bucket("socialtrendingtweets-v1")

# Get Todays Tweets
tweet_files = [object_summary.key for object_summary in my_bucket.objects.filter(Prefix="tweets/{}".format(today))]
results = []
for x in tqdm(tweet_files):
    obj = s3.Object("socialtrendingtweets-v1", x)
    file_content = obj.get()['Body'].read().decode('utf-8')
    json_content = json.loads(file_content)
    for tweet in json_content['tweets']:
        tweet['trend_name'] = json_content['name']
    results.extend(json_content['tweets'])

all_tweets = []
for tweet in results:
    all_tweets.append(clean_tweet(tweet['full_text']))

# Load Yesterdays Word2Vec Model
with BytesIO() as data:
    s3.Bucket("tweet-vectors").download_fileobj("models/{}.model".format(yesterday), data)
    data.seek(0)    # move back to the beginning after writing
    model = pickle.load(data)

    # Update model
model.build_vocab(all_tweets, update=True)  # Update the vocabulary
model.train(all_tweets, total_examples=len(all_tweets), epochs=model.epochs)

s3.Object("tweet-vectors", "models/{}.model".format(today)).put(Body=pickle.dumps(model))

