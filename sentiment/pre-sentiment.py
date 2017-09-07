#! /usr/bin/python2.7

'''
	This package takes care of training of
	our model on tweets data.
'''
import nltk
import numpy as np 
import matplotlib as plt 
import time
from copy import deepcopy
from collections import Counter
import pandas as pd 
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidVectorizer

from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer()


import gensim
from gensim.models.word2vec import Word2Vec
LabeledSentence = gensim.models.doc2vec.LabeledSentence


data_base = "/home/padam/Documents/data/sentiment-data/"

def ingest_tweets():
	data = pd.read_csv(data_base+'train_tweet.csv')
	data.drop(['ItemID' , 'SentimentSource' ,'Date' , 'Blank'] , axis =1 , inplace = True)
	# data = data[datab.Sentiment.isnull == False]
	data['Sentiment'] = data['Sentiment'].map(int)
	data = data[data['SentimentText'].isnull() == False]
	data.reset_index(inplace=True)
	data.drop('index' , axis=1 , inplace=True)
	return data 

data = ingest_tweets()
# print data.head(5)

def tokenize_tweet(tweet):
	'''
		Tokenizes the tweets and removes 
		unwanted features like '#' , '@' , urls

		INPUT : tweet - string
		OUTPUT : tokenized tweet 
	'''
	try:
		tweet = unicode(tweet.decode('utf-8').lower())
		tokens = tokenizer.tokenize(tweet)
		tokens = filter(lambda t : not t.startswith('@') , tokens)
		tokens = filter(lambda t : not t.startswith('#') , tokens)
		tokens = filter(lambda t : not t.startswith('http') , tokens)
		return tokens 
	except:
		return 'NE' 

def post_process(data , num = 1200000):
	'''
		Apply tokenize function to each SentimentText
		row.

		* Deepcopy fixes the issue of SettingWithCopyWarning
		* in pandas
	'''
	data = deepcopy(data.head(num))
	data['tokens'] = data['SentimentText'].map(tokenize_tweet) 
	data = data[data.tokens != 'NE']
	data.reset_index(inplace =True)
	data.drop('index' , axis = 1 , inplace=True)
	return data 

x = post_process(data)
# print x.head(5)

# Word2Vec model
n_num = 1200000
X_train , X_test , Y_train , Y_test = train_test_split(np.array(x.head(n_num).tokens) , 
														np.array(x.head(n_num).Sentiment),
														test_size = 0.2)

def label_tweet(tweet , label):
	labeled = []
	for i,v in tqdm(enumerate(tweet)):
		l = '%s_%s'%(label,i)
		labeled.append((LabeledSentence(v , [l])))
	return labeled

X_train = label_tweet(X_train , 'TRAIN')
X_test = label_tweet(X_test,'TEST')

w2v = Word2Vec(size=200 , min_count=10)
w2v.build_vocab([x.words for x in tqdm(X_train)])
w2v.train([x.words for x in tqdm(X_train)] , total_examples = w2v.corpus_count , epochs=w2v.iter)

# print w2v.most_similar('good')

# Works well !

w2v.wv.save_word2vec_format('./word2vec_model.bin' , binary=True)

# Loading of model
# >>> model = gensim.models.KeyedVectors.load_word2vec_format('./word2vec_model.bin', binary=True , unicode_errors='ignore')
# Make sure using unicode_errors='ignore' or 'replace' or else
# use coding -utf-8- shebang (not sure)