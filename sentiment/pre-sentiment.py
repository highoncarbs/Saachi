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
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords 
import matplotlib.pyplot as plt 

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
	data['Sentiment'] = data['Sentiment'].map( {4:1, 0:0})
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

'''
 * Use this only to train a new model , word2vec_model
 	is already trained on tweet data.

w2v = Word2Vec(size=200 , min_count=10)
w2v.build_vocab([x.words for x in tqdm(X_train)])
w2v.train([x.words for x in tqdm(X_train)] , total_examples = w2v.corpus_count , epochs=w2v.iter)

# print w2v.most_similar('good')

# Works well !

w2v.wv.save_word2vec_format('./word2vec_model.bin' , binary=True)

# Loading of model
 	>>> model = gensim.models.KeyedVectors.load_word2vec_format('./word2vec_model.bin', binary=True , unicode_errors='ignore')
 	Make sure using unicode_errors='ignore' or 'replace' or else
 	use coding -utf-8- shebang (not sure)

'''
w2v_load = gensim.models.KeyedVectors.load_word2vec_format('/home/padam/Documents/git/Saachi/sentiment/word2vec_model.bin', binary=True , unicode_errors='ignore')

print ("Building tf-idf matrix ... ... ")
vectorizer = TfidfVectorizer(analyzer = lambda x :x , min_df = 10)
matrix = vectorizer.fit_transform([x.words for x in X_train])
tfidf = dict(zip(vectorizer.get_feature_names() , vectorizer.idf_))
print 'vocab size : ',len(tfidf)

def word_vector(tokens , size = 200):
	vec = np.zeros(size).reshape(1 , size)
	count = 0
	for word in tokens:
		try: 
			vec += w2v_load[word].reshape(1,size)*tfidf[word] 
			count += 1 
		except KeyError: # Handle when token not id corpus
			continue
	if count != 0:
		vec /= count 
	return vec

# Normalizing the data

from sklearn.preprocessing import scale 

size = 200
train_vecs_w2v = np.concatenate([word_vector(z ,size) for z in tqdm(map(lambda x : x.words , X_train))])
train_vecs_w2v = scale(train_vecs_w2v)

test_vecs_w2v = np.concatenate([word_vector(z,size) for z in tqdm(map(lambda x:x.words , X_test))])
test_vecs_w2v = scale(test_vecs_w2v)

# Classifier using Feed Forward Neual Network

import torch
import torch.nn as nn 
import torch.autograd.Variable as Variable 
import torch.nn.functional as F 

learning_rate = 0.001
num_epochs = 60
D_in , H , D_out = 200 , 32 ,1

# Model

'''

class net(nn.Module):
	def __init__(self):
		super(net , self).__init__()
		self.l1 = nn.Linear(200, 32)
		self.relu = F.ReLU()
		self.l2 = nn.Linear(32 , 1)
		self.sig = F.sigmoid()

	def forward(self , x):
		x = self.relu(self.l1(x))
		x = self.l2(x)
		x = self.sig(x)
		return x 

# neural_net = net()
'''

# Sequential Model Approach 

net = torch.nn.Sequential(
		torch.nn.Linear(D_in , H),
		torch.nn.ReLU(),
		torch.nn.Linear(H , D_out),
		torch.nn.Sigmoid()
	)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(net.parameters() ,lr = learning_rate)
inputs = Variable(torch.from_numpy(train_vecs_w2v))
targets = Variable(torch.from_numpy(Y_train))

for epoch in range(num_epochs):
	
	optimizer.zero_grad()
	outputs = net(inputs)
	loss = criterion(outputs , targets)
	loss.backward()
	optimizer.step()

	if(epoch+1)%5 ==0:
		print('Epoch [%d%d] , Loss : %.4f'%(epoch+1 , num_epochs , loss.data[0]))

# Plotting graph

# predicted = neural_net(Variable(torch.from_numpy(X_train))).data.numpy()
# plt.plot(X_train, Y_train, 'ro', label='Original data')
# plt.plot(X_train, predicted, label='Fitted line')
# plt.legend()
# plt.show()