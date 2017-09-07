# coding -utf-8-
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener 
import tweepy
import sys 

# Keys

con_key = "ucleZiPt36pzv8JvTxmFbC5Bj"
con_secret = "DH0tbHl57Bo10mYg5MbShw8LuDJjA9iW5NzPP0EbYgv6W0CFDA"
acc_token = "276857219-Mj5kFn4ZVUr8tG9Y5M0Q7BZ9yMnejJEBbLDgVDtp"
acc_secret = "Ba2vtsKJzrYOee6ncXcQE9uXuCVRYJ4Zhngxw3DMu7R69"

class listener(StreamListener):

	def on_data(self , data):
		print(data)
		return  True

	def on_error(self , status):
		print(status)

auth = OAuthHandler(con_key , con_secret)
auth.set_access_token(acc_token , acc_secret)
api = tweepy.API(auth , wait_on_rate_limit = True,
			wait_on_rate_limit_notify=True)

if (not api):
	print ("Authentication Error : Unable to authenticate")
	sys.exit(-1)

search_results = api.search(q="Mulayam Singh Yadav" , count=100)

result_tweets = []

for result in search_results:
	result_tweets.append(result.text)

