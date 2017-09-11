# coding -utf-8-
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener 
import tweepy
import sys 
import indicoio 

# Keys

con_key = "ucleZiPt36pzv8JvTxmFbC5Bj"
con_secret = "DH0tbHl57Bo10mYg5MbShw8LuDJjA9iW5NzPP0EbYgv6W0CFDA"
acc_token = "276857219-Mj5kFn4ZVUr8tG9Y5M0Q7BZ9yMnejJEBbLDgVDtp"
acc_secret = "Ba2vtsKJzrYOee6ncXcQE9uXuCVRYJ4Zhngxw3DMu7R69"
indicoio.config.api_key = 'ef0a7214e6de676cd0c531587165ba35'


auth = OAuthHandler(con_key , con_secret)
auth.set_access_token(acc_token , acc_secret)
api = tweepy.API(auth , wait_on_rate_limit = True,
			wait_on_rate_limit_notify=True)

if (not api):
	print ("Authentication Error : Unable to authenticate")
	sys.exit(-1)

def getSentiment(politic):
	search_results = api.search(q=politic , count=100)

	result_tweets = []

	for result in search_results:
		result = result.text.encode('ascii' ,errors='ignore')
		result_tweets.append(result)

	sentiment = indicoio.sentiment(result_tweets)
	pos = int(100*(sum(sentiment)/len(sentiment)))
	neg = int(100-pos)
	return pos,neg