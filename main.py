'''
This Package runs the Saachi API 
'''
from flask import Flask , jsonify
import stream_tweets

app = Flask(__name__)

@app.route('/sentiment/<politician>')
def getSentiment(politician):
	tempstring = politician.replace('-' , ' ')
	print tempstring
	pos , neg = stream_tweets.getSentiment(tempstring)
	return jsonify({'data' : {'pos' : pos , 'neg' :neg} ,
					'name' : tempstring})

def recognize()
if __name__ =="__main__":
	app.run(port=5050,debug=True)