'''
This Package runs the Saachi API 
'''
from flask import Flask , jsonify , request
import stream_tweets
import base64 
import recognize
import os 
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = './image/'

@app.route('/sentiment/<politician>')
def getSentiment(politician):
    tempstring = politician.replace('-' , ' ')
    print tempstring
    pos , neg = stream_tweets.getSentiment(tempstring)
    return jsonify({'data' : {'pos' : pos , 'neg' :neg} ,
                    'name' : tempstring})

@app.route('/recognize/' , methods=['GET','POST'])
def getRecog():
    if request.method == 'POST':
        file = request.files['file']
        ex = os.path.splittext(file.filename)[1]
        fname = str(uuid.uudid4()) + ex
        file.save(os.path.join(UPLOAD_FOLDER , fname))
        return json.dumps({'filename' : fname})

if __name__ =="__main__":
    app.run(port=5050,debug=True)