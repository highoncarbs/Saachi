'''
This Package runs the Saachi API 
'''
from flask import Flask , jsonify
import stream_tweets
import base64 
import recognize
import os 

app = Flask(__name__)

@app.route('/sentiment/<politician>')
def getSentiment(politician):
    tempstring = politician.replace('-' , ' ')
    print tempstring
    pos , neg = stream_tweets.getSentiment(tempstring)
    return jsonify({'data' : {'pos' : pos , 'neg' :neg} ,
                    'name' : tempstring})

@app.route('/recognize/<base64_data>')
def getRecog(base64_data):
    img_data = base64_data
    with open("./image_test.jpeg", "wb") as fh:
        fh.write(base64.decodebytes(img_data))
    path = os.path.abspath("./image_test.jpeg")
    name = recognize.predict_face(path)
    print name 
    return jsonify({'name' : name})    

if __name__ =="__main__":
    app.run(port=5050,debug=True)