

'''
This Package runs the Saachi API
'''
from flask import Flask , jsonify , request
import stream_tweets
import base64
import recognize
import os
import uuid
from werkzeug.datastructures import ImmutableMultiDict

from elasticsearch import Elasticsearch
from json import dumps
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = './image/'

ES = Elasticsearch()
@app.route('/sentiment/<politician>')
def getSentiment(politician):
    tempstring = politician.replace('-' , ' ')
    print tempstring
    pos , neg = stream_tweets.getSentiment(tempstring)
    return jsonify({'data' : {'pos' : pos , 'neg' :neg} ,
                    'name' : tempstring})

@app.route('/recognize' , methods=['GET','POST'])
def getRecog():
    if request.method == "POST":
        file = request.files['file']
        # ex = request.form['file']
        print file
        fname = "recog_img.jpg"
        file_path = os.path.join(UPLOAD_FOLDER , fname)
        file.save(file_path)
        face = recognize.predict_face(file_path)
        # print face
        return jsonify({'name': face})
    else:
        return jsonify({'name': "nannana test"})

@app.route('/search/<politician>')
def findPolitician(politician):
    size = request.args['size']
    search_query = {
        'size': size,
        'query': {
            'dis_max': {
                'queries': [
                    {
                        'match': {
                            'name': {
                                'query': politician,
                                'boost': 2
                            }
                        }
                    },
                    {
                        'match': {
                            'S/o|D/o|W/o': {
                                'query': politician,
                                'boost': 1
                            }
                        }
                    }
                ]
            }
        },
        'sort': [
            {'_score': {'order': 'desc'}}
        ]
    }

    results = ES.search(index='hackocracy', doc_type='document',
                        body=dumps(search_query), sort='_score')
    return jsonify(results)

if __name__ =="__main__":
    app.run(port=5000,debug=False)
