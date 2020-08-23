import flask
import pickle
import numpy as np
from flask import request as fl_requests, jsonify
from botnoi import cv

app = flask.Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "<h1>DES G6</p>"


modFile = './models/mymodel.p'
mymod = pickle.load(open(modFile,'rb'))
def predictimg(imgurl):
  a = cv.image(imgurl)
  feat = a.getresnet50()
  probList = mymod.predict_proba([feat])[0]
  maxprobind = np.argmax(probList)
  prob = probList[maxprobind]
  outclass = mymod.classes_[maxprobind]
  result = {}
  result['class'] = outclass
  result['probability'] = prob
  return result

@app.route('/api', methods=['GET'])
def api():
    if 'url' in fl_requests.args:
        url = fl_requests.args['url']
    else:
        return "Error: No url field provided. Please specify an url."

    return predictimg(url)

app.run()