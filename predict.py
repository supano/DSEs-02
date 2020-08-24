from botnoi import cv
import pickle
import numpy as np

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