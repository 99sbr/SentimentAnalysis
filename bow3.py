import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer


train = pd.read_csv("/home/subir_sbr/Desktop/labeledTrainData.tsv",delimiter="\t", quoting=3,nrows=5000)

pred = pd.read_csv("/home/subir_sbr/Desktop/pred_sentiment.csv",delimiter=",", quoting=3,nrows=5000,names=['index','sentiment'])

train_df=pd.DataFrame(data=train['sentiment'])
pred_df=pd.DataFrame(data=pred['sentiment'])

print(train_df.head(5))
print(pred_df.head(5))


from sklearn.metrics import confusion_matrix
CF=(confusion_matrix(train_df,pred_df))

TN=CF[0,0]
FP=CF[0,1]
FN=CF[1,0]
TP=CF[1,1]
total=TP+TN+FP+FN
Accuracy= ((TP+TN)/total)
print("Model accuracy is %f \n" %Accuracy)
Miscalssification_rate=(FP+FN)/total
print("Model Error rate is %f \n" % Miscalssification_rate)
Precision=(TP/(TP+FP))
Recall=(TP/(TP+FN))
F_score=(2*Precision*Recall)/(Precision+Recall)
print("Model Precision is %f \n"% Precision)
print("Model Recall is %f \n" % Recall)
print("Model F score is %f\n " % F_score)
'''
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
tls.set_credentials_file(username='99sbr', api_key='nmalq8q5ov')

data = [go.Bar(
            x=['Accuracy', 'Error Rate', 'Precision','Recall','F_score'],
            y=[Accuracy, Miscalssification_rate,Precision,Recall,F_score]
    )]

py.iplot(data, filename='basic-bar')'''
from matplotlib import pyplot as plt
import numpy as np
l=[Accuracy,Miscalssification_rate,Precision,Recall,F_score]
x=['Accuracy', 'Error Rate', 'Precision','Recall','F_score']
width = .35
ind = np.arange(len(l))
plt.bar(ind,l, width=width)
plt.xticks(ind + width / 2, x)
plt.show()