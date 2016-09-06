import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer


train1 = pd.read_csv("/home/subir_sbr/Desktop/sentiment_lexicon.csv",names=['words','neg','pos'])
df=pd.DataFrame(data=train1[1:])


sentiment=pd.DataFrame(columns=['sentiment'])
df=pd.concat([df,sentiment],axis=1)
df.loc[df['pos']>=df['neg'],'sentiment']=1 
df.loc[df['pos']<df['neg'],'sentiment']=0 
del df['pos']
del df['neg']
pos_bag=[]
num=df['sentiment'].size
for i in range(1,num):
	if((i+1)%5000==0):
		print("positive Review %d of %d\n" % ( i+1,num) ) 
	if df.loc[i].sentiment==1:
		pos_bag.append(df.loc[i].words)
#from sets import Set
neg_bag=[]
for i in range(1,num):
	if((i+1)%5000==0):
		print("Negative review %d of %d \n" % (i+1,num))
	if df.loc[i].sentiment==0:
		neg_bag.append(df.loc[i].words)


train2 = pd.read_csv("/home/subir_sbr/Desktop/labeledTrainData.tsv",delimiter="\t", quoting=3,nrows=5000)


def review_to_words(raw_review):
        ex=raw_review
        # now we import BeautifulSoup to remove the tags and markups from the text data
       
        soup=BeautifulSoup(ex,"lxml")
        clean_text=soup.get_text()
        # Dealing with Punctuations
        
        # Use regular expressions to do a find-and-replace
        letters_only=re.sub("[^a-zA-Z]", " ", clean_text)
        # tokenizing
        lower_case=letters_only.lower()
        words=lower_case.split()
        # remove stopwords
        
        stops=set(stopwords.words('english'))
        words= [w for w in words if not w in stops]
        #Stemming
        port=PorterStemmer()
            
        stem_word=[]
        for word in words:
            stem_word.append(port.stem(word))
            new_set=" ".join( stem_word )
        return stem_word
    
num_reviews = train2["review"].size
print("Cleaning and parsing the training set movie reviews...\n")
clean_train_reviews = []
reviews=[]
for i in range( 0, num_reviews ):
    # If the index is evenly divisible by 1000, print a message
    if( (i+1)%1000 == 0 ):
        print("Analyzing sentiment %d of %d\n" % ( i+1, num_reviews) )       
    text=train2['review']                                                                
    clean_train_reviews.append(review_to_words(text[i]))
    
   
final_Sentiment=[]



for i in range(0,len(clean_train_reviews)):
	pos_freq=0
	neg_freq=0
	for w in pos_bag:
		pos_freq=pos_freq+(clean_train_reviews[i].count(w))
	for w in neg_bag:
		neg_freq=neg_freq+(clean_train_reviews[i].count(w))
	if(pos_freq>=neg_freq):
		final_Sentiment.append(1)
	else:
		final_Sentiment.append(0)
	


print(train2.sentiment.size)
print(len(final_Sentiment))