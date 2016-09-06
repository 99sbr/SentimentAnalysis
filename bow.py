import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer

#from sets import Set
train = pd.read_csv("/home/subir_sbr/Desktop/labeledTrainData.tsv",delimiter="\t", quoting=3,nrows=5000)


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
            #new_set=" ".join( stem_word )
        return stem_word
    
num_reviews = train["review"].size
BOW_df = pd.DataFrame(0,columns=['-ve','+ve'],index=[''])
print("Cleaning and parsing the training set movie reviews...\n")
clean_train_reviews = []
for i in range( 0, num_reviews ):
    # If the index is evenly divisible by 1000, print a message
    if( (i+1)%1000 == 0 ):
        print("Analyzing sentiment ")
        print("Review %d of %d\n" % ( i+1, num_reviews) )  
    sentiment=train['sentiment']
    text=train['review']  
    split_text=[]                                                               
    split_text=review_to_words(text[i])
    for word in split_text:     
        if word not in clean_train_reviews:
            clean_train_reviews.append(word)
            BOW_df.loc[word]=[0,0]
            BOW_df.ix[word][sentiment[i]] +=1
        else:
            BOW_df.ix[word][sentiment[i]] +=1


print(BOW_df.head(4))
# Use pandas to write the comma-separated output file
BOW_df.to_csv("/home/subir_sbr/Desktop/sentiment_lexicon.csv",header=None)
