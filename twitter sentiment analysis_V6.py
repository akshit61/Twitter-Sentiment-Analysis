# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:06:09 2020

@author: DELL
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup 
from collections import Counter
from wordcloud import WordCloud
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,TweetTokenizer
#from nltk import ngrams
#from gensim.corpora import Dictionary
#from gensim.models.phrases import Phraser,Phrases

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
pd.set_option('display.max_rows', None,'display.max_columns', None)

df = pd.read_csv('train.csv',encoding='ISO-8859-1')
df['sentiment'].value_counts()

#tweet = df['tweet'].str.cat()
tweet = df['tweet'].str.cat()
emos = set(re.findall(r"[xX:;][-']?.",tweet))
emos_count = []
for emo in emos:
    emos_count.append((tweet.count(emo),emo))
    
happy_emo = r' ([xX;:]-?[dD)]|:-?[\)]|[;:][pP]) '
sad_emo = r" (:'?[/|\(]) "
print("Happy emoticons:", set(re.findall(happy_emo, tweet)))
print("Sad emoticons:", set(re.findall(sad_emo, tweet)))


def find_most_common_words(tweet_text):
    tweettokenizer = TweetTokenizer(strip_handles  =True)
    tokens = tweettokenizer.tokenize(tweet_text.lower())
    most_common_words = nltk.FreqDist(tokens)
    #return sorted(most_common_words,key = most_common_words.__getitem__,reverse = True)[:100]
    top_100_words = pd.DataFrame(Counter(most_common_words).most_common(200))
#    plt.figure(figsize=(14,10))
#    sns.barplot(x = 0,y =1,data = top_100_words)
#    plt.xticks(rotation = 90)
#    plt.show()
    return top_100_words

find_most_common_words(tweet)

df['tweet'] = df['tweet'].str.lower()
df['hastags'] = df['tweet'].apply(lambda x: re.findall('#[\w\d]*',str(x)))

def hashtag_analysis(df,sentiment):
    hashtags_list= []
    hashtags_list.extend(df[df['sentiment'] == sentiment]['hastags'].apply(lambda x: x))
    hashtags =[]
    for i in hashtags_list:
           hashtags.extend(i)
    #hashtags = set(hashtags)
    hashtags = Counter(hashtags)
    #most_common_hashtags = pd.DataFrame(hashtags.most_common(30))
#    plt.figure(figsize = (12,10))
#    sns.barplot(x = 0, y=1,data= most_common_hashtags)
#    plt.xticks(rotation = 45)
#    plt.show
    return hashtags
    
hashtags_0 = hashtag_analysis(df,0)
hashtags_1 = hashtag_analysis(df,1)
hashtags_2 = hashtag_analysis(df,2)
hashtags_3 = hashtag_analysis(df,3)

common_hashtags = set(hashtags_0).intersection(set(hashtags_1))
common_hashtags = common_hashtags.intersection(set(hashtags_2))
common_hashtags = list(common_hashtags.intersection(set(hashtags_3)))

stopwords = ['the','to','at','for','a','rt','in','is','of','and','on','i','you','an','my','it','this','be','are','by','that','me','i\'m','it\'s','w','as','we','our','s','us']

df_clean = pd.DataFrame(columns = ['tweet'])
class Text_Processing(BaseEstimator,TransformerMixin):
    def __init__(self,common_hashtags=[],use_handles = False):
        self.common_hashtags = common_hashtags
        self.use_handles = use_handles
    def fit(self,X,y=None):
        return self
    
    def clear_links(self,X):
        try:
            pattern_1 = r'{link}'
            pattern_2 = r'http[s]*:\/\/[a-zA-Z0-9./]+'
            pattern_3 = r'[a-z0.9]+.*[a-z0-9]+\/[a-zA-Z0-9./]+'
            combined_pattern = r'|'.join((pattern_1,pattern_2,pattern_3))
            X['tweet']= X['tweet'].str.replace(combined_pattern,'')
            return X
        except Exception as e:
            print('Exception occured in clear_links; exception message : ',str(e))
    
    def clear_common_hashtags(self,X):
        try:
            for hashtag in self.common_hashtags:
                X['tweet'] = X['tweet'].str.replace(hashtag,'')
            return X
        except Exception as e:
            print('Exception occured in clean_common_hashtags; exception message : ',str(e))            
    
    def tokenize_tweet(self,X,strip_handle_bool = False):
        try:
            tweet_tokenizer = TweetTokenizer(strip_handles = strip_handle_bool,reduce_len = True)
            X['tokenized_tweet'] =X['tweet'].apply(lambda x:tweet_tokenizer.tokenize(x))
            return X['tokenized_tweet']
        except Exception as e:
            print('Exception occured in tokenize_tweet; exception message : ',e)
 
    
    def transform(self,X,y = None):
        X['tweet'] = X['tweet'].apply(lambda row :BeautifulSoup(str(row),'lxml').get_text())
        X = self.clear_links(X)
        X = self.clear_common_hashtags(X)
        X['tweet'] = X['tweet'].str.replace('ipad 2','ipad two')
        X['tweet'] = X['tweet'].str.replace('#','')
        X['tweet'] = X['tweet'].str.replace(r"(.)\1+", r"\1\1")
        X['tweet'] = X['tweet'].str.replace(happy_emo,'happyemoticon')
        X['tweet'] = X['tweet'].str.replace(sad_emo,'sademoticon')
        
        X['tweet'] = X['tweet'].apply(lambda x : x.encode('ascii', errors='ignore').strip().decode('ascii'))
        X_new = pd.Series(index = X.index.tolist(),name = 'tokenized_tweet')
        X_new = self.tokenize_tweet(X,strip_handle_bool=self.use_handles)
        X_new = X_new.apply(lambda row : [x for x in row if x not in stopwords])
        X_new = X_new.apply(lambda row: [x for x in row if re.search('[\w]+',x)])
        X_new = X_new.apply(lambda row: [x for x in row if len(x)>2])
        lemmatizer = WordNetLemmatizer()
        X_new =  X_new.apply(lambda row: [lemmatizer.lemmatize(x) for x in row ])
        X_new = X['tweet']
        return X_new
    
            
vectorizer = TfidfVectorizer(ngram_range = (1,2),max_features=None)

pipeline = Pipeline([('text_pre_processing',Text_Processing(common_hashtags = common_hashtags,use_handles = True)),
                     ('vectorizer',vectorizer)])    

X_train,X_test,y_train,y_test = train_test_split(df[['tweet_id','tweet']],df['sentiment'],test_size = 0.2,random_state = 42)    
X_train_vectorized = pipeline.fit_transform(X_train).toarray()
X_test_vectorized = pipeline.transform(X_test).toarray()



gnb = MultinomialNB(alpha = 0.5)
gnb.fit(X_train_vectorized,y_train)
gnb.score(X_test_vectorized,y_test)

lr = LogisticRegression(random_state=42)
ovr = OneVsRestClassifier(estimator = lr)
ovr.fit(X_train_vectorized,y_train)
ovr.score(X_test_vectorized,y_test)

svc = LinearSVC(random_state = 42)
svc.fit(X_train_vectorized,y_train)
svc.score(X_test_vectorized,y_test)

bnb = BernoulliNB(alpha =0.7)
bnb.fit(X_train_vectorized,y_train)
bnb.score(X_test_vectorized,y_test)


gridSearchPipeline = Pipeline([('text_pre_processing',Text_Processing()),
                               ('vectorizer',TfidfVectorizer()),
                               ('model',LinearSVC())])

params = {'text_pre_processing__common_hashtags':[[],common_hashtags],
          'text_pre_processing__use_handles': [True,False],
          'vectorizer__max_features':[5000,10000,20000,40000,None],
          'vectorizer__ngram_range':[(1,2),(1,1)]}

gridSearch = GridSearchCV(gridSearchPipeline,param_grid = params,scoring = 'f1_weighted',cv = 5)
gridSearch.fit(df[['tweet_id','tweet']],df['sentiment'])

'''Pipeline(memory=None,
         steps=[('text_pre_processing',
                 Text_Processing(common_hashtags=['#sxswi', '#apple', '#iphone',
                                                  '#sxswâ', '#tapworthy',
                                                  '#uxdes', '#android',
                                                  '#bettersearch', '#circles',
                                                  '#google', '#fail', '#japan',
                                                  '#ipad2', '#startupbus',
                                                  '#sxsw', '#newsapps', '#qagb',
                                                  '#news', '#austin', '#ipad'],
                                 use_handles=True)),
                ('vectorizer',
                 TfidfVectorizer(analyzer='word'...
                                 preprocessor=None, smooth_idf=True,
                                 stop_words=None, strip_accents=None,
                                 sublinear_tf=False,
                                 token_pattern='(?u)\\b\\w\\w+\\b',
                                 tokenizer=None, use_idf=True,
                                 vocabulary=None)),
                ('model',
                 LinearSVC(C=1.0, class_weight=None, dual=True,
                           fit_intercept=True, intercept_scaling=1,
                           loss='squared_hinge', max_iter=1000,
                           multi_class='ovr', penalty='l2', random_state=None,
                           tol=0.0001, verbose=0))],
         verbose=False)'''


best_model = gridSearch.best_estimator_
best_model.fit(X_train,y_train)
best_model.score(X_test,y_test)

df_test = pd.read_csv('test.csv',encoding = 'utf-8')
#X_cv_vectorized = pipeline.transform(df_test).toarray()
#y_pred = svc.predict(X_cv_vectorized)
y_pred = best_model.predict(df_test)
df_out = pd.Series(data =y_pred,index = df_test['tweet_id'])
df_out.to_csv('Submission.csv',header = ['sentiment'],index_label = 'tweet_id')
