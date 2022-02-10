'''
# -*- coding: utf-8 -*-
# Author: Michael Bewong (michael.bewong@mymail.unisa.edu.au)
#         Lecturer in Data Science, AI and Cyber Security
#         Data Science Research Unit, Charles Sturt University
#         [Website]

# Contributors: John Wondoh (john.wondoh@mymail.unisa.edu.au)
#               Selasi Kwashie (selasi.kwashie@mymail.unisa.edu.au)

'''
#Comments by MIKE:  This file will be included in the main package for publication.
#-------------------------------------------------------------------------------
#Classical Imports
#from sklearn.preprocessing import MinMaxScaler # To be removed in final cleaning if not needed
#from  sklearn.metrics.pairwise import cosine_similarity # To be removed in final cleaning if not needed
import numpy as np
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#from sklearn.decomposition import LatentDirichletAllocation # To be removed in final cleaning if not needed
#import math # To be removed in final cleaning if not needed
#-------------------------------------------------------------------------------
#Local Imports
from src.sompy import sompy as sompy
#-------------------------------------------------------------------------------
#Read data in pkl format 

def getData(source):
    '''
    Returns the pickle file given as the source
    --------------------------------------------
    param source (path) : This is the path to the .pkl file that we try to read in 
    eg. 'airline_DF_Clean.pkl' look for 'airline_DF_Clean.pkl' within the local folder.
    --------------------------------------------
    NOTES: the path to the file has to be included as part of the "source" para if .pkl file is located 
    in a different folder
    '''
    return(pd.read_pickle(source))
#Convert Data from json to pckle
def saveData(file, fileName):
    '''
    Saves file as pickle (under construction)
    -------------------------------------------
    param file (pandas dataframe) : This is the pandas dataframe that we want to save for later re-use.
    
    param fileName (string) : This is the filname that we want to give to the "file" that we persist in
    the drive
    --------------------------------------------
    NOTES: This needs to be improved to get the file paths automatically. At present the file path
    needs to be included in the file name. If a path is not included in the the "fileName" it will be 
    stored in the local working folder.
    '''
    file.to_pickle(fileName)
    
def getTopicLabelledTweets(dataSource='airline_DF_Clean.pkl', exclude = ["Can't Tell"], save = False):
    '''
    This function is used to get all tweets with a labelled category. It returns the 
    dataframe of labelled tweets. The tweets associated with 'excluded' are removed. 
    Example: exclude  = ["Can't Tell"] will exclude any topics labelled as Can't Tell
    -----------------------------------------------------------------
    param dataSource (path) : This is the path to the pickle file of the cleaned tweets with labels.
    The default is our example airline dataset.
    
    param exclude (list) : This is a list of labels for which tweets related to must be excluded.
    
    param save (boolean) : This determines whether we save the result or not. 
    --------------------------------------------------------------------
    NOTES: Param save will cause the file to be saved in the folder containg the .pkl in "dataSource"
    param
    
    TODO: Put reference for the source of original airline data here if code is published, otherwise
    use a dummy dataset.
    '''
    tweets_DF =getData(dataSource)
    tweets_DF = tweets_DF[(~tweets_DF.negativeReason.isin(exclude)) & \
                                                      (tweets_DF.negativeReason.notnull())]
    
    if save:
        fileName = dataSource[0:dataSource.index('.')]
        tweets_DF.to_pickle(fileName+"_knownTopics.pkl")
    return tweets_DF
    

# def textOfTopicAsList(tweets, topic = "All", save = False):
def textOfTopicAsList(tweets, topic = "All", label="negativeReason", text="text", save = False):
    """
    This function takes the tweets dataframe and returns the tweets associated
    with the param 'topic' as a list.
    -------------------------------------------------------------------------------------
    param tweets (dataframe) : this is the input tweets which we want to analyse as dataframe data
    type. 
    
    param topic (string): This specifies which topic's tweets we want to obtain from the input tweets
    dataframe. Default is All. This means that we get all the text as a list regardless of the topic
    
    param save (boolean): This determines whether we save the resulting list of tweets or not
    ----------------------------------------------------------------
    NOTES: param save, if true, will cause the list of tweets to be saved as a .pkl file in the local
    working folder
    """
#     columnIndex1 = tweets.columns.get_loc("negativeReason")
#     columnIndex2 = tweets.columns.get_loc("text")
    columnIndex1 = tweets.columns.get_loc(label)
    columnIndex2 = tweets.columns.get_loc(text)
    #print('columnIndex :', columnIndex )
    d = list()
    
    if topic != "All":
        for row in tweets.itertuples():
            #print(str(row[columnIndex+1 ]), ' and ', topic)
            if  str(row[columnIndex1 +1]) == topic:
                #print("yes", str(row[columnIndex1 +1]))
                d.append(str(row[columnIndex2 +1]))
                #print(row)
    else:
        for row in tweets.itertuples():
            d.append(str(row[columnIndex2 +1]))
    
    topicTweetList = d
    topicTweetDF=pd.DataFrame(data = d)
    if save:
        topicTweetDF.to_pickle(topic+".pkl")
    return topicTweetList
    
def getKeywordsOfTopics(topicTweetList, numberofTerms = 20):
    '''
    This function gets the keywords of tweets associated to a particular topic. It takes as input the 
    list of tweets related to that topic and returns 'numberofTerms' top keywords.
    -------------------------------------------------------------------------------------------------
    param topicTweetList (list): This is the list of tweets pertaining to a specific topic for which we
    want to get the keywords
    
    param numberofTerms (int): This is the number of keywords that we want to retrieve from the tweets 
    list 
    -------------------------------------------------------------------------------------------------
    NOTES: 
    1. This step is considered a screening step from which we now select manually the best 50% of 
    'numberofTerms' terms that describe the  topic. If the top 5 words that are returned naturally from 
    this function are specific and semantically related to the topic, we call the topic an Easy (E) 
    topic, otherwise we call it Hard (H). At present we do some manual intervention, I suggest we
    get the top 20 biterms and then we select 5 ( or more, if the terms have the same support and are
    relevant) from this to be used.
    
    2. We are interested in the most used bigrams becuase these often have more semantic meaning than
    single words. Since in LDA and LDA-SOM we only get single words, it justifes why our matching 
    criteria in the evaluation section only uses 50% match. That is so long as we get one of the terms 
    in the pair (bigram) then we think that is enough.
    
    '''
    numberofTerms =numberofTerms
   # tf_vectorizer = TfidfVectorizer(input='content',
    #                                    #max_df=0.95, 
    #                                    #min_df=10,
    ##                                    ngram_range = (2,3),
    ##                                    max_features=30,
    #                                    stop_words='english')
    tf_vectorizer = CountVectorizer(input='content',
                                        #max_df=0.95, 
                                        #min_df=10,
#                                         ngram_range = (2,2),
                                        ngram_range = (2,2),
                                        max_features= numberofTerms,
                                        stop_words='english')

    
    tf = tf_vectorizer.fit(topicTweetList)
    topMosttfterms=tf.get_feature_names()
    #print(topMosttfterms)
    
    cv = tf_vectorizer.transform(topicTweetList)
    sum_words = cv.sum(axis=0)
    
    words_freq = [(word, sum_words[0, idx]) for word, idx in tf.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    
    print(words_freq)
    average_words_freq = np.mean([i[1] for i in words_freq])
    print("average support", average_words_freq)
    return topMosttfterms
    #print(cv.toarray().sum(axis=0))
    #print(tf_vectorizer.vocabulary_)
    #So up to this point, we get the top twenty most common terms.
    '''
    We may choose to save it as a file by using some of the following code:
    
    
    #Next we order  these most common terms by their idf
    topMostidf=tf_vectorizer.idf_.argsort()
    #print(topMostidf)
    topMostidf = topMostidf[::-1]
    print(topMostidf)
    #topMostidf = sorted(topMostidf, reverse=True)
    #print(topMostidf)
    #topMosttfterms[]
    #topMostidf[0:2]

    vocab = [topMosttfterms[x] for x in topMostidf[:numberofTerms]]
    print(vocab)
    
    tf_vectorizer = TfidfVectorizer(input='content',
                                    #max_df=0.95, 
                                    #min_df=2,
                                    #max_features=10,
                                    vocabulary =vocab,
                                    stop_words='english')

    tf = tf_vectorizer.fit_transform(topicTweetList)

    thekeywords=tf_vectorizer.get_feature_names()
    keywords = ','.join(thekeywords)
    print(keywords)
    
    #f = open('TopicKeywordsDic.txt','w')
    #a = input('is python good?')
    #f.write('Lost Luggage:'+keywords)
    #f.close()
    '''
def cleanText_udf(text):
    '''
    This function removes some unecessary characters as well as stop words from the tweets. Takes as
    input a string and returns a string
    -------------------------------------------
    param text (string) : This is the string value which we want to clean.
    '''
    tmp = stop.sub('', problemchars.sub('', emojis.sub('', url_finder.sub
                                                       ('', username.sub('', stop2.sub
                                                                         ('', text.lower().strip()))))))
    tmp = lemmaWordsinString(tmp)
    #tmp = stop2.sub('')
    return tmp

def lemmaWordsinString(strinput):
    '''
    This function lematizes a strings
    -------------------------------------------
    param text (string) : This is the string value which we want to clean.
    '''
    wnl = WordNetLemmatizer()
    keys = re.findall(r"[\w']+", strinput)
    
    
    keys = [wnl.lemmatize(w) for w in keys] # default POS is noun 'n' thus this part considers words to be nouns and lemmatizes them eg attendants -> attendant
    keys = [wnl.lemmatize(w, 'v') for w in keys] # adding 'v' lemattizes verbs so that 'loving' is 'love', however, still not work for 'could'
    tmp = ' '.join(keys)
    return tmp

url_finder = re.compile(r'http[s]?:\/\/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
username = re.compile(r'(@)\w+( )')
stop = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
problemchars = re.compile(r'[\[=\+/&<>;:!\\|*^\'"\?%$.@)°#(_\,\t\r\n0-9-—\]]')
emojis = re.compile(u'['
    u'\U0001F300-\U0001F64F'
    u'\U0001F680-\U0001F6FF'
    u'\u2600-\u26FF\u2700-\u27BF]+', 
    re.UNICODE)
# this dictionary is to remove those words that are frequent in GSR text but very unlikely in tweets or other social media as signals
stopwordsGSR = ["amp"]
stop2 = re.compile(r'\b(' + r'|'.join(stopwordsGSR) + r')\b\s*')

