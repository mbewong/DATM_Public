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
from sklearn.preprocessing import MinMaxScaler, normalize
from  sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import math
from tqdm import tqdm
#Local Imports
from src.sompy import sompy as sompy
from src.generalFunctions import functions as fs # general functions 
import src.classes.docObject_class as dc ## for defined classes
#------------------------------------------------------------------------------------------------------

def getLDATopics(Docs, n_topics = 11, use_idf = False, n_top_words = 10, n_features = "auto", norm = None):
    '''
    This takes the tweets as a list (called docs) and returns a set of topics (a list of terms lists)
    using the LDA model. 
    -------------------------------------------------------------------------------------------------
    param Docs (list of strings): This is the documents as a list of strings to be analysed.
    
    param n_topics** (integer): This is the number of topics we expect. Default is 11.
    
    param use_idf (True/False): This specifies whether the document term matrix generated is reweighted 
    using idf. This is useful if we are interested to reduce the effect of words based on how frequent 
    the appear in other documents. Default is False.
    
    param n_top_words (integer): This is the number of words to be used to describe each topic. Default 
    is 10.
    
    param n_features ("auto", any integer): This specifies the number of features (terms) to be used in
    building the doc term matrix for the topic model. If auto, transformdoc will calculate a number
    using 0.2 * the number of docs. Else a number should be specified. Default is a string "auto".
    
    param norm *** (None, "l1", "l2"): This specifies whether or not each row of document in the matrix 
    will be normalised or not. Normalisation is required if we dont want the individuaal lengths of 
    documents to affect the overall outcome. Default is none.
    -----------------------------------------------------------------------------------------------
    NOTE:
    ** Experiments show that the "ideal" number to use is 1.2 x of the true topics if known. In our 
    research we are not aiming to solve the problem of specififying the right number of topics. We 
    leave this as an open question for now.
    
    *** In our use case of tweet normalisation may not be needed since it is assumed that tweets are of
    equal length. We thefore donet expect the length of each individual document to affect the final
    result too much. In cases where document lengths differ drastically, this may become necessary.
    
    **** It seems there is a relationship between n_topics and n_features. It looks like the number
    of features should be at least equal to the number of topics x the number of top words. It also
    seems that for every 10000 documents, 500 features should be used. This will be further
    investigated
    '''
    # MODEL: LDA
    tf, tf_feature_names = transformDocs(Docs, use_idf, n_features, norm)
    #tf above is a document term matrix, tf_feature_names are the terms used in the dtm
    
   # print(tf)
    #print("Fitting LDA models with tf features, n_features=%d..."
    #      % n_features)
    lda = LatentDirichletAllocation(n_components=n_topics, 
                                    max_iter=10,
                                    learning_method='batch', 
                                    learning_offset=50.,
                                    random_state=0).fit(tf)

    #In the parameters are more or less default parameters for LDA. Note that our research objective 
    #is not necessarily to change the specs of LDA to improve it, but to show that by a data 
    #transformation technique, we can improve the result of any topic model, including LDA 
    
    #print("\nTopics in LDA model:")
    #print_top_words(lda, tf_feature_names, n_top_words)
    #print(n_topics)

    topic_words = list_top_words(lda, tf_feature_names, n_top_words)
    
    return topic_words


def list_top_words(model, feature_names, n_top_words):
    '''
    This function returns the topics as a list of word lists. 
    -----------------------------------------------------------------------------------------
    param model(topic model): This is the topic model eg lda
    
    param feature_names(list of features): This is the features that were used to build the document
    term matrix. In LDA the positional numbers are used, we use this list to retireve the actual terms.
    
    param n_top_words(integer) : This is the number of words to retrieve per  topic.
    '''
    list_topic = []
    
    for topic_idx, topic in enumerate(model.components_):
        list_mini_topic = []

        for i in topic.argsort()[:-n_top_words - 1:-1]:
            list_mini_topic.append(feature_names[i])
        
        list_topic.append(list_mini_topic)
        
    return list_topic


def transformDocs(Docs, use_idf=False, n_features = "auto", norm = None):
    '''
    This function learns the frequencies (either tf or tfidf) of terms in the corpus and transforms
    the corpus into a document-term matrix. The nature of the matrix depends on the parameters.
    -----------------------------------------------------------------------------------------------
    param Docs(list of strings): This is the documents as a list of strings to be analysed.
    
    param use_idf (True/False): This specifies whether the document term matrix generated is reweighted 
    using idf. This is useful if we are interested to reduce the effect of words based on how frequent 
    the appear in other documents. Default is False.
    
    param n_features ("auto", any integer): This specifies the number of features (terms) to be used in
    building the doc term matrix for the topic model. If auto, transformdoc will calculate a number
    using 0.2 * the number of docs. Else a number should be specified. Default is a string "auto".
    
    param norm *** (None, "l1", "l2"): This specifies whether or not each row of document in the matrix 
    will be normalised or not. Normalisation is required if we dont want the individuaal lengths of 
    documents to affect the overall outcome. Default is none.
    
    NOTE:
    *** In our use case of tweet normalisation may not be needed since it is assumed that tweets are of
    equal length. We thefore donet expect the length of each individual document to affect the final
    result too much. In cases where document lengths differ drastically, this may become necessary.
    
    **** If the combination of use_idf=false and norm = None is used, the end result is the same as a 
    countvectorisor
    
    '''
    
    if n_features == "auto": # Will calculate n_features automatically
        n_features = int(0.2 * len(Docs))
        if n_features < 5 : print("Number of features too low, do not use 'n_features = auto' option,\
        specify a number")     
    
    tf_vectorizer = TfidfVectorizer(input='content',
                                max_df=0.95, 
                                min_df=2,
                                    ngram_range = (1,2),
                                max_features=n_features,
                                stop_words='english', use_idf = use_idf, norm = norm)
    
    tf = tf_vectorizer.fit_transform(Docs)
    featureNames = tf_vectorizer.get_feature_names()
    #print(tf, featureNames)
    return tf, featureNames 
    
def getSOM_LDATopics(Docs, n_topics = 11, use_idf = False, n_top_words = 10, n_features = "auto", norm = None, mapsize = "auto"):
    '''
    This takes the tweets as a list (called docs) and returns a set of topics (a list of terms lists)
    using SOM first for aggregation and then the LDA model. 
    -------------------------------------------------------------------------------------------------
    param Docs (list of strings): This is the documents as a list of strings to be analysed.
    
    param n_topics** (integer): This is the number of topics we expect. Default is 11.
    
    param use_idf (True/False): This specifies whether the document term matrix generated is reweighted 
    using idf. This is useful if we are interested to reduce the effect of words based on how frequent 
    the appear in other documents. Default is False.
    
    param n_top_words (integer): This is the number of words to be used to describe each topic. Default 
    is 10.
    
    param n_features ("auto", any integer): This specifies the number of features (terms) to be used in
    building the doc term matrix for the topic model. If auto, transformdoc will calculate a number
    using 0.2 * the number of docs. Else a number should be specified. Default is a string "auto".
    
    param norm *** (None, "l1", "l2"): This specifies whether or not each row of document in the matrix 
    will be normalised or not. Normalisation is required if we dont want the individuaal lengths of 
    documents to affect the overall outcome. Default is none.
    
    Mapsize is a tuple/list with the map matrix size if non will use the default method to calculate this.
    --------------------------------------------------------------------------------------------
    NOTE:
    ** Experiments show that the "ideal" number to use is 1.2 x of the true topics if known. In our 
    research we are not aiming to solve the problem of specififying the right number of topics. We 
    leave this as an open question for now.
    
    *** In our use case of tweet normalisation may not be needed since it is assumed that tweets are of
    equal length. We thefore donet expect the length of each individual document to affect the final
    result too much. In cases where document lengths differ drastically, this may become necessary.
    
    **** It is seems there is a relationship between n_topics and n_features. It looks like the number
    of features should be at least equal to the number of topics x the number of top words. It also
    seems that for every 10000 documents, 500 features should be used. This will be further
    investigated.

    TODO:
    At present the LDA component is within this function ie, we are calling LatentDirichletAllocation() 
    to finally convert the output of our SOM to a set of topics. To determine the effectiveness of our 
    approach ie SOM+LDA(or any other topic model), we need to use the output of SOM, ie the sMatrix, as
    input to the other techniques such as BTM et al. This should be a simple fix.
    '''
    
    tf,featureNames = transformDocs(Docs, use_idf, n_features, norm)
    #tf above is a document term matrix, tf_feature_names are the terms used in the dtm
    tfDense =tf.todense() 
    #Above is required to convert matrix to dense format so that it is compatible with SOM
    if mapsize == "auto":
        m = int(math.sqrt(0.7* len(Docs))) #- We noticed that about mapsize which is about 2/3 of the data size gives good results. 
        #NOTE that I work this out becaus the inbuilt function in sompy is not correct.
        #m = int(math.sqrt(5 * math.sqrt(len(Docs)))) #- Reference in Shalaginov 2015
        mapsize = [m,m]#[50,50]
        print("Your mapsize is :", mapsize)
    som = sompy.SOMFactory.build(tfDense, mapsize, mask=None,\
                                         mapshape='planar', lattice='rect',\
                                         normalization='var', initialization='pca',\
                                         neighborhood='gaussian', training='batch', name='sompy')  
    #Defaults have been used in the above.
    
    som.train(n_job=100, verbose='debug', #train_rough_len=2,train_finetune_len=2, 
              train_len_factor = 2) 
    #Defaults are being used in the above.
    #n_job - We may use this to paralellize the process to invsitigate times
    #train_len factor will multiply each train len by the factor. The training len are calculated by 
    #defualt if not specified but note that the trainining len for rough and fine has to be specified 
    #for cases with very small number of documents otherwise the operations take forever.
    #Train len for all cases is the number of iterations to be used.
    
    #NOTE: verbose='debug' will print more, 'Info' will print some text and verbose=None wont
    #print anything. We use defaults for now.
    
    sMatrix = np.matrix(som.codebook.matrix)
    scaler = MinMaxScaler()
    sMatrixTranspose = sMatrix.T
    #Smatrix is a document- term matrix ( documents are rows and terms are columns). MinMaxsclaer transforms each column to lie between 0 and 1. By doing this with the sMatrix it means that we are normalising along the column axis. This removes the relative distribution of words from the document.
    #By transposing the smatrix, we now have doucments as columns and terms as rows.
    scaler.fit(sMatrixTranspose)
    sMatrixTranspose = scaler.transform(sMatrixTranspose)
    #Now when we apply the min max scaler we normalise all the terms within a document and not a term accross all teh documents. This way the relative distribution within a document is preserved
    
    sMatrix = sMatrixTranspose.T
    #By re-transoposing the results, we rturn the normalised matrix back into a document term matrix with the documents being the rows and the terms being the columns. This is fed to LDA.
    #The folllowing LDA spec should be the same as the one used in LDAOnly.

    lda = LatentDirichletAllocation(n_components=n_topics, 
                                    max_iter=10,
                                    learning_method='batch', 
                                    learning_offset=50.,
                                    random_state=0).fit(sMatrix)
    
    topic_words = list_top_words(lda, featureNames, n_top_words)
    
    return som, topic_words
  
def TfidfVectorizer_UDF(GSR_text, n_features):
    """
       This transforms the day's corpus (GSR_text) into a matrix and returns this matrix along with the 
       vocabulary
       i.e. a list of the feature names.Here we vectorizer functions from Sklearn, namely
       CountVectorizer and tfidfVectorizer to convert the text to a matrix. See sklearn for more
       details on this.
       ------------------------------------------------------------------------------------------------
       param GSR_text : this is the corpus of tweets as a list of the format [tweet_1,...,tweet_n] to
       be converted to a matrix. 

       param n_features*** : this is the number of features to be extracted from the corpus and used to 
       construct the matrix. There may be no need to worry about this as
       -------------------------------------------------------------
       NOTES
       *** - these parameters affect the results
       *** - the parameters in tf_vectorizer1 can affect the results especially ngram_range
           - the tfidf weighting might also affect the analysis
    """
    tf_vectorizer1 = CountVectorizer(input='content',
                                        #max_df=0.95, 
                                        #min_df=2,
                                        ngram_range = (1,3),
                                        #max_features=10000,
                                        stop_words='english')
    #***the parameters in tf_vectorizer1 can affect the results especially ngram_range******
    tf1 = tf_vectorizer1.fit_transform(GSR_text)
   # print(tf_vectorizer1.get_feature_names())
    
    tf1dense=tf1.todense()
    #print(tf1dense)
    sumtf=np.sum(tf1dense, axis=0)
    sumtf=np.squeeze(np.asarray(sumtf))
   # print(tf1dense)
    #TF.1. The above gives the tf values for the features 
    tf_vectorizer = TfidfVectorizer(input='content',
                                    #max_df=0.95, 
                                    #min_df=2,
                                    ngram_range = (1,3),
                                    #max_features=10000,
                                    stop_words='english')
    tf=tf_vectorizer.fit_transform(GSR_text)
   # print(tf_vectorizer1.get_feature_names())
    newIDF= tf_vectorizer.idf_
    #print(len(newIDF), len(sumtf))
   #TF.2. The above gives the idf values for the features
  
    idftfScores=[newIDF[i]*sumtf[i] for i in range(len(newIDF)) ]
   #TF.3. The above multiplies the tf and idf to give me the actual tfidf values** Potential issue is the multiplication can be mismatched. It doesnt seem like this will happen though.   
    
    Allterms=tf_vectorizer.get_feature_names()
    idftfScores=np.array(idftfScores)
    topMostidf=idftfScores.argsort()
    topMostidf = topMostidf[::-1]
    vocab = [Allterms[x] for x in topMostidf[:n_features]]
    tf_vectorizer = TfidfVectorizer(input='content',
                                    #max_df=0.95, 
                                    #min_df=2,
                                    #max_features=10,
                                    vocabulary =vocab,
                                    stop_words='english')
        #print("vocab", vocab)
    #print(tf_vectorizer.fit_transform(GSR_text))
    print("original", vocab)
    return (tf_vectorizer.fit_transform(GSR_text), vocab)

def malg(corpus,noOfBuckets):
    '''
    *malg means to fix / to make better in Nabt (a Gur language of the people of northen Ghana) 
    
    malg takes the documents eg tweets as a list (called corpus) and returns a transformed corpus ( a 
    list of strings) which can then be fed to any topic model such as LDA, BTM or PTM to produce the 
    topics. 
    
    -------------------------------------------------------------------------------------------------
    para corpus :  This is the documents as a list of strings to be analysed. Same as Docs in 
    getSOM_LDATopics and getLDATopics
    
    para noOfBuckets :  This is the number of final documents that we want in the end. This is a user 
    specified constraint. Detailed in the paper
    '''
    transCorp, globalPairs =fs.wordPairs(corpus)
    #m1 : wordPairs gets combinations within a document and save finally as a list of wordpair lists
    #(trasncorp). Also globalPairs is a dictionary that counts all the word pairs across multiple docs.

    totalCost, maxDocLen,docObjectList = fs.initialiseDocObjects (globalPairs)
    #m2 : Here we copy the global pairs dictionary as new objects and assign these as a list of objects
    
    foldedDocObjectList =   foldback(docObjectList,noOfBuckets, maxDocLen, totalCost, globalPairs) 
    #m3 : Here we begin to fold the docObjects back on themselves based on the noOfBuckets that we 
    #expect/defined 
    
    transformedCorpus = genFinalCorpus(docObjectList)
    
    return(transformedCorpus)


def malg2(corpus):
    ''' just not to touch the main algo '''
    transCorp, globalPairs =fs.wordPairs(corpus)
    totalCost, maxDocLen,docObjectList = fs.initialiseDocObjects (globalPairs)
    # we return a dictionary with everything required for folding
    result = {'transCorp': transCorp,
              'globalPairs': globalPairs,
              'totalCost': totalCost,
              'maxDocLen': maxDocLen,
              'docObjectList': docObjectList}
    return result


def folded_docs(docObjectList,noOfBuckets, maxDocLen, totalCost, globalPairs):
    """ just so I can combine folding and generating the final corpus"""
    foldedDocObjectList = foldback(docObjectList,noOfBuckets, maxDocLen, totalCost, globalPairs)
    transformedCorpus = genFinalCorpus(foldedDocObjectList)

    return(transformedCorpus)


def foldback(docObjectList,noOfBuckets, maxDocLen, totalCost, globalPairs, verbose = True):
    '''
    This function folds the objects in the docObjectList back on themselves such that we minimise the 
    overall wordpair cost and meet the maximum number of transformed documents determined by 
    noOfBuckets.
    -------------------------------------------------------------------------------------
    para docObjectList : this is the list of document objects.
    
    para noOfBuckets: This is the number of final documents that we want in the end. This is a user 
    specified constraint. Detailed in the paper.
    
    para totalCost: This is the total cost from the docObjectList. 
    
    para maxDocLen (int) : This parameter specifies the maximum document length. It is recommended that 
    we generally use the total count of the number wordpairs in the corpus as our maxdoclen. This is 
    done for all onward usage of this function. be updated. 
    
    para globalPairs : this a dictionary of all the word pairs and their counts in the corpus 
    ie wordpair:count
    
    para verbose : This indicates whether details will be printed for debugging or not. Default is True
    ----------------------------------------------------------------------------------------
    TODO: Explore the use of ideas of KD trees to improve the efficiency ie use a binary search.
    '''
    if verbose:
        print("The total initial cost is " + str(totalCost))
    #sort docObjectList by cost. This ensures that we use the biggest cosst object for the 
    #scan. The logic is that a merged document will be cheaper than the cheaper of the two
    # hence by using the biggest at teh start, we eliminate that lareg cost component.
    #This will go into the paper after the algorithm
    #print(len(docObjectList))#
    docObjectList.sort(key=lambda x: x.cost, reverse = True)
    #docObjectList_sorted = sorted(docObjectList, key=lambda x: x.cost, reverse = True)
    #for u in range(docObjectList):
     #   print(docObjectList[u].cost)
    #[print(x.cost) for x in docObjectList]
    #print(len(docObjectList))#[0].cost)
    
    curNoOfBuckets = len(docObjectList)
    noFolds = curNoOfBuckets - noOfBuckets
    
    if verbose:
        print ("The number of folds to perform is ", noFolds)
        print ("The number documents is ", curNoOfBuckets)
    if noFolds <= 0:
        print("input documents is less than or equal required no of buckets")
    else:
        
        x = 0
        pbar = tqdm(total=noFolds)
        while len(docObjectList) > noOfBuckets and x < len(docObjectList):
            # we can use x to mean incremen in the number of folds as well as the next 
            #candidate doc to check
            #print("doc len is",len( docObjectList))
            #print("x is ",x)
            candiDocObj = docObjectList[x]
            z = len(docObjectList)
            minCost = None
            comboValid = False
            while not comboValid and z> 0:
                #print("Before",z)
                z-=1
                #print("After", z)
                #print(len(docObjectList))
                comboValid, candiCost = docObjectList[z].checkObj(candiDocObj, globalPairs, 
                                                                  maxDocLen)
              #  print("Candidate cost ",z, candiCost)
                '''
            
                if z == 1:
                    minCost = candiCost
                    minIndex = z
                elif candiCost is None:
                    continue
                elif candiCost <minCost:
                    minCost = candiCost
                    minIndex = z
                '''
                minIndex = z
                minCost = candiCost
                    
            if minCost is not None:
                previousCost = docObjectList[minIndex].cost
                newCost = minCost
                docObjectList[minIndex].updateObj(candiDocObj, newCost)
                totalCost = totalCost - previousCost - candiDocObj.cost +newCost
                #print("previousCost1, previousCOst2 newCost ",previousCost, candiDocObj.cost, newCost   )
                del docObjectList[x]
            else:
                x+= 1
            #sleep(0.5)    
            #tqdm._instances.clear()
            pbar.update(1)
            tqdm._instances.clear()
        pbar.close()
        #tqdm._instances.close()
        if verbose:
            print("The new total cost after round x is ", totalCost, "and new corpus size is ", 
                  len(docObjectList))
    return (docObjectList)                
    
    
    
def genFinalCorpus(docObjectList):
    '''
    This function returns the new list of documents by transfomring the docObjectList. Remember that a
    document is a string and a corpus is a list of document lists.
    -----------------------------------------------------------------------------------------------
    para docObjectList : this is the list of document objects.
    '''
    
    newCorpus = []
    for x in range(len(docObjectList)):
        #print (x)
        newDoc = ""
        
        for t1,t2 in docObjectList[x].termPairsDic.keys():
            #print("terms", t1,t2)
            rep = docObjectList[x].termPairsDic[(t1,t2)]
            #print(rep)
            newT = t1+" "+t2+" "
            #print (newT)
            tempStr = newT*rep
            tempStr.strip()
            #print("tempStr", tempStr)
            newDoc = newDoc+tempStr
            #print("newDoc", newDoc)
            #t2 = t2+" "
            #tempStr = t2*rep
            #print("2nd", tempList)
            #newDoc.extend(tempList)
        
        newCorpus.append(newDoc.strip())
    return(newCorpus)
    
    
    
    
    