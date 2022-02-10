'''
# -*- coding: utf-8 -*-
# Author: Michael Bewong (michael.bewong@mymail.unisa.edu.au)
#         Lecturer in Data Science, AI and Cyber Security
#         Data Science Research Unit, Charles Sturt University
#         [Website]

# Contributors: John Wondoh (john.wondoh@mymail.unisa.edu.au)
#               Selasi Kwashie (selasi.kwashie@mymail.unisa.edu.au)
'''
#local imports
import src.classes.docObject_class as dc ## for defined classes

def wordPairIndex(doc,maxDocLen):
    '''
    This calculates the wordpair index in a doc. Here a doc is a dictionary containing the wordpairs in 
    a single ocument and their associated counts. Details of the calculation in the paper.
    -------------------------------------------------------------------------------------------------
    para maxDocLen (int) : This parameter specifies the maximum document length. It is recommended that 
    we generally use the total count of the number wordpairs in the corpus as our maxdoclen. This is 
    done for all onward usage of this function. be updated. 
        
    para doc (dict) : This is a dictionary containing the wordpairs of a single document ie 
    {wordpair:count} 
    '''
    
    probSum = 0
    for x in doc.keys():
        s = (doc[x]/maxDocLen)**2
        probSum += s
    wpIndex = 1-probSum
    
    return wpIndex

def wordPairs (corpus):
    '''
    This generates the wordpairs in in a corpus and returns (1) transcorp: a list of wordpair lists 
    where each wordpair list is the wordpairs generated from a document and each wordpair is a tuple; 
    (2)globalwordpairs which is a dictionary of all the wordpairs ie wordpair(key):count (value).
    ---------------------------------------------------------------------------------------------    
    para doc (corpus) : This is the original corpus which is a list of strings where each string 
    represents a document.
    '''
    globalWordpairs = dict()
    corpusSize = len(corpus)
    transCorp = [None for t in range(corpusSize)]#create an output list
    i=0
    while i < corpusSize:
        allWordpairs = list()
        
        doc = sorted(corpus[i].split())
        docSize = len(doc)
        for x in range(docSize-1):
            y = x+1
            
            while y < docSize:
                #print ("x:", x)
                #print ("y:", y)
                #print(docSize)
                if doc[x]!=doc[y]:
                    wordpair = (doc[x],doc[y])
                    allWordpairs.append(wordpair)
                    if wordpair in globalWordpairs.keys():
                        globalWordpairs[wordpair] = globalWordpairs[wordpair]+1
                    else:
                        globalWordpairs.update({wordpair:1})
                y+= 1
        transCorp[i] = allWordpairs
        i+=1
    return(transCorp,globalWordpairs) 

def wordPairsChecker (termset1, termset2,globalPairs):
    """
    This checks the validity of the candidate pairs by checking if the combination of these termsets 
    will be spurious or not
    -------------------------------------------------------------------------------------
    para termset1 : this is the set of terms in one object ie document being transformed 
    
    para termset2 : this is the set of terms in another object ie document being transformed 
    
    para globalPairs : this a dictionary of all the word pairs and their counts in the corpus 
    ie wordpair:count

    """
    valid = True
    unionOfSets = termset1.union(termset2)
    
    doc = sorted(unionOfSets)
    docSize = len(doc)
    for x in range(docSize-1):
    

        for y in range (x+1, docSize):
            #print ("x:", x)
            #print ("y:", y)
            #print(docSize)
            
            wordpair = (doc[x],doc[y])
            #allWordpairs.append(wordpair)
            if wordpair not in globalPairs.keys():
                valid = False
                break
        else:
            continue
            
        break
    return(valid) 

def initialiseDocObjects (globalPairs):
    '''
    This generates an initial list of objects using the word pairs. Each element in the list is an 
    object defined as an instance of the class "docObjectClass". The function returns the totalcost 
    (wordpair index), maxdoclen (from the total count of wordpairs) and the docobjectlist
    ---------------------------------------------------------------------------------------------    
    para globalPairs (corpus) : This is a dictionary of wordpairs in the whole corpus ie wordpair:count
    '''
    maxDocLen = sum(globalPairs.values())
    #print(maxDocLen)
    docObjectList = []
    totalCost = 0.0
    for x in globalPairs.keys():
        #print(x)
        termSet = set(x)
        #print(termSet)
        termPairsDic = {x:globalPairs[x]}
        cost = wordPairIndex(termPairsDic, maxDocLen)
        #print(cost)
        totalCost +=cost
        newObject = dc.DocumentObject(termSet,termPairsDic,cost)
        docObjectList.append(newObject)
    return(totalCost,maxDocLen, docObjectList)
