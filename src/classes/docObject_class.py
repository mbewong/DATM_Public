'''
# -*- coding: utf-8 -*-
# Author: Michael Bewong (michael.bewong@mymail.unisa.edu.au)
#         Lecturer in Data Science, AI and Cyber Security
#         Data Science Research Unit, Charles Sturt University
#         [Website]

# Contributors: John Wondoh (john.wondoh@mymail.unisa.edu.au)
#               Selasi Kwashie (selasi.kwashie@mymail.unisa.edu.au)
'''
#Classical Classes
#import math

#Local imports # can be deleted if not needed
#import src.features.build_globalVariables as  gv## For global variables
from src.generalFunctions import functions as fs

class DocumentObject:
    """
    This is class creates objects that represent the transformed document as a set of termpairs 
    (wordpairs)
    
    *Each object describes a document and has the following properties:
    -----------------------------------------------------------------------------------
    property termSet : This is the set of terms when all the term pairs are combined
    
    property termPairsDic: This is a dictionary of the termpairs and their counts within the document.
    
    property cost : this is the total cost of the transformed document ie. sum of wordpair index for 
    each wordpair in the transformed doc.
    
    *The following describes the functions related to the object:
    -------------------------------------------------------------------------
    function checkObj : checks the validity of the candidate pairs ie is for any two wordpairs, will
    their combination lead to a spurious wordpair - a wordpair that never existed in the original 
    document
    
    function updateObj : checks the validity of the candidate pairs ie is for any two wordpairs, will
    their combination lead to a spurious wordpair - a wordpair that never existed in the original 
    document
    
    """
    def __init__(self, termSet, termPairsDic, cost):
        self.termSet = termSet
        self.termPairsDic = termPairsDic
        self.cost = cost
        
    def checkObj(self, candiDocObj, globalPairs, maxDocLen):
        """
        This checks the validity of the candidate pairs first and if valid calculates the cost of
        merging the candiDocobj to the current object in question.
        -------------------------------------------------------------------------------------
        para candiDocObj : this is object (word combination) to be merged with the current object
        
        para globalPairs : this a dictionary of all the word pairs and their counts in the corpus 
        ie wordpair:count
        
        para maxDocLen : This parameter specifies the maximum document length. It is recommended that
        we generally use the total count of the number wordpairs in the corpus as our maxdoclen. 
        """
        #the following checks whether the termpairs will lead to a spurios term pair or not
        newTempPairsDic = None
        candiCost=None
        valid = fs.wordPairsChecker(candiDocObj.termSet, self.termSet, globalPairs)
        
        if valid:
            
            newTempPairsDic = dict(self.termPairsDic)
            for wordPair in candiDocObj.termPairsDic.keys(): 
                if wordPair in newTempPairsDic.keys():
                    newTempPairsDic[wordPair] = newTempPairsDic[wordPair] + candiDocObj.termPairsDic[wordPair]
                else:
                    newTempPairsDic[wordPair] = candiDocObj.termPairsDic[wordPair]
            #newTempPairsDic = dict(self.termPairsDic, **candiDocObj.termPairsDic)
            candiCost = fs.wordPairIndex (newTempPairsDic, maxDocLen)
        return (valid, candiCost)
    
    def updateObj(self,candiDocObj, newCost):
        """
        This merges the candidate object with the current object and updates the cost
        -------------------------------------------------------------------------------------------
        para candiDocObj : this is object (word combination) to be merged with the current object 
        
        para newCOst : this is cost calculated from the checkObj function which determines the cost of 
        merging the candidate object with this current object ie total wordpair cost.
        """
        #newTempPairsDic = dict(self.termPairsDic, **candiDocObj.termPairsDic)
        newTempPairsDic = dict(self.termPairsDic)
        for wordPair in candiDocObj.termPairsDic.keys(): 
            if wordPair in newTempPairsDic.keys():
                newTempPairsDic[wordPair] = newTempPairsDic[wordPair] + candiDocObj.termPairsDic[wordPair]
            else:
                newTempPairsDic[wordPair] = candiDocObj.termPairsDic[wordPair]
        
        
        self.termPairsDic = newTempPairsDic
        
        self.cost = newCost
        
        self.termSet = self.termSet.union(candiDocObj.termSet)
        
        return ()
    
    