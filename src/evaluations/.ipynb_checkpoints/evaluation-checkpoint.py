'''
# -*- coding: utf-8 -*-
# Author: Michael Bewong (michael.bewong@mymail.unisa.edu.au)
#         Lecturer in Data Science, AI and Cyber Security
#         Data Science Research Unit, Charles Sturt University
#         [Website]

# Contributors: John Wondoh (john.wondoh@mymail.unisa.edu.au)
#               Selasi Kwashie (selasi.kwashie@mymail.unisa.edu.au)
'''
#Classical Imports
from difflib import SequenceMatcher
#--------------------------------------------------------------------------
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def evaluate_correctness(TopicsFound, ActualTopics, EvalSpec = 2):
    """
    This function is used to evaluate the topics found by our model in comparison to the actual topics. 
    It prints out the Recall, Precision, averageInSpecificity and averageOutSpecificity values.
    -------------------------------------------------------------------------------------------------
    param TopicsFound (list of word lists) : This is the list of topics discovered from your lda/SOM
    model as a list of word lists, where each word list is a topic
    
    param ActualTopics (Dictionary of word lists) : This is a dictionary of word list of the format
    [{topic-H OR -E:[term1, term 2]}, ...{}]. -H and -E must be manually appended to indicate -H (hard) 
    or -E(easy) for a topic based on the criteria in the function getKeywordsofTopics. 
    
    param EvalSpec (integer) : This specifies how many terms of a topic in TopicsFOund need to overlap 
    with the terms of the topic in ActualTopics for a match to be acknowledged. Default is 2.
    """
    # Here we shall try to match the topics found to the actual topics
    #1. Following initailises count of the actual existing hard, easy and overall number of topics.
    existHard = 0
    existEasy = 0
    existActual = 0
    
    Eval_Dic_ActualTopics = dict()
    for actualTopicKey in ActualTopics.keys():
        #2. Counts the various categories of hard, easy and overall topics.
        #actualTopicKey = list(ActualTopic.keys())[0]
        if actualTopicKey.split("-")[1] == "H":
            existHard += 1
        else :
            existEasy += 1
        existActual = existHard+existEasy
        
        #3. Assign each actual known topic at a time to a tester
        tester = ActualTopics[actualTopicKey]
       # EvalSpec1 =EvalSpec
        #if EvalSpec>len(tester): EvalSpec1 = len(tester)
        
        #4. Following initialises a dictionary that will store a word-lists that are considered a match 
        #for current topic t, of the found tipics, under test. Eval_Dic_Topics will be the same length 
        #as the topics found and there will be 1 Eval_Dic_Topics for each actual known topic.
        Eval_Dic_TopicsFound = dict()
       # print("ActualTopic",ActualTopic)
        for t in range(len(TopicsFound)):
            topicKeywords = TopicsFound[t]
            matchedTerms = list()
            for eachTerm in topicKeywords:
                for eachActualterm in tester:
                   # print(eachTerm,eachActualterm)
                    if similar(eachTerm,eachActualterm) >= 0.6:
                        matchedTerms.append(eachTerm)
                        #print(eachTerm,"and",eachActualterm,similar(eachTerm,eachActualterm) )
            Eval_Dic_TopicsFound[t] = list(set(matchedTerms))
                    
        #5. Following is a dictionary of dictionaries. Each key is an actual know topic and the value 
        #is a dictionary of lists of all found topics. If the dictonariy of found topics has a list of 
        #terms that exceeds the Evalspec value then we say there is a match  
        Eval_Dic_ActualTopics[actualTopicKey ] = Eval_Dic_TopicsFound
    
    #print (Eval_Dic_ActualTopics)
#Evaluate(TopicsFound, ActualTopics, 1)
    
    ##### we do the recall and inSpecificity calculations here next#####################
    #print(existActual, existEasy, existHard) 
    Eval_Dic_TopicsFound_Spec = dict() 
    correctlyFoundTopics =[None]
    totalcountActual =0
    totalcountEasy =0
    totalcountHard =0
    
    allInSpeciperTopic =0
    allInSpeciperTopicE =0
    allInSpeciperTopicH =0
    
    for evalTopicKey in Eval_Dic_ActualTopics.keys():
        dictionaryOfTopics = Eval_Dic_ActualTopics[evalTopicKey]
        tocountActual =0
        tocountEasy =0
        tocountHard =0
        
        inSpecidenoCount=0## this gets the number of matched topics we found for that same topic
        inSpecidenoCountE=0
        inSpecidenoCountH=0
        
       # if EvalSpec>
        
        EvalSpec1 =EvalSpec
        if EvalSpec>len(ActualTopics[evalTopicKey]): EvalSpec1 = len(ActualTopics[evalTopicKey])
        for eachFoundTopic in dictionaryOfTopics.keys():
            
            #Eval_Dic_TopicsFound_Spec[eachFoundTopic]= ['test']
            if len(dictionaryOfTopics[eachFoundTopic])>= EvalSpec1:
                inSpecidenoCount+=1
                if(evalTopicKey.split("-")[1]=="H"):
                    inSpecidenoCountH+=1
                else:
                    inSpecidenoCountE+=1
                correctlyFoundTopics.append(evalTopicKey)
                tocountActual =1
                if eachFoundTopic in Eval_Dic_TopicsFound_Spec.keys():
                    #Eval_Dic_TopicsFound_Spec[eachFoundTopic] = \
                    Eval_Dic_TopicsFound_Spec[eachFoundTopic].append(evalTopicKey)
                   # print("First",eachFoundTopic, Eval_Dic_TopicsFound_Spec[eachFoundTopic])
                else:
                    Eval_Dic_TopicsFound_Spec[eachFoundTopic] =[evalTopicKey]
                    #print("second",eachFoundTopic,Eval_Dic_TopicsFound_Spec[eachFoundTopic])
        if inSpecidenoCount >0:
            allInSpeciperTopic+=(1/inSpecidenoCount)  
        if inSpecidenoCountE >0:
            allInSpeciperTopicE+=(1/inSpecidenoCountE)
        if inSpecidenoCountH >0:
            allInSpeciperTopicH+=(1/inSpecidenoCountH)
        
        totalcountActual += tocountActual
       # print(totalcountActual)
        EasyorHard = evalTopicKey.split("-")
        if EasyorHard[1] == "H": 
            totalcountHard +=tocountActual
        else:
            totalcountEasy +=tocountActual
    if totalcountActual > 0:   
        avgInSpeci = allInSpeciperTopic/totalcountActual # finds the average for all teh topics
    else: avgInSpeci= "NaN"
    if totalcountHard > 0:
        avgInSpeciH = allInSpeciperTopicH/totalcountHard # finds the average Hard for all teh topics
    else: avgInSpeciH = "NaN"
    if totalcountEasy >0:
        avgInSpeciE = allInSpeciperTopicE/totalcountEasy # finds the average Easy for all teh topics
    else: avgInSpeciE = "NaN"
    
    
    recallActual = totalcountActual/existActual
    recallEasy =totalcountEasy/existEasy
    recallHard = totalcountHard/existHard
    
    recallValues =[str(recallActual), str(recallEasy), str(recallHard)]
    
    print("RecallActual, RecallEasy, RecallHard",recallValues)
    correctlyFoundTopics.remove(None)
    correctlyFoundTopics = list(set(correctlyFoundTopics))
    print("correctlyFoundTopics",correctlyFoundTopics)
    
    for h in range(len(TopicsFound)):
        if h not in Eval_Dic_TopicsFound_Spec.keys():
            Eval_Dic_TopicsFound_Spec[h] = [None]
    
    ###############Here we calculate precision########################
    
    precisionActual = len(correctlyFoundTopics)/ max(len(correctlyFoundTopics), len(TopicsFound))
    
    print("Precision",precisionActual)
    print("Eval_Dic_TopicsFound_Spec",Eval_Dic_TopicsFound_Spec)
    
    ####################here we calculate specificity
    allOutSpeci=0
    allOutSpeciCount=0
    for eachTopic in Eval_Dic_TopicsFound_Spec.keys():
        #denom =0
        if Eval_Dic_TopicsFound_Spec[eachTopic] != None:
            allOutSpeci+=(1/len(Eval_Dic_TopicsFound_Spec[eachTopic] ))
            allOutSpeciCount+=1
    avgOutSpeci = allOutSpeci/allOutSpeciCount
    print("avgOutSpeci",avgOutSpeci, "avgInSpeci",avgInSpeci,"avgInSpeciE",avgInSpeciE,"avgInSpeciH",avgInSpeciH)
    
    
#Evaluate(TopicsFound, ActualTopics, 2)
#Note AverageOut reflects how many real topics that relate to the found topic higher the better
#Note AverageIn reflects how many found topic relates to a real topic- So we can group to easy and hard
#the higher the better
    
            
                
            
            
            
                    
            
    