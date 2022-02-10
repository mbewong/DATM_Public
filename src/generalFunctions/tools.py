def getData_2(source):
    "This gets the data raw or processed"
    return(pd.read_pickle(source))

def cleanText_udf(text):
    """
    Removes 'bad' characters from a text. Text is required. 
    """
    stopwordsGSR = ["amp"]
    stop2 = re.compile(r'\b(' + r'|'.join(stopwordsGSR) + r')\b\s*')
    stop = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    stop2 = re.compile(r'\b(' + r'|'.join(stopwordsGSR) + r')\b\s*')
    tmp = stop.sub('', problemchars.sub('', emojis.sub('', url_finder.sub('', username.sub('', stop2.sub('', text.lower().strip()))))))
    tmp = lemmaWordsinString(tmp)
    #tmp = stop2.sub('')
    return tmp

def cleanText_udf_new(text):
    tmp = url_finder.sub('', username.sub('', text.lower().strip()))
    tmp = lemmaWordsinString(tmp)
    return tmp

def readCSVtoPKL(source):
    import pandas as pd
    return(pd.read_csv(source))

def changeDate(): # seems necessary for the airline data
    thedates=list(indo_EngTweetsPd["tweet_created"])
    splitDate= thedates[0].split('/')
    print(splitDate)
    splitYear = '20'+splitDate[2].split(' ')[0]
    print(splitYear)
    dateTuple = date(int(splitYear),int(splitDate[0]),int(splitDate[1]))
    print(dateTuple)
    if True:
        thedates=list(indo_EngTweetsPd["tweet_created"])
        newDateList =list()
        #print(topicList)
        #SplitDates = [thedates[aDate] for aDate in thedates ]
        for splitDate1 in thedates:
            splitDate= splitDate1.split('/')
            splitYear = '20'+splitDate[2].split(' ')[0]
            dateTuple = date(int(splitYear),int(splitDate[0]),int(splitDate[1]))
            newDateList.append(dateTuple)
       # print(newDateList)
        #existingTopics = set(newTopicList)
        #indo_EngTweetsPd['negativereason'] = newTopicList
    source = 'Airline.pkl'
    indo_EngTweetsPd1 =getData_2(source)
    indo_EngTweetsPd1['publicationDate'] =newDateList 
    indo_EngTweetsPd1
    
def saveAsPKL():
    indo_EngTweetsPd1.to_pickle("Airline.pkl")
    
def changeColumnName():
    indo_EngTweetsPd=indo_EngTweetsPd.rename(columns={'publicationDate':'testpublicationDate'}, inplace = True)
    indo_EngTweetsPd.to_pickle("Airline.pkl")
    
def addSentiment():
    indo_EngTweetsPd =getData_2("McDonalds.pkl")
    indo_EngTweetsPd_sent = sentiCalc(indo_EngTweetsPd) 

    indo_EngTweetsPd_sent.to_pickle("McDonalds.pkl")
    
def sentiCalc(dataframe):
    "This function returns a new dataframe of tweets from the old with the added column sentiment"
    indo_EngsentEnd = dataframe.count()[1] # this just gets the number of tweets there are
    #print(indo_EngsentEnd)
    indo_EngTweetsPd_sent = dataframe.copy()
    indo_EngTweetsPd_sent["sentiment"] = ""
    
    #indo_Engsentences = [dataframe.iloc[x][2] for x in range(0,indo_EngsentEnd)]       
    #indo_EngsentimentValues =[]
    sid = SentimentIntensityAnalyzer()
    labels = [indo_EngTweetsPd_sent.index[c] for c in range(indo_EngsentEnd)]
    for f in labels:
        tense = cleanText_udf_new(indo_EngTweetsPd_sent.loc[f].text)
        tense = translateInd1(tense)
        ss = sid.polarity_scores(tense) # this returns a dictionary with 4 keys, compound (an overall combination), negative, neutral and postive values
        indo_EngTweetsPd_sent.loc[f,'sentiment'] =ss["compound"]
        indo_EngTweetsPd_sent.loc[f,'text'] = tense

    #indo_EngTweetsPd_sent = dataframe.copy()
    #indo_EngTweetsPd_sent["sentiment"] = Series(indo_EngsentimentValues)
    return(indo_EngTweetsPd_sent)