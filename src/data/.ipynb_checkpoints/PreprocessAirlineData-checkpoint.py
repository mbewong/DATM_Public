'''
# -*- coding: utf-8 -*-
# Author: Michael Bewong (michael.bewong@mymail.unisa.edu.au)
#         Lecturer in Data Science, AI and Cyber Security
#         Data Science Research Unit, Charles Sturt University
#         [Website]

# Contributors: John Wondoh (john.wondoh@mymail.unisa.edu.au)
#               Selasi Kwashie (selasi.kwashie@mymail.unisa.edu.au)
'''
#Comments by MIKE: This file s purely designed to help transform the airline dataset. We need to generalize it to be able able to handle any csv dataset. It may not be necessary to include it in the the main package for publication. However, it will be useful for ouw own internal experiments to convert datasets.
#-----------------------------------------------------------------------------------------------
#Classical Imports 
import pandas as pd

#Local Imports
from src.data import DataPreprocessing as dp

#Functions
def preprocessAirline(convertcsv = False, save = False, clean = True, dataSource = None, dataStorage = None):
    ''' 
    This function takes the data either as CSV or dataframe and returns a dataframe with the text
    preprocessed and the columns in the dataframe aligned for subsequent processing.
    ---------------------------------------------------------------------------------------------
    param convertcsv : This param specifies whether the input file is csv and if so converts to 
    dataframe first. Default is False.
    
    param save : This determines whether we save our pandas datafram at each stage.
    
    param clean : This param determines whether we need to preprocess the text in the data or not.
    
    param dataSource : This param determines where the data should loaded from. Default is None. If 
    None, data will be loaded from the same folder where the calling function is located. 
    
    param dataStorage : This param determines where the data should be stored. Default is None. If 
    None, data will be stored in the same folder where the calling function is located. 
    -----------------------------------------------------------------------------------------
    NOTE : This function is defined purely for Airline dataset. In the future we may generalize this
    but we are only in the dev stage atm.
    '''
    #-0- Specify storage and source location
    if dataSource == None:
            source = 'Airline.csv'
    else:
        source =dataSource +'/Airline.csv'
    
    if dataStorage == None:
            storage = 'airline_DF'
    else:
            storage = dataStorage +'/airline_DF'
            
    #-1- Read csv file
    if convertcsv :
        airlineCsv = pd.read_csv(source)
    #-2- Create pandas with relevant columns from csv file
        airline_DF = pd.DataFrame(columns=['tweetID', 'text', 'author', 'publicationDate',
                                           'negativeReason'])
        airline_DF[['tweetID', 'text', 'author', 'publicationDate','negativeReason']] \
        = airlineCsv[['tweet_id','text','name','tweet_created','negativereason']]
        
        if save :
            airline_DF.to_pickle(storage+'.pkl')
    #-3- Clean  text in data
        for index, row in airline_DF.iterrows():
            airline_DF.loc[airline_DF.index.values == index,'text'] = \
            dp.cleanText_udf(airline_DF.loc[index]['text'])
        
        if save :
            airline_DF.to_pickle(storage+'_Clean.pkl')
        #print("Step 1")
    elif not convertcsv and  clean:
        airline_DF = dp.getData(storage+'.pkl')
        
        for index, row in airline_DF.iterrows():
            airline_DF.loc[airline_DF.index.values == index,'text'] = \
            dp.cleanText_udf(airline_DF.loc[index]['text'])
        
        if save :
            airline_DF.to_pickle(storage+'_Clean.pkl')
        #print("Step 2")   
    else:
        airline_DF = dp.getData(storage+'_Clean.pkl')
        #print("Step 3")
    return airline_DF
    
    
        
        