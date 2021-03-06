# -*- coding: utf-8 -*-
"""
Created on Sat May 02 14:10:50 2015

    Handy script to load the data from the 'Pump it up"-challenge into
    a pandas dataframe.

@author: Hans-Christiaan
"""

import pandas as pd

# For easily loading the data in the iPython console.
TRAIN_X_PATH = "../../train/train_x.csv"
TRAIN_Y_PATH = "../../train/train_y.csv"
TEST_PATH = "../../train/test.csv"

CATEGORICALS = ['funder','wpt_name','basin','subvillage','region','lga',
                  'ward','recorded_by','scheme_management','scheme_name',
                  'payment_type','payment','quality_group','management_group',
                  'management','water_quality','quality_group','quantity',
                  'quantity_group','source','source_type','source_class',
                  'waterpoint_type','waterpoint_type_group','status_group',
                  'extraction_type','extraction_type_group','installer',
                  'extraction_type_class','permit','public_meeting']            

def load_data(train_x_path=TRAIN_X_PATH, train_y_path=TRAIN_Y_PATH):
    """ Loads the training data in a pandas dataframe.
        The y-values are added to the dataframe as a seperate column
        (called 'status_group'). 
    """
    print "Loading data..."
    # Load the data and merge them into a single dataframe
    train_x = pd.read_csv(train_x_path)
    train_y = pd.read_csv(train_y_path)
    train = pd.merge(train_x, train_y, on='id')
    
    # Set the categorical variables to pandas special 'category' data type.   
    for train_cat in CATEGORICALS:
        train[train_cat] = train[train_cat].astype('category')
    
    #print "features found:"
    #for col in train.columns: print "  -", col     
    print "Data loaded!"    
    
    return train

def load_data_test(test_path=TEST_PATH):
    print "Loading test set..."
    
    test_data = pd.read_csv(test_path)
    for train_cat in CATEGORICALS:
        if train_cat != 'status_group':
            test_data[train_cat] = test_data[train_cat].astype('category')
    print "Data loaded!"
    return test_data
    
def factorize_data(data):
    """ Factorizes the categorical variables in the pandas data frame 'data':
        Every ocurring category in a variable is mapped to a positive natural
        number.        
        E.g. for 'status_group' the categories become:
            - functional -> 0
            - functional_needs_repair -> 1
            - non_functional -> 2
    """
    for cat in CATEGORICALS:
        data[cat] = pd.factorize(data[cat], sort = True)[0]    
    return data
    
def factorize_data_testset(data):
    """ Factorizes the categorical variables in the pandas data frame 'data':
        Every ocurring category in a variable is mapped to a positive natural
        number.        
        E.g. for 'status_group' the categories become:
            - functional -> 0
            - functional_needs_repair -> 1
            - non_functional -> 2
    """
    for cat in CATEGORICALS:
        if cat != 'status_group':
            data[cat] = pd.factorize(data[cat], sort = True)[0]    
    return data