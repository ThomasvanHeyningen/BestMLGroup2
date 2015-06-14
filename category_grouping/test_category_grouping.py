# -*- coding: utf-8 -*-
"""
Created on Thu May 21 18:11:32 2015

@author: Hans-Christiaan
"""

import pandas as pd
import numpy as np
import load_data as ld
import category_grouping as cg

import sklearn.ensemble as ensm
import sklearn.metrics as met

VARS_TO_GROUP = ["source", "source_type", "source_class", "payment", 
                 "payment_type", "quality_group", "quantity", "waterpoint_type",
                 "waterpoint_type_group", "basin"]
                 
GROUPED_PATH = "../../train/train_grouped.csv"
GROUPED_PATH_NAMED = "../../train/train_grouped_named.csv"
FACTORIZED_PATH = "../../train/train_factorized.csv"

def group_factorize_and_save():
    """ 1. Loads the original data.
        2. Finds the best splits of all categorical variables.
        3. Factorizes the resulting dataframe 
        4. Saves it as a CSV-file.
    """
    data = ld.load_data()
    
    for var in VARS_TO_GROUP:
        print "\nfinding best split of variable \"", var, "\""
        data = cg.group_categories(data, var) 
    
    #data = ld.factorize_data(data)        
    del data["status_group"]  
    
    data.to_csv(GROUPED_PATH_NAMED)    

def factorize_and_save():
    """ 1. Loads the original data.
        2. Factorizes the resulting dataframe
        3. Saves it as a CSV-file.
    """ 
    data = ld.load_data()
    data = ld.factorize_data(data)
    del data["status_group"]
    data.to_csv(FACTORIZED_PATH)
    
def classify_and_compare(data_path_1, data_path_2):
    data_1 = ld.load_data(data_path_1)
    data_2 = ld.load_data(data_path_2)    
    
    y = data_1["status_group"].tolist()
    
    del data_1["status_group"]
    del data_1["date_recorded"]
    del data_2["status_group"]
    del data_2["date_recorded"]
    
    x_1 = data_1.as_matrix()
    x_2 = data_2.as_matrix()
    
    frac_test = 0.2 
    len_test = int(frac_test * len(y))
    indices = np.random.choice(range(0,len(y)), len_test)
    
    test_set_1 = [x_1[i] for i in indices]
    train_set_1 = [x_1[i] for i in range(0,len(y)) if i not in indices]
    
    test_y = [y[i] for i in indices]
    train_y = [y[i] for i in range(0,len(y)) if i not in indices]
    
    test_set_2 = [x_2[i] for i in indices]
    train_set_2 = [x_2[i] for i in range(0,len(y)) if i not in indices]
   
    classifier_1 = ensm.RandomForestClassifier(n_estimators = 100)
    classifier_2 = ensm.RandomForestClassifier(n_estimators = 100)
    
    classifier_1.fit(train_set_1, train_y)
    classifier_2.fit(train_set_2, train_y)
    
    prediction_1 = classifier_1.predict(test_set_1)
    prediction_2 = classifier_2.predict(test_set_2)
    
    print "accuracy classifier 1 =", met.accuracy_score(test_y, prediction_1)
    print "accuracy classifier 2 =", met.accuracy_score(test_y, prediction_2)
    
    
    