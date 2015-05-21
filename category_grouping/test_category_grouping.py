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
FACTORIZED_PATH = "../../train/train_factorized.csv"

def group_factorize_and_save():
    data = ld.load_data()
    
    for var in VARS_TO_GROUP:
        print "finding best split of variable \"", var, "\""
        cg.group_categories(data, var) 
    
    data = ld.factorize_data(data)        
    del data["status_group"]  
    
    data.to_csv(GROUPED_PATH)

def factorize_and_save():
    data = ld.load_data()
    data = ld.factorize_data(data)
    del data["status_group"]
    data.to_csv(FACTORIZED_PATH)
    
def classify(data_path):
    data = ld.load_data(data_path)
    
    y = data["status_group"].tolist()
    
    del data["status_group"]
    del data["date_recorded"]
    
    x = data.as_matrix()
    
    frac_test = 0.2 
    len_test = int(frac_test * len(y))
    indices = np.random.choice(range(0,len(y)), len_test)
    
    test_set = [x[i] for i in indices]
    train_set = [x[i] for i in range(0,len(y)) if i not in indices]
    
    test_y = [y[i] for i in indices]
    train_y = [y[i] for i in range(0,len(y)) if i not in indices]
    
    classifier = ensm.RandomForestClassifier(n_estimators = 100)
    
    classifier.fit(train_set, train_y)
    
    prediction = classifier.predict(test_set)
    
    print "accuracy =", met.accuracy_score(test_y, prediction)
    
    
    