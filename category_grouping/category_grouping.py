# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:41:31 2015

@author: Hans-Christiaan
"""

import pandas as pd
import numpy as np
import load_data as ld
import itertools as it
import scipy.stats as st

def get_probs(group, data, column_name, y_name):
    sl = data[data[column_name].isin(group)]
    y_cats = data[y_name].cat.categories
    probs = []
    len_tot = len(sl)
    
    for y_cat in y_cats:        
        probs.append(len(sl[sl[y_name] == y_cat]) / float(len_tot))
        
    return probs

def entropy(groups, data, column_name, y_name):
    total_entropy = 0
    for group in groups:
        probs = get_probs(group, data, column_name, y_name)
        total_entropy = total_entropy + st.entropy(probs)
    return total_entropy    

def group_categories(data, column_name, y_name):
    cats = data[column_name].cat.categories
    min_entropy = 100
    min_split = []
    
    for i in range(1,len(cats)-1):
        combs = it.combinations(cats,i)
        for comb in combs:
            groups = []
            groups.append(comb)
            groups.append(np.setdiff1d(cats, comb))
            entr = entropy(groups, data, column_name, y_name)
            if entr < min_entropy:
                min_entropy = entr
                min_split = groups
    return min_split, min_entropy
            
