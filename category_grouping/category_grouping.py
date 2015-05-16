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

def get_distr(data, column_name, y_name):
    
    y_cats = data[y_name].cat.categories
    probs = []
    len_tot = len(data)
    
    for y_cat in y_cats:        
        probs.append(len(data[data[y_name] == y_cat])+1 / float(len_tot+3))
        
    return probs
    
def group_categories(data, column_name, y_name):
    
    if data is None:
        return None
        
    col = data[column_name].cat.remove_unused_categories()    
    cats = col.cat.categories
    
    if len(cats) <= 2:
        return None
        
    min_entropy = 100
    min_split_left = None
    min_split_right = None
    min_split_left_names = ""
    min_split_right_names = ""
    
    c = 1
    tot = len(cats)    
    for i in range(1,int(len(cats)/2)+1):       
        combs = it.combinations(cats,i)       
        for comb in combs:
            print "\rtesting split", c, "(", i, "-", tot-i, "split)",
            
            cats_left = np.array(comb)
            cats_right = np.setdiff1d(cats, comb)
                        
            split_left = data[data[column_name].isin(cats_left)]
            split_right = data[data[column_name].isin(cats_right)]
            
            entropy_left = st.entropy(get_distr(split_left, column_name, y_name))
            entropy_right = st.entropy(get_distr(split_right, column_name, y_name))
            
            w_l = len(cats_left) / float(len(cats))
            w_r = len(cats_right) / float(len(cats))            
            
            entropy_total = w_l * entropy_left + w_r * entropy_right             
            
            if entropy_total < min_entropy:
                min_entropy = entropy_total
                min_split_left = split_left
                min_split_right = split_right
                min_split_left_names = list(cats_left)
                min_split_right_names = list(cats_right)
            c = c+1
        
    print "\n[", entropy_left, "]", min_split_left_names
    print "[", entropy_right, "]", min_split_right_names
    print "weighted entropy = ", entropy_total, "\n"
    
    group_categories(min_split_left, column_name, y_name)
    group_categories(min_split_right, column_name, y_name)
            
