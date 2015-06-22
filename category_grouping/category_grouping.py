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

def get_distr(data, y_name):
    """ Gets the distribution of the classes in the column with the name
        'y_name' of the pandas dataframe 'data'.
        E.g. when 40% of the rows are of class 1, 30% are of class 2 and
        30% are of class 3, [0.4, 0.3, 0.3] is returned.
    """
    y_cats = data[y_name].cat.categories
    probs = []
    len_tot = len(data)
    
    for y_cat in y_cats:
        y_len = len(data[data[y_name] == y_cat])+1        
        probs.append( y_len / float(len_tot+3))
        
    return probs

def group_categories(data, column_name, y_name="status_group"):
    best_grouping = find_best_grouping(data, column_name, y_name)
    print "\nbest grouping for variable", column_name, "=\n", best_grouping
    for group in best_grouping:
        if isinstance(group, list):
            new_cat = group[0]
            for cat in group:                
                data[column_name].replace(to_replace = cat, value = new_cat, inplace=True)
               
    data[column_name] = data[column_name].astype('category')
    
    new_names = []
    for old_name in data[column_name].cat.categories:
        for group in best_grouping:
            if old_name in group:
                new_name = str(group)
                new_names = new_names + [new_name]    
    
    data[column_name] = data[column_name].cat.rename_categories(new_names)
           
    return data        
    
       
def find_best_grouping(data, column_name, y_name="status_group", prev_entr=2.0):
    """ Tries to find the best split between categories in a categorical
        variable.    
    """

    # Remove unused categories    
    col = data[column_name].cat.remove_unused_categories()    
    cats = col.cat.categories
    
    if len(cats) == 1:
        return [cats[0]]
        
    min_entropy = 100
    min_split_left = None
    min_split_right = None
    min_entropy_left = 2
    min_entropy_right = 2
    min_split_left_names = ""
    min_split_right_names = ""
    
    c = 1
    tot = len(cats)
    # find all possible two-way splits (1-4, 2-3 etc.)     
    for i in range(1,int(len(cats)/2)+1):
        # Find all combinations of categories of length i
        combs = it.combinations(cats,i)       
        for comb in combs:            
            # divide the categories in the ones that are in this combination
            # and the ones that are not.
            cats_left = np.array(comb)
            cats_right = np.setdiff1d(cats, comb)
            # split the data            
            split_left = data[data[column_name].isin(cats_left)]
            split_right = data[data[column_name].isin(cats_right)]
            
            entropy_left = st.entropy(get_distr(split_left, y_name))
            entropy_right = st.entropy(get_distr(split_right, y_name))
            # weight the entropy based on the amount of categories in the split.
            w_l = len(cats_left) / float(len(cats))
            w_r = len(cats_right) / float(len(cats))            
            
            entropy_total = w_l * entropy_left + w_r * entropy_right
            print "\rtested split", c, "(", i, "-", tot-i, "split) [", entropy_total, ">", min_entropy, "]",             
            # save the split with the minimal entropy
            if entropy_total < min_entropy:
                min_entropy = entropy_total
                min_split_left = split_left
                min_split_right = split_right
                min_entropy_left = entropy_left
                min_entropy_right = entropy_right
                min_split_left_names = list(cats_left)
                min_split_right_names = list(cats_right)
            c = c+1
    
    if min_entropy > prev_entr:        
        print "\n", list(cats), "weighted entropy =", prev_entr
        return [list(cats)]
        
    #print "\n[", min_entropy_left, "]", min_split_left_names
    #print "[", min_entropy_right, "]", min_split_right_names
    #print "weighted entropy = ", min_entropy, "\n"
        
    left = find_best_grouping(min_split_left, column_name, y_name, min_entropy_left)
    right = find_best_grouping(min_split_right, column_name, y_name, min_entropy_right)
    
    return left + right
    

            
