# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 22:53:13 2015

@author: Hans-Christiaan
"""

def edit_distance(s1, s2):
    """ Computes the edit distance between two strings.
        Source = http://stackoverflow.com/questions/2460177/edit-distance-in-python
    """
    m=len(s1)+1
    n=len(s2)+1

    tbl = {}
    for i in range(m): tbl[i,0]=i
    for j in range(n): tbl[0,j]=j
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            tbl[i,j] = min(tbl[i, j-1]+1, tbl[i-1, j]+1, tbl[i-1, j-1]+cost)

    return tbl[i,j]

def dic_of_lists_has_value(dic,value):
        for b in dic.itervalues():
            if value in b: return True
        return False

def find_similar_cats(cats, file_name):
    """ Finds for all the categories in the list of strings "cats" the most
        similar categories in terms of edit distance, and saves it to a file
        named "file_name".
        Warning: is of order n^2, so may take a few minutes if the amount of
        categories is large.
    """    
    
    dic = {}    
    # similarity threshold is set somewhat arbitrarily, but produces sufficient
    # results
    for cat in cats:
        #if not dic_of_lists_has_value(dic,cat):
            dic[cat] = [sim for sim in cats if edit_distance(cat,sim) < len(sim)/2]
    
    # write the dictionary to a human readable file
    f = open(file_name,'w')
    for cat in dic.iterkeys():
        f.write(cat.rjust(40) + " : " + str(dic[cat]) + "\n")