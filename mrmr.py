# -*- coding: utf-8 -*-
"""
Created on Sat May 02 14:06:46 2015

Minimum-redundancy-maximum-relevance feature selection.

@author: Hans-Christiaan
"""

import load_data as ld

def mRMR(data, n, class_vector_name = 'status_group'):
    """ Minimum-redundancy-maximum-relevance
        feature selection.
        
        Correlation is used as the information measure.
        
        Uses a greedy approach. It picks the feature with the
        highest correlation with the class vector first, and
        subsequently picks the features with the highest correlation
        with the class vector, and lowest correlations with the already 
        selected features.        
    """
    
    # using correlation as the measure
    corr = data.corr()
    corr = corr.abs()
    # correlation of the features with the class vector
    cor_y = corr[class_vector_name].copy()
    # select the feature with the highest correlation with the class vector
    # as the first one
    cor_y.sort(ascending = False)   
    sel_feats = [cor_y.index[1]]    
    # select n-1 other features
    for i in range(0,n-1):
        scores = []
        for col in corr.columns:
            if col not in sel_feats + [class_vector_name]:
                # compute the mean correlation with the already selected
                # features
                cor_x = sum([corr[col][sel_feat] for sel_feat in sel_feats])                        
                cor_x = cor_x / len(sel_feats)             
                       
                scores = scores + [(cor_y[col] - cor_x, col)]
        # add the feature with the highest score to the list of selected
        # selected features
        scores = sorted(scores, reverse=True)
        sel_feats = sel_feats + [scores[0][1]]
        
    return sel_feats
    
def load_and_setup_data():
    N_FEATURES = 10
        
    data = ld.load_data(ld.TRAIN_X_PATH, ld.TRAIN_Y_PATH)
    data = ld.factorize_data(data)
    
    del data['id']
    del data['recorded_by']
    del data['num_private']
    
    return mRMR(data, N_FEATURES)