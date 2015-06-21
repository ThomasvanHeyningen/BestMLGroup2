# -*- coding: utf-8 -*-
"""
Created on Wed May 06 14:10:37 2015

@author: Hans-Christiaan
"""

from mrmr import load_data as ld


def run():
    train = ld.load_data(ld.TRAIN_X_PATH, ld.TRAIN_Y_PATH)
    train = ld.factorize_data(train)
    
    