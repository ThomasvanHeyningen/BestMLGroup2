# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:09:01 2015

@author: Hans-Christiaan
"""
# used for table manipulation
import pandas as pd
# used for plotting
import pylab as pl
import numpy as np

from collections import Counter
from operator import add

# For easily loading the data in the iPython console.
TRAIN_X_PATH = "../../train/train_x.csv"
TRAIN_Y_PATH = "../../train/train_y.csv"

TRAIN_X_GROUPED_PATH = "../../train/train_grouped_named.csv"

CATEGORICALS = ['funder','wpt_name','basin','subvillage','region','lga',
                  'ward','recorded_by','scheme_management','scheme_name',
                  'payment_type','payment','quality_group','management_group',
                  'management','water_quality','quality_group','quantity',
                  'quantity_group','source','source_type','source_class',
                  'waterpoint_type','waterpoint_type_group','status_group',
                  'extraction_type','extraction_type_group','installer',
                  'extraction_type_class']

def load_data(train_x_path, train_y_path):
    """ Loads the training data in a pandas dataframe.
        The y-values are added to the dataframe as a seperate column
        (called 'status_group'). 
    """
    # Load the data and merge them into a single dataframe
    train_x = pd.read_csv(train_x_path)
    train_y = pd.read_csv(train_y_path)
    train = pd.merge(train_x, train_y, on='id')
    
    # Set the categorical variables to pandas special 'category' data type.   
    for train_cat in CATEGORICALS:
        train[train_cat] = train[train_cat].astype('category')
    
    print "features found:"
    for col in train.columns: print "  -", col     
   
    
    return train

def plot_bar_stacked(train, column_name, size=(8.0,8.0)):
    """ Plots the frequencies (amount of functional, non functional pumps
        and pumps that need repair) of the categories in a column as a
        horizontal bar plot.
        -- train = pandas dataframe with at least a column 'column_name' and
                    a column 'status_group'
        -- column_name = name (string) of the column
        -- size = [optional] set the size (x,y)-tuple of the plot in inches
    """
    cpy = train.copy()
    # get a list of all the categories 
    labels = train[column_name].cat.categories
    
    # seperate the values in the column by their pumps' status and count the
    # frequency of the categories.
    func =      Counter(cpy[column_name][cpy.status_group=='functional'])
    non_func =  Counter(cpy[column_name][cpy.status_group=='non functional'])
    repair =    Counter(cpy[column_name][cpy.status_group=='functional needs repair'])
    
    func_values = [func[label] for label in labels]
    non_func_values = [non_func[label] for label in labels]
    repair_values = [repair[label] for label in labels] 
    
    # explicitly set the counts of the categories not in a status group to 0
    # for the correct aligning of bars to work
    for label in labels:
        if func[label] is 0:
            func[label] = 0
        if non_func[label] is 0:
            non_func[label] = 0
        if repair[label] is 0:
            repair[label] = 0        
    # list of x-values used to correctly align the bars.
    left_non_func = map(add, func_values, repair_values)
    
    for label in labels:
        print label, non_func[label] + func[label] + repair[label]
    
    # plot the bars
    pl.barh(np.arange(len(labels)), func_values,\
                        color = 'green', align='center')    
    pl.barh(np.arange(len(labels)), repair_values,\
                        color = 'darkorange', align='center', left=func_values)
    pl.barh(np.arange(len(labels)), non_func_values,\
                        color = 'red', align='center', left=left_non_func)
    # set the plot's title                    
    ax = pl.gca()
    ax.set_title(column_name)
        
    pl.ylim([-1,len(labels)])              
    # annotate the bars with their category names
    text_x = max(func_values)/10    
    for i in range(0,len(func_values)):
        ax.annotate(labels[i], (text_x,i), size='large', verticalalignment='center')
    
    fig = pl.figure(num=1)    
    fig.set_size_inches(size[0], size[1])    
    
    pl.show()

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
        data[cat] = pd.factorize(data[cat])[0]    
    return data


def correlation_matrix(data, size=8.0):
    """ Calculates and shows the correlation matrix of the pandas data frame
        'data' as a heat map.
        Only the correlations between numerical variables are calculated!
    """
    # calculate the correlation matrix
    corr = data.corr()
    #print corr
    lc = len(corr.columns)
    # set some settings for plottin'
    pl.pcolor(corr, vmin = -1, vmax = 1, edgecolor = "black")
    pl.colorbar()
    pl.xlim([-5,lc])
    pl.ylim([0,lc+5])
    pl.axis('off')
    # anotate the rows and columns with their corresponding variables
    ax = pl.gca()            
    for i in range(0,lc):
        ax.annotate(corr.columns[i], (-0.5, i+0.5), \
            size='large', horizontalalignment='right', verticalalignment='center')
        ax.annotate(corr.columns[i], (i+0.5, lc+0.5),\
            size='large', rotation='vertical',\
            horizontalalignment='center', verticalalignment='right')
    # change the size of the image
    fig = pl.figure(num=1)    
    fig.set_size_inches(size+(size/4), size)     
    
    pl.show()

def plot_lat_long(train):
    """ Plots the 'latitude' and 'longitude' columns of the training data 'train'
        as a scatter plot, ultimating in a map of tanzania showing where
        all the pumps are located and if they work or not.
    """
    colors = ['r','darkorange','g']
    
    cpy = train.copy()
    cpy = cpy[cpy.longitude != 0]    
    
    lat_func = cpy.latitude[cpy.status_group == 'functional']
    lat_non_func = cpy.latitude[cpy.status_group == 'non functional']
    lat_repair = cpy.latitude[cpy.status_group == 'functional needs repair']
    
    long_func = cpy.longitude[cpy.status_group == 'functional']
    long_non_func = cpy.longitude[cpy.status_group == 'non functional']
    long_repair = cpy.longitude[cpy.status_group == 'functional needs repair']    
    # comment one or more of these lines out to remove their respective
    # status from the plot.
    # (They tend to overlap, making the combined plot hard to read)
    pl.scatter(long_non_func, lat_non_func, c=colors[0])    
    pl.scatter(long_repair, lat_repair, c=colors[1])
    pl.scatter(long_func, lat_func, c=colors[2])    
    
    fig = pl.figure(num=1)
    fig.set_size_inches(14.0, 14.0)
    #fig.savefig('map_needs_repair.png',dpi=100)    
    
    pl.show()
    


    