import os
import platform
import time
import re
import numpy as np
import pandas as pd
import datetime as dt
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

# Variables used throughout the file
data_dir='../data/'

# Variables used throughout the file
data_dir='../data/'

def frequency_feature(data_dir,traindata,testdata):
    names_parameters=['installer','basin','region','lga','ward','scheme_name']

def closepumps(data_dir,traindata,testdata, trainlabels, k=5):
    '''
    This one does NOT WORK!!!
    '''
    names_parameters=['longitude', 'latitude']
    (testrows, testcolumns)=testdata.shape
    (trainrows, traincolumns)=traindata.shape
    traindata= traindata.fillna(-1)
    testdata = testdata.fillna(-1)
    train = np.column_stack((traindata['longitude'], traindata['latitude']))
    test = np.column_stack((testdata['longitude'], testdata['latitude']))

    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(train)
    distances, indices = nbrs.kneighbors(train)
    trainfunctionalarray = []
    trainrepairarray =[]
    trainbrokenarray =[]
    for i in xrange(0, len(indices)):
        functional = 0
        repair=0
        broken=0
        for pump in indices[i][1:]:
            pumplabel = trainlabels.loc[pump]['status_group']
            if(pumplabel=='functional needs repair'):
                repair+=1
            elif(pumplabel=='functional'):
                functional+=1
            else:
                broken+=1

        trainfunctionalarray.append(functional)
        trainrepairarray.append(repair)
        trainbrokenarray.append(broken)

    testdistances, testindices = nbrs.kneighbors(test)
    testfunctionalarray = []
    testrepairarray =[]
    testbrokenarray =[]

    for i in xrange(0, len(testindices)):
        functional = 0
        repair=0
        broken=0
        for pump in testindices[i][:-1]:
            pumplabel = trainlabels.loc[pump]['status_group']
            if(pumplabel=='functional needs repair'):
                repair+=1
            elif(pumplabel=='functional'):
                functional+=1
            else:
                broken+=1

        testfunctionalarray.append(functional)
        testrepairarray.append(repair)
        testbrokenarray.append(broken)
    newtestdata=np.zeros((testrows,1), dtype=int)
    newtestdata[:, 0]=np.array(testfunctionalarray)
    store_data(newtestdata, train=False,labels=(str(k) + '_nearest_functional'), one=True)
    newtestdata=np.zeros((testrows,1), dtype=int)
    newtestdata[:, 0]=np.array(testrepairarray)
    store_data(newtestdata, train=False,labels=(str(k) + '_nearest_need_repair'), one=True)
    newtestdata=np.zeros((testrows,1), dtype=int)
    newtestdata[:, 0]=np.array(testbrokenarray)
    store_data(newtestdata, train=False,labels=(str(k) + '_nearest_broken'), one=True)
    newtraindata=np.zeros((trainrows,1), dtype=int)
    newtraindata[:, 0]=np.array(trainfunctionalarray)
    store_data(newtraindata, train=True,labels=(str(k) + '_nearest_functional'), one=True)
    newtraindata=np.zeros((trainrows,1), dtype=int)
    newtraindata[:, 0]=np.array(trainrepairarray)
    store_data(newtraindata, train=True,labels=(str(k) + '_nearest_need_repair'), one=True)
    newtraindata=np.zeros((trainrows,1), dtype=int)
    newtraindata[:, 0]=np.array(trainbrokenarray)
    store_data(newtraindata, train=True,labels=(str(k) + '_nearest_broken'), one=True)
    encode_categorical(data_dir,traindata,testdata, ['funder_clean', 'installer_clean'])
    frequency_feature(data_dir,traindata,testdata, ['funder_clean', 'installer_clean'])


def clean_text(data_dir,traindata,testdata):
    '''
    After cleaning the funder and installer functions by hand (over 30% was cleaned) this function is used to derive some of the features.
    The data given is:
    data_dir: not used
    traindata: the new traindata file where funder and installer data are cleaned
    testdata: the new testdata file where funder and installer data are cleaned
    after reading all the data into data frames the function calls the encode categorical and the frequency feature
    encode categorical encodes the textual values to numeric values (unordered).
    frequency feature calculates the frequency of each data point.
    '''
    names_parameters=['funder', 'installer'] # the parameters to work with
    (testrows, testcolumns)=testdata.shape # data size of train set
    (trainrows, traincolumns)=traindata.shape # data size of test set

    traindata= traindata.fillna(-1) # fill all empty values with -1 (easy fix for trouble later on)
    testdata = testdata.fillna(-1)
    for feature in names_parameters:
        train_text = []
        test_text = []
        #for each value in traindata we do some automatic cleany (mostly redundant and from an earlier automatic quick fix):
        for text in traindata[feature]:
            if text is -1: #all missing values as -1
                train_text.append(-1)
            elif text is '0': #all missing values as -1: 0's likely represent a missing value for these features.
                train_text.append(-1)
            else:
                train_text.append(text.lower().replace(" ", "")) #lowercase and remove spaces for quick unification of data.
        #for each value in testdata we do some the same cleanup:
        for text in testdata[feature]:
            if text is -1:
                test_text.append(-1)#all missing values as -1
            elif text is '0':
                test_text.append(-1) #all missing values as -1: 0's likely represent a missing value for these features.
            else:
                test_text.append(text.lower().replace(" ", "")) #lowercase and remove spaces for quick unification of data.
        #we add the new features to the training and test data:
        traindata[feature +'_clean2'] = pd.Series(train_text, index=traindata.index)
        testdata[feature +'_clean2'] = pd.Series(test_text, index=testdata.index)
    #We instruct the categorical and frequency data to encode the newly created clean data
    # (those functions will also store it into the extratrainfeatures.csv file):
    encode_categorical(data_dir,traindata,testdata, ['funder_clean2', 'installer_clean2'])
    frequency_feature(data_dir,traindata,testdata, ['funder_clean2', 'installer_clean2'])


def frequency_feature(data_dir,traindata,testdata, parameters=None):
    if parameters == None:
        names_parameters=['funder', 'installer','basin','region','lga','ward','scheme_name']
    else:
        names_parameters=parameters
    (testrows, testcolumns)=testdata.shape
    (trainrows, traincolumns)=traindata.shape
    traindata=traindata.fillna(-1)
    testdata = testdata.fillna(-1)
    for feature in names_parameters:
        countdata=pd.concat([traindata[feature],testdata[feature]], axis=0)
        unique, counts = np.unique(countdata, return_counts=True)
        #print len(unique)
        #for i in range(0,len(unique)):
        #    print str(unique[i]) + ' <- name  -- count -> ' + str(counts[i])
        train_frq = []
        test_frq = []
        for freq in traindata[feature]:
            if freq is -1:
                train_frq.append(-1)
            elif freq is '0':
                train_frq.append(-1)
            else:
                train_frq.append(counts[np.where(unique==freq)[0][0]])
        for freq in testdata[feature]:
            if freq is -1:
                test_frq.append(-1)
            elif freq is '0':
                test_frq.append(-1)
            else:
                test_frq.append(counts[np.where(unique==freq)[0][0]])

        newtestdata=np.zeros((testrows,1), dtype=int)
        newtestdata[:, 0]=np.array(test_frq)
        store_data(newtestdata, train=False,labels=(feature +'_freq'), one=True)
        newtraindata=np.zeros((trainrows,1), dtype=int)
        newtraindata[:, 0]=np.array(train_frq)
        store_data(newtraindata, train=True,labels=(feature +'_freq'), one=True)


def encode_categorical(data_dir,traindata,testdata, parameters=None):
    if parameters == None:
        names_parameters=['funder','installer','wpt_name','basin','subvillage','region','lga','ward','public_meeting','recorded_by'
        ,'scheme_management','scheme_name','permit','extraction_type','extraction_type_group','extraction_type_class','management','management_group','payment',
                        'payment_type','water_quality','quality_group','quantity','quantity_group','source','source_type','source_class',
                       'waterpoint_type','waterpoint_type_group']
    else:
        names_parameters=parameters
    (testrows, testcolumns)=testdata.shape
    (trainrows, traincolumns)=traindata.shape
    traindata[names_parameters] = traindata[names_parameters].fillna('nan')
    testdata [names_parameters] = testdata [names_parameters].fillna('nan')
    for feature in names_parameters:
        le = LabelEncoder()
        fitdata = np.append(traindata[feature].values,testdata[feature].values)
        le.fit(fitdata)
        print feature
        train_cat = le.transform(traindata[feature])
        test_cat  = le.transform(testdata[feature])
        newtestdata=np.zeros((testrows,1), dtype=int)
        newtestdata[:, 0]=np.array(test_cat)
        store_data(newtestdata, train=False,labels=(feature +'_num'), one=True)
        newtraindata=np.zeros((trainrows,1), dtype=int)
        newtraindata[:, 0]=np.array(train_cat)
        store_data(newtraindata, train=True,labels=(feature +'_num'), one=True)

def date_features(data):
    (rows, columns)=data.shape
    years = []
    months =[]
    days = []
    age = []
    distance = []
    for date, construction in zip(data['date_recorded'], data['construction_year']):
        match=re.match('([0-9]{4})\-([0-9]{2})\-([0-9]{2})',date)
        years.append(match.group(1))
        months.append(match.group(2))
        days.append(match.group(3))
        recorddate=dt.date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        age.append(recorddate.year-construction)
        begin2014 = dt.date(2014, 01, 01)
        dist = abs(begin2014 - recorddate)
        distance.append(dist.days)
    newdata=np.zeros((rows,5), dtype=int)
    newdata[:, 0]=np.array(years)
    newdata[:, 1]=np.array(months)
    newdata[:, 2]=np.array(days)
    newdata[:, 3]=np.array(age)
    newdata[:, 4]=np.array(distance)
    return(newdata)

def store_data(newdata, ids=None, data_dir=data_dir, train=True, labels=(''), one=False):
    if platform.platform() == "Windows":
        data_dir='..\\data\\'
    else:
        data_dir=os.path.dirname(os.path.abspath('')) + '/data/'

    if train:
        if os.path.exists(data_dir + 'extratrainfeatures.csv'):
            trainfile = pd.read_csv(data_dir + 'extratrainfeatures.csv')
            #trainfile.set_index('id')
        else:
            trainfile=pd.DataFrame(data=ids)  # only needed to generate the file the first time
        index=0
        if one:
            trainfile[labels]=newdata[:,0]
        else:
            for label in labels:
                trainfile[label]=newdata[:,index]
                index=index+1
        trainfile.to_csv(data_dir + 'extratrainfeatures.csv', index_label='id', index= False)
    else:
        if os.path.exists(data_dir + 'extratestfeatures.csv'):
            testfile = pd.read_csv(data_dir + 'extratestfeatures.csv')
            testfile = pd.read_csv(data_dir + 'extratestfeatures.csv')
            #testfile.set_index('id')
        else:
            testfile=pd.DataFrame(data=ids) # only needed to generate the file the first time
        index=0
        if one:
            testfile[labels]=newdata[:,0]
        else:
            for label in labels:
                testfile[label]=newdata[:,index]
                index=index+1
        testfile.to_csv(data_dir + 'extratestfeatures.csv', index_label='id', index=False)

def main():
    print(" - Start.")
    if platform.platform() == "Windows":
        data_dir='..\\data\\'
    else:
        data_dir=os.path.dirname(os.path.abspath('')) + '/data/'

    train = pd.read_csv(data_dir + 'trainset.csv')
    trainlabels = pd.read_csv(data_dir + 'trainlabels.csv')
    test = pd.read_csv(data_dir + 'testset.csv')
    try:
        newtrain = pd.read_csv(data_dir + 'nieuwtrain.csv')
        newtest = pd.read_csv(data_dir + 'nieuwtest.csv')
    except ValueError:
        pass
    #Clean up textual data to prevent duplicates with different names by removing spaces and uppercase
    clean_text(data_dir,newtrain,newtest)

    #to make the categorical features numeric:
    encode_categorical(data_dir,train,test)
    #feature on distance to other pumps:
    #closepumps(data_dir,train,test, trainlabels, k=5) # ran it with k=5,10,20 and 40

    #to create the datelabels
    newtrain = date_features(train)
    newtest = date_features(test)
    store_data(newtrain, train['id'], data_dir, train=True,labels=('year_recorded','month_recorded','day_recorded','age_of_pump','date_recorded_distance_days_20140101'))
    store_data(newtest, test['id'], data_dir, train=False,labels=('year_recorded','month_recorded','day_recorded','age_of_pump','date_recorded_distance_days_20140101'))

    #To make features of frequency counts:
    frequency_feature(data_dir,train,test)

    print(" - Finished.")

if __name__ == '__main__':
    start_time = time.time()
    main()
    #statist()
    print("--- execution took %s seconds ---" % (time.time() - start_time))
