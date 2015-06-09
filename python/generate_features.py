import os
import time
import re
import numpy as np
import pandas as pd
import datetime as dt

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

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
        for pump in testindices[i][:-1]:
            functional = 0
            repair=0
            broken=0
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
    newtraindata[:, 0]=np.array(testbrokenarray)
    store_data(newtraindata, train=True,labels=(str(k) + '_nearest_broken'), one=True)
    #encode_categorical(data_dir,traindata,testdata, ['funder_clean', 'installer_clean'])
    #frequency_feature(data_dir,traindata,testdata, ['funder_clean', 'installer_clean'])


def clean_text(data_dir,traindata,testdata):
    names_parameters=['funder', 'installer']
    (testrows, testcolumns)=testdata.shape
    (trainrows, traincolumns)=traindata.shape
    traindata= traindata.fillna(-1)
    testdata = testdata.fillna(-1)
    for feature in names_parameters:
        train_text = []
        test_text = []
        for text in traindata[feature]:
            if text is -1:
                train_text.append(-1)
            elif text is '0':
                train_text.append(-1)
            else:
                train_text.append(text.lower().replace(" ", ""))
        for text in testdata[feature]:
            if text is -1:
                test_text.append(-1)
            elif text is '0':
                test_text.append(-1)
            else:
                test_text.append(text.lower().replace(" ", ""))

        traindata[feature +'_clean'] = pd.Series(train_text, index=traindata.index)
        testdata[feature +'_clean'] = pd.Series(test_text, index=testdata.index)
    encode_categorical(data_dir,traindata,testdata, ['funder_clean', 'installer_clean'])
    frequency_feature(data_dir,traindata,testdata, ['funder_clean', 'installer_clean'])


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


def encode_categorical(data_dir,traindata,testdata, parameters):
    if parameters == None:
        names_parameters=['funder','installer','wpt_name','basin','subvillage','region','lga','ward','public_meeting','recorded_by'
        ,'scheme_management','scheme_name','permit','extraction_type','extraction_type_group','extraction_type_class','management','management_group','payment',
                        'payment_type','water_quality','quality_group','quantity','quantity_group','source','source_type','source_class',
                       'waterpoint_type','waterpoint_type_group']
    else:
        names_parameters=parameters
    (testrows, testcolumns)=testdata.shape
    (trainrows, traincolumns)=traindata.shape
    for feature in names_parameters:
        le = LabelEncoder()
        fitdata=np.append(traindata[feature].values,testdata[feature].values)
        le.fit(fitdata)
        train_cat=le.transform(traindata[feature])
        test_cat = le.transform(testdata[feature])
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

def store_data(newdata, ids=None, data_dir=None, train=True, labels=(''), one=False):

    if train:
        try:
            # Unix
            data_dir=os.path.dirname(os.path.abspath('')) + '/data/'
            trainfile = pd.read_csv(data_dir + 'extratrainfeatures.csv')
        except IOError:
            # Windows
            data_dir='..\\data\\'
            trainfile = pd.read_csv(data_dir + 'extratrainfeatures.csv')
        #trainfile=pd.DataFrame(data=ids)  # only needed to generate the file the first time
        trainfile.set_index('id')
        index=0
        if one:
            trainfile[labels]=newdata[:,0]
        else:
            for label in labels:
                trainfile[label]=newdata[:,index]
                index=index+1
        trainfile.to_csv(data_dir + 'extratrainfeatures.csv', index_label='id', index=False)
    else:
        try:
            # Unix
            data_dir=os.path.dirname(os.path.abspath('')) + '/data/'
            testfile = pd.read_csv(data_dir + 'extratestfeatures.csv')
        except IOError:
            # Windows
            data_dir='..\\data\\'
            testfile = pd.read_csv(data_dir + 'extratestfeatures.csv')
        #testfile=pd.DataFrame(data=ids) # only needed to generate the file the first time
        testfile.set_index('id')
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
    try:
        # Unix
        data_dir=os.path.dirname(os.path.abspath('')) + '/data/'

        train = pd.read_csv(data_dir + 'trainset.csv')
        trainlabels = pd.read_csv(data_dir + 'trainlabels.csv')
        test = pd.read_csv(data_dir + 'testset.csv')
    except IOError:
        # Windows
        data_dir='..\\data\\'
        train = pd.read_csv(data_dir + 'trainset.csv')
        trainlabels = pd.read_csv(data_dir + 'trainlabels.csv')
        test = pd.read_csv(data_dir + 'testset.csv')

    #Clean up textual data to prevent duplicates with different names by removing spaces and uppercase
    #clean_text(data_dir,train,test)

    #to make the categorical features numeric:
    #encode_categorical(data_dir,train,test)

    #feature on distance to other pumps:
    closepumps(data_dir,train,test, trainlabels, k=5)

    #to create the datelabels
    #newtrain = date_features(train)
    #newtest = date_features(test)
    #store_data(newtrain, train['id'], data_dir, train=True,labels=('year_recorded','month_recorded','day_recorded','age_of_pump','date_recorded_distance_days_20140101'))
    #store_data(newtest, test['id'], data_dir, newtest, train=False,labels=('year_recorded','month_recorded','day_recorded','age_of_pump','date_recorded_distance_days_20140101'))

    #To make features of frequency counts:
    #frequency_feature(data_dir,train,test)

    print(" - Finished.")

if __name__ == '__main__':
    start_time = time.time()
    main()
    #statist()
    print("--- execution took %s seconds ---" % (time.time() - start_time))