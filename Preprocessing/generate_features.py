'''
File to generate and preprocess some features of the data from the Waterdragers group for the Pump it up challenge.
Members: Bas van Berkel, Hans-Christiaan Braun, Erik Eppenhof, Thomas van Heyningen, Senna van Iersel, and  Harmen Prins
Most functions are commented in the main function because they only have to be run once or twice on the data.
'''

import os
import time
import re
import numpy as np
import pandas as pd
import datetime as dt

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

def closepumps(traindata,testdata, trainlabels, k=5):
    '''
    Function to report the classmembership of the k closests pumps in the training set for both the train and test data.
    Sadly this function makes the algorithm overfit on the training data, getting a cv score of .83-.84 but a
    leaderboard score of .80-.81

    '''
    #Get the sizes of the data:
    (testrows, testcolumns)=testdata.shape
    (trainrows, traincolumns)=traindata.shape
    #Take the train and testdata (fill empty fields with -1)
    traindata= traindata.fillna(-1)
    testdata = testdata.fillna(-1)
    #Take the relevant columns from the data
    train = np.column_stack((traindata['longitude'], traindata['latitude']))
    test = np.column_stack((testdata['longitude'], testdata['latitude']))

    #find the nearest Neigbors
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(train)
    #Get the distances and indices of the k+1 nearest neighbors
    distances, indices = nbrs.kneighbors(train)
    #Initialize necessary arrays
    trainfunctionalarray = []
    trainrepairarray =[]
    trainbrokenarray =[]
    #Walk throught the complete train set
    for i in xrange(0, len(indices)):
        functional = 0
        repair=0
        broken=0
        #for each pump in the neigbors of the current pump get the label, we exclude the first as that is the pump itself.
        for pump in indices[i][1:]:
            pumplabel = trainlabels.loc[pump]['status_group']
            if(pumplabel=='functional needs repair'):
                repair+=1
            elif(pumplabel=='functional'):
                functional+=1
            else:
                broken+=1
        #add the data to the array for each pump
        trainfunctionalarray.append(functional)
        trainrepairarray.append(repair)
        trainbrokenarray.append(broken)

    #get the nn of the testset and the indexes
    testdistances, testindices = nbrs.kneighbors(test)
    testfunctionalarray = []
    testrepairarray =[]
    testbrokenarray =[]

    #Walk throught the complete test set
    for i in xrange(0, len(testindices)):
        functional = 0
        repair=0
        broken=0
        #for each pump in the neigbors of the current pump get the label, don't take the last because we use k+1
        # to get one extra for the training set.
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
    #Store all of the new data arrays to the extrafeatures file.
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
    '''
    For some categorical features frequency representations of their parameters could be informative.
    We again use the store_data function to store the data to the relevant file
    '''
    #The parameters we want to get frequency values for (unless the parameters are specified in the call.
    if parameters == None:
        names_parameters=['funder', 'installer','basin','region','lga','ward','scheme_name']
    else:
        names_parameters=parameters

    #Get the size of the data and load in the data, filling empty fields with -1
    (testrows, testcolumns)=testdata.shape
    (trainrows, traincolumns)=traindata.shape
    traindata=traindata.fillna(-1)
    testdata = testdata.fillna(-1)

    #For all of the colomns/features we want to get the frequency from:
    for feature in names_parameters:
        #Count the data and get the unique values and their counts:
        countdata=pd.concat([traindata[feature],testdata[feature]], axis=0)
        unique, counts = np.unique(countdata, return_counts=True)
        '''
        #Some code to get insight into the data
        print len(unique)
        for i in range(0,len(unique)):
            print str(unique[i]) + ' <- name  -- count -> ' + str(counts[i])
        '''
        #Initialize some arrays
        train_frq = []
        test_frq = []
        #Loop for all the train data:
        for freq in traindata[feature]:
            if freq is -1: # if the data is missing put -1
                train_frq.append(-1)
            elif freq is '0': # if the data is missing put -1
                train_frq.append(-1)
            else: #otherwise put the frequency of that field using the unique/counts mapping.
                train_frq.append(counts[np.where(unique==freq)[0][0]])
        #Loop for the test data:
        for freq in testdata[feature]:
            if freq is -1: # if the data is missing put -1
                test_frq.append(-1)
            elif freq is '0': # if the data is missing put -1
                test_frq.append(-1)
            else: #otherwise put the frequency of that field using the unique/counts mapping.
                test_frq.append(counts[np.where(unique==freq)[0][0]])

        #Create a new array to put the data in and store it in the file using the store_data function
        newtestdata=np.zeros((testrows,1), dtype=int)
        newtestdata[:, 0]=np.array(test_frq)
        store_data(newtestdata, train=False,labels=(feature +'_freq'), one=True)
        newtraindata=np.zeros((trainrows,1), dtype=int)
        newtraindata[:, 0]=np.array(train_frq)
        store_data(newtraindata, train=True,labels=(feature +'_freq'), one=True)


def encode_categorical(data_dir,traindata,testdata, parameters):
    '''
    Function to encode the categorical features as numerical features using a LabelEncoder.
    Important is to encode the test and train file in the same manner
    '''
    #Unless pre-picked parameters to work on are given we work on these parameters.
    if parameters == None:
        names_parameters=['funder','installer','wpt_name','basin','subvillage','region','lga','ward','public_meeting','recorded_by'
        ,'scheme_management','scheme_name','permit','extraction_type','extraction_type_group','extraction_type_class','management','management_group','payment',
                        'payment_type','water_quality','quality_group','quantity','quantity_group','source','source_type','source_class',
                       'waterpoint_type','waterpoint_type_group']
    else:
        names_parameters=parameters
    #Get the shape of the data
    (testrows, testcolumns)=testdata.shape
    (trainrows, traincolumns)=traindata.shape

    #For all of the specified features:
    for feature in names_parameters:
        #We initialize an encoder.
        le = LabelEncoder()
        #We add the test and train data together
        fitdata=np.append(traindata[feature].values,testdata[feature].values)
        #Because the mapping to numbers has to be the same in both, thus we fit them together.
        le.fit(fitdata)
        #We however want to work with the train and test data separate so we transform them separately
        train_cat=le.transform(traindata[feature])
        test_cat = le.transform(testdata[feature])
        #We initialize empty arrays for the train and test data and store these to the relevant files.
        newtestdata=np.zeros((testrows,1), dtype=int)
        newtestdata[:, 0]=np.array(test_cat)
        store_data(newtestdata, train=False,labels=(feature +'_num'), one=True)
        newtraindata=np.zeros((trainrows,1), dtype=int)
        newtraindata[:, 0]=np.array(train_cat)
        store_data(newtraindata, train=True,labels=(feature +'_num'), one=True)

def date_features(data):
    '''
    Function to create features using the two data features
    '''
    #Get the shape:
    (rows, columns)=data.shape
    #We are generate the following 5 features:
    years = []
    months =[]
    days = []
    age = []
    distance = []
    #We walk to the data and construction array at the same time using a zip:
    for date, construction in zip(data['date_recorded'], data['construction_year']):
        #Using regex we match the textual date in date to 3 numbers
        match=re.match('([0-9]{4})\-([0-9]{2})\-([0-9]{2})',date)
        #These numbers represent year, month and day. We store them in the relevant array.
        years.append(match.group(1))
        months.append(match.group(2))
        days.append(match.group(3))
        #Data of recording is also derived, we reformulate the data to calculate the distance
        recorddate=dt.date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        #We calculate the distance between the construction and the date of recording in years
        age.append(recorddate.year-construction)
        #We also calculate the date between recording and 1 January 2014 (everything was recorded before that data)
        begin2014 = dt.date(2014, 01, 01)
        dist = abs(begin2014 - recorddate)
        distance.append(dist.days)
    #We return the data as one matrix (which is stored via the main function, different from the other functions above)
    newdata=np.zeros((rows,5), dtype=int)
    newdata[:, 0]=np.array(years)
    newdata[:, 1]=np.array(months)
    newdata[:, 2]=np.array(days)
    newdata[:, 3]=np.array(age)
    newdata[:, 4]=np.array(distance)
    return(newdata)

def store_data(newdata, ids=None, data_dir=None, train=True, labels=(''), one=False):
    '''
    Function to store the data generated with the other features into separate files called extratrainfeatures.csv
    and extratestfeatures.csv because we don't like to change the original data.
    '''
    if train:
        try:
            # first try to save it as a Unix path
            data_dir=os.path.dirname(os.path.abspath('')) + '/data/'
            trainfile = pd.read_csv(data_dir + 'extratrainfeatures.csv')
        except IOError:
            # then try a Windows path
            data_dir='..\\data\\'
            trainfile = pd.read_csv(data_dir + 'extratrainfeatures.csv')
        #trainfile=pd.DataFrame(data=ids)  # only needed to generate the file the first time
        trainfile.set_index('id') # set the index
        index=0
        if one: # if theres only one new label
            trainfile[labels]=newdata[:,0]
        else: # if there are multiple labels we are saving.
            for label in labels:
                trainfile[label]=newdata[:,index]
                index=index+1
        #The actual saving
        trainfile.to_csv(data_dir + 'extratrainfeatures.csv', index_label='id', index=False)
    #All of the above, but now for the test features
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
        # load the data as a Unix path
        data_dir=os.path.dirname(os.path.abspath('')) + '/data/'

        train = pd.read_csv(data_dir + 'trainset.csv')
        trainlabels = pd.read_csv(data_dir + 'trainlabels.csv')
        test = pd.read_csv(data_dir + 'testset.csv')
        newtrain = pd.read_csv(data_dir + 'nieuwtrain.csv')
        newtest = pd.read_csv(data_dir + 'nieuwtest.csv')
    except IOError:
        # If Unix fails we try a Windows path
        data_dir='..\\data\\'
        train = pd.read_csv(data_dir + 'trainset.csv')
        trainlabels = pd.read_csv(data_dir + 'trainlabels.csv')
        test = pd.read_csv(data_dir + 'testset.csv')
        newtrain = pd.read_csv(data_dir + 'nieuwtrain.csv')
        newtest = pd.read_csv(data_dir + 'nieuwtest.csv')

    ##All of the calls are commented because running them twice is most of the times not necessary.

    #Clean up textual data to prevent duplicates with different names by removing spaces and uppercase
    #clean_text(data_dir,newtrain,newtest)

    #to make the categorical features numeric:
    #encode_categorical(data_dir,train,test)

    #feature on distance to other pumps:
    #closepumps(train,test, trainlabels, k=5) # ran it with k=5,10,20 and 40

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
