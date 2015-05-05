
from __future__ import division
import sys
import time

import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer

def load_data(train_size=0.8, testdata=False):
    '''
    This function loads the data from the relevant data files.
    After loading the data a selection is made of the data to use (hardcoded).

    train_size: the size of the training set in the train-test split: between 0 and 1.
    testdata: If true the test set is loaded to generate a submission.

    return:
    if testdata is true: the relevant features from the test data
    else: the train and validation set from the train data to learn an algorithm
    '''

    # loading the data from relevant files:
    data_dir='..\\data\\'
    train = pd.read_csv(data_dir + 'trainset.csv')
    trainlabels = pd.read_csv(data_dir + 'trainlabels.csv')
    test = pd.read_csv(data_dir + 'testset.csv')
    extratrain=pd.read_csv('extratrainfeatures.csv')
    extratest=pd.read_csv('extratestfeatures.csv')

    #out of use:
    vec = DictVectorizer()

    #for name in train.columns[1:-1]:
    #    names_cat.append(name)
    #   print name, len(np.unique(train[name]))

    #The selection of the labels/features to use for the training/testing.
    #If a label is removed please add it to the removed labels.
    numerical_label = ['gps_height','longitude','latitude','region_code','district_code','population','construction_year']
    #removed labels: num_private
    extra_label=['funder_num','installer_num','basin_num','region_num','lga_num','ward_num'
        ,'public_meeting_num','scheme_management_num','scheme_name_num','permit_num','extraction_type_num'
        ,'extraction_type_group_num','extraction_type_class_num','management_num','management_group_num','payment_num'
        ,'payment_type_num','water_quality_num','quality_group_num','quantity_num','quantity_group_num','source_num'
        ,'source_type_num','source_class_num','waterpoint_type_num','waterpoint_type_group_num'
        ,'month_recorded','age_of_pump','date_recorded_distance_days_20140101','funder_freq','installer_freq'
        ,'basin_freq','region_freq','lga_freq','ward_freq','scheme_name_freq', 'year_recorded']
    #removed labels: recorded_by_num, day_recorded, wpt_name_num, subvillage_num, amount_tsh

    #Processing the labels into the train and test sets.
    X_train_num=train[numerical_label]
    X_test_num=test[numerical_label]
    X_extratrain_num=extratrain[extra_label]
    X_extratest_num=extratest[extra_label]

    Xtrain=np.hstack((X_train_num, X_extratrain_num))
    Xtest=np.hstack((X_test_num, X_extratest_num))
    trainset = np.column_stack((Xtrain,trainlabels['status_group']))
    print("Training set has {0[0]} rows and {0[1]} columns/features".format(train.shape))
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        trainset[:, 0:-1], trainset[:, -1], train_size=train_size
    , random_state=6)

    if testdata:
        return (Xtest, test['id'])
    else:
        return(X_train, X_valid, Y_train, Y_valid)

def trainrf():
    '''
    Code to train a classifier. Gets the data from load_data.

    Returns: the classifier and an encoder (I think this one is out of use.
    '''
    #loading the data from load_data:
    X_train, X_valid, y_train, y_valid = load_data(train_size=0.80, testdata=False)

    # Number of trees, increase this to improve
    n_estimators = 200
    clf = RandomForestClassifier(n_jobs=3, n_estimators=n_estimators, max_depth=23, random_state=5)
    #clf=GradientBoostingClassifier(n_estimators=60, max_depth=10, max_features=20, min_samples_leaf=4,verbose=1, subsample=0.85) # 0.85 score
    #clf = SGDClassifier(loss="hinge", penalty="l2", verbose=True)
    print(" -- Start training.")
    clf.fit(X_train, y_train)
    print clf.feature_importances_
    y_prob = clf.predict_proba(X_valid)
    print(" -- Finished training 1")

    y_pred = clf.predict(X_valid)
    print classification_report(y_valid, y_pred)
    print accuracy_score(y_valid, y_pred)

    return clf

def make_submission(clf, path='my_submission.csv'):
    '''
    Code to make a submission:
    Gets a classifier and uses this to classify the test-set which is loaded using load_data
    '''
    path = sys.argv[3] if len(sys.argv) > 3 else path
    X_test, ids = load_data(testdata=True)

    y_pred = clf.predict(X_test)
    with open(path, 'w') as f:
        f.write('id,status_group\n')
        for id, pred in zip(ids, y_pred):
            f.write(str(id))
            f.write(',')
            f.write(pred)
            f.write('\n')
    print(" -- Wrote submission to file {}.".format(path))

def main():
    print(" - Start.")
    model = trainrf()
    #make_submission(model)
    print(" - Finished.")

if __name__ == '__main__':
    start_time = time.time()
    main()
    #statist()
    print("--- execution took %s seconds ---" % (time.time() - start_time))
