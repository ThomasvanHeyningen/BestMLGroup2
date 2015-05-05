
from __future__ import division
import sys
import time

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, log_loss

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
    clfs = []

    print(" -- Start training.")
    clf = RandomForestClassifier(n_jobs=3, n_estimators=100, max_depth=23, random_state=5)
    clf.fit(X_train, y_train)
    print('RFC 1 LogLoss {score}'.format(score=log_loss(y_valid, clf.predict_proba(X_valid))))
    print('RFC 1 accuracy {score}'.format(score=accuracy_score(y_valid, clf.predict(X_valid))))
    clfs.append(clf)

    gbm=GradientBoostingClassifier(n_estimators=40, max_depth=10, max_features=15, min_samples_leaf=3,verbose=1, subsample=0.8, random_state=7)
    gbm.fit(X_train, y_train)
    print('GBM LogLoss {score}'.format(score=log_loss(y_valid, gbm.predict_proba(X_valid))))
    print('GBM accuracy {score}'.format(score=accuracy_score(y_valid, gbm.predict(X_valid))))
    clfs.append(gbm)
    
    clf2 = RandomForestClassifier(n_jobs=3, n_estimators=100, max_depth=23, random_state=8)
    clf2.fit(X_train, y_train)
    print('RFC 1 LogLoss {score}'.format(score=log_loss(y_valid, clf2.predict_proba(X_valid))))
    print('RFC 1 accuracy {score}'.format(score=accuracy_score(y_valid, clf2.predict(X_valid))))
    clfs.append(clf2)
    print(" -- Finished training")

    predictions = []
    for clf in clfs:
        predictions.append(clf.predict_proba(X_valid))
    #the algorithms need a starting value, right now we chose 0.5 for all weights
    #its better to choose many random starting points and run minimize a few times
    starting_values = [0.5]*len(predictions)

    #adding constraints  and a different solver as suggested by user 16universe
    #https://kaggle2.blob.core.windows.net/forum-message-attachments/75655/2393/otto%20model%20weights.pdf?sv=2012-02-12&se=2015-05-03T21%3A22%3A17Z&sr=b&sp=r&sig=rkeA7EJC%2BiQ%2FJ%2BcMpcA4lYQLFh6ubNqs2XAkGtFsAv0%3D
    cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
    #our weights are bound between 0 and 1
    bounds = [(0,1)]*len(predictions)
    print len(predictions)
    print len(y_valid)
    res = minimize(log_loss_func, starting_values, (predictions, y_valid),  method='SLSQP', bounds=bounds, constraints=cons)

    print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
    print('Best Weights: {weights}'.format(weights=res['x']))

    #print clf.feature_importances_
    #y_pred = clf.predict(X_valid)
    #print classification_report(y_valid, y_pred)

    return clf

def log_loss_func(weights, predictions, y_valid):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction

    return log_loss(y_valid, final_prediction)

def accuracy_func(weights, predictions, y_valid):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
        final_prediction += weight*prediction

    return accuracy_score(y_valid, final_prediction)

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
