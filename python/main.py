
from __future__ import division
import sys
import time
import os

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.svm import LinearSVC
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from multilayer_perceptron  import MultilayerPerceptronClassifier
from sklearn.feature_extraction.text import TfidfTransformer

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
    try:
        # Unix
        data_dir=os.path.dirname(os.path.abspath('')) + '/data/'
        train = pd.read_csv(data_dir + 'trainset.csv')
        trainlabels = pd.read_csv(data_dir + 'trainlabels.csv')
        test = pd.read_csv(data_dir + 'testset.csv')
        extratrain=pd.read_csv(data_dir + 'extratrainfeatures.csv')
        extratest=pd.read_csv(data_dir + 'extratestfeatures.csv')
    except IOError:
        # Windows
        data_dir='..\\data\\'

        print (data_dir + 'trainset.csv')
        train = pd.read_csv(data_dir + 'trainset.csv')
        trainlabels = pd.read_csv(data_dir + 'trainlabels.csv')
        test = pd.read_csv(data_dir + 'testset.csv')
        extratrain=pd.read_csv(data_dir + 'extratrainfeatures.csv')
        extratest=pd.read_csv(data_dir + 'extratestfeatures.csv')

    #for name in train.columns[1:-1]:
    #    names_cat.append(name)
    #   print name, len(np.unique(train[name]))

    #The selection of the labels/features to use for the training/testing.
    #If a label is removed please add it to the removed labels.
    numerical_label = ['gps_height','longitude','latitude','region_code','district_code','population','construction_year']
    #removed labels: num_private

    extra_label=['basin_num','region_num','lga_num','ward_num'
        ,'public_meeting_num','scheme_management_num','scheme_name_num','permit_num','extraction_type_num'
        ,'extraction_type_group_num','extraction_type_class_num','management_num','management_group_num','payment_num'
        ,'payment_type_num','water_quality_num','quantity_num','quantity_group_num','source_num'
        ,'source_class_num','waterpoint_type_num','waterpoint_type_group_num'
        ,'month_recorded','age_of_pump','date_recorded_distance_days_20140101'
        ,'basin_freq','region_freq','lga_freq','ward_freq','scheme_name_freq', 'year_recorded'
        ,'funder_clean2_num','installer_clean2_num', 'funder_clean2_freq','installer_clean2_freq']
        #,'5_nearest_functional','5_nearest_need_repair','5_nearest_broken','10_nearest_functional' have to look at this, seems that validation data is leaking through
        #,'10_nearest_need_repair','10_nearest_broken','20_nearest_functional','20_nearest_need_repair'
        #,'20_nearest_broken','40_nearest_functional','40_nearest_need_repair','40_nearest_broken']

    #removed labels: recorded_by_num, day_recorded, wpt_name_num, subvillage_num, amount_tsh,'quality_group_num','source_type_num'
    #more removed: 'funder_num','installer_num', 'funder_freq','installer_freq'
    #after hand cleaning: 'funder_clean_num','installer_clean_num', 'funder_clean_freq','installer_clean_freq'

    #Processing the labels into the train and test sets.
    X_train_num=train[numerical_label]
    X_test_num=test[numerical_label]
    X_extratrain_num=extratrain[extra_label]
    X_extratest_num=extratest[extra_label]

    Xtrain=np.hstack((X_train_num, X_extratrain_num))
    Xtest=np.hstack((X_test_num, X_extratest_num))
    trainset = np.column_stack((Xtrain,trainlabels['status_group']))
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        trainset[:, 0:-1], trainset[:, -1], train_size=train_size
    , random_state=6)

    if testdata:
        print("Testing set has {0[0]} rows and {0[1]} columns/features".format(test.shape))
        return (Xtest, test['id'])
    else:
        print("Training set has {0[0]} rows and {0[1]} columns/features".format(train.shape))
        return(X_train.astype(float), X_valid.astype(float), Y_train.astype(str), Y_valid.astype(str))

def trainclf():
    '''
    Code to train a classifier. Gets the data from load_data.

    Returns: the classifier and an encoder (I think this one is out of use.
    '''
    #loading the data from load_data:
    X_train, X_valid, y_train, y_valid = load_data(train_size=0.999, testdata=False)

    # Number of trees, increase this to improve
    clfs = []
    print(" -- Start training.")

    '''
    # K-nearest neighbor classifier .70 accuracy (too low to be considered by the ensemble).
    nn = KNeighborsClassifier(15, weights='distance')
    nn.fit(X_train, y_train)
    print('nn 1 LogLoss {score}'.format(score=log_loss(y_valid, nn.predict_proba(X_valid))))
    print('nn 1 accuracy {score}'.format(score=accuracy_score(y_valid, nn.predict(X_valid))))
    clfs.append(nn)
    '''
    ''' Voor Senna om mee te klooien, maar doet voorlopig nog weinig
    #Code voor MLP
    transformer = TfidfTransformer(norm='l2', smooth_idf=True, sublinear_tf=False, use_idf=True)
    X_test, ids = load_data(testdata=True) #TESTDATA WATCH OUT DO NOT USE FOR OTHER PURPOSES!!
    transformer.fit_transform(np.vstack([X_valid,X_train, X_test]))
    X_train_tf = transformer.transform(X_train)
    X_valid_tf = transformer.transform(X_valid)

    mlp = MultilayerPerceptronClassifier(hidden_layer_sizes = (30,15), activation = 'relu',\
                                             max_iter = 200, alpha = 0.0008, verbose=0, learning_rate='invscaling', learning_rate_init=0.9, power_t=0.8)
    mlp.fit(X_train_tf, y_train)
    print('MLP LogLoss {score}'.format(score=log_loss(y_valid, mlp.predict_proba(X_valid_tf))))
    print('MLP accuracy {score}'.format(score=accuracy_score(y_valid, mlp.predict(X_valid_tf))))
    clfs.append(mlp)
    '''
    # Normal RandomForestClassifier
    clf = RandomForestClassifier(n_jobs=3, n_estimators=800, max_depth=23, random_state=180)
#   AdaBoost with RF, random_state omitted, max_depth & n_estimators lower
#   clf = AdaBoostClassifier(RandomForestClassifier(n_jobs=3, n_estimators=200, max_depth=15))
    clf.fit(X_train, y_train)
    print('RFC 1 LogLoss {score}'.format(score=log_loss(y_valid, clf.predict_proba(X_valid))))
    print('RFC 1 accuracy {score}'.format(score=accuracy_score(y_valid, clf.predict(X_valid))))
    clfs.append(clf)

    gbm=GradientBoostingClassifier(n_estimators=70, max_depth=13, max_features=20, min_samples_leaf=3,verbose=1, subsample=0.85, random_state=187)
    gbm.fit(X_train, y_train)
    print('GBM LogLoss {score}'.format(score=log_loss(y_valid, gbm.predict_proba(X_valid))))
    print('GBM accuracy {score}'.format(score=accuracy_score(y_valid, gbm.predict(X_valid))))
    clfs.append(gbm)

    gbm2=GradientBoostingClassifier(n_estimators=70, max_depth=15, max_features=20, min_samples_leaf=5,verbose=1, subsample=0.90, random_state=186)
    gbm2.fit(X_train, y_train)
    print('GBM 2 LogLoss {score}'.format(score=log_loss(y_valid, gbm2.predict_proba(X_valid))))
    print('GBM 2 accuracy {score}'.format(score=accuracy_score(y_valid, gbm2.predict(X_valid))))
    clfs.append(gbm2)

    clf2 = RandomForestClassifier(n_jobs=3, n_estimators=800, max_depth=29, random_state=188)
    clf2.fit(X_train, y_train)
    print('RFC 2 LogLoss {score}'.format(score=log_loss(y_valid, clf2.predict_proba(X_valid))))
    print('RFC 2 accuracy {score}'.format(score=accuracy_score(y_valid, clf2.predict(X_valid))))
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
    res = minimize(log_loss_func, starting_values, (predictions, y_valid),  method='SLSQP', bounds=bounds, constraints=cons)

    print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
    print('Best Weights: {weights}'.format(weights=res['x']))

    ## This will combine the model probabilities using the optimized weights
    y_prob = 0
    for i in range( len(predictions)):
        y_prob += predictions[i]*res['x'][i]
    y_prob=np.array(y_prob)
    max_index=np.argmax(y_prob, axis=1)
    y_compare = []
    for i in range(len(max_index)):
        if max_index[i] == 0:
            y_compare.append('functional')
        elif max_index[i] == 1:
            y_compare.append('functional needs repair')
        elif max_index[i] == 2:
            y_compare.append('non functional')
        else:
            y_compare.append('error')

    print ('Ensemble accuracy: {accuracy}'.format(accuracy=accuracy_score(y_valid, y_compare)))

    print clf.feature_importances_
    #y_pred = clf.predict(X_valid)
    #print classification_report(y_valid, y_pred)

    return clfs, res['x']

def log_loss_func(weights, predictions, y_valid):
    '''
    Function to optimize the weights based on log_loss. Which is not ideal because we want to optimize accuracy.
    It does however provide some functionality, and different from the accuracy function this one works.
    '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction

    return log_loss(y_valid, final_prediction)

def accuracy_func(weights, predictions, y_valid):
    '''
    Function to optimize the weights based on accuracy.
    We sadly did not get this function to work. So this still is a "work in progress".
    '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
        final_prediction += weight*prediction

    return accuracy_score(y_valid, final_prediction)

def make_submission(clfs, weights):
    '''
    Code to make a submission:
    Gets a list of classifiers and uses these to classify the test-set which is loaded using load_data
    also gets a list of weights to use the probabilities proposed by the classifiers with those weights.
    '''
    #path to store the submission to:
    path = ('..\\submissions\\my_submission_{date}.csv'.format(date=time.strftime("%y%m%d%H%M")))

    #path = ('..\submissions\my_submission_{date}.csv'.format(date=time.strftime("%y%m%d%H%M"))) # alternative path

    X_test, ids = load_data(testdata=True) # load the data
    y_prob_tot = 0

    #We weight the probabilities provided by the different classifiers:
    for i in range(len(clfs)): # for all classifiers
        y_prob = clfs[i].predict_proba(X_test) # predict for the test set
        y_prob_tot += y_prob*weights[i] # and weigh
    y_prob_tot=np.array(y_prob_tot) # make the prob matrix an numpy matrix
    #pick the prediction (out of funct, non funct, need repair) with the highest probability:
    # note that the probabilities don't add up to one but to the number of classifiers,
    # this is however not a problem as we are only interested in the maximum value.
    max_index=np.argmax(y_prob_tot, axis=1)
    y_pred = []
    #in this loop we translate the predictions to the actual terms:
    for i in range(len(max_index)):
        if max_index[i] == 0:
            y_pred.append('functional')
        elif max_index[i] == 1:
            y_pred.append('functional needs repair')
        elif max_index[i] == 2:
            y_pred.append('non functional')
        else:
            y_pred.append('error')
    #Open a file and write the data (with some formatting) to the location in path
    with open(path, 'w') as f:
        f.write('id,status_group\n')
        for id, pred in zip(ids, y_pred):
            f.write(str(id))
            f.write(',')
            f.write(pred)
            f.write('\n')
    print(" -- Wrote submission to file {}.".format(path))

def main():
    '''
    The main function of the program:
    Here we first call the training, and on some occasions we also call the submission maker.
    Weights have to be set manually as it is most likely that these were derived from a 0.8 split train validation run.
    When we run with all the training data (we actually use a 0.999 split but that is not a significant difference)
    the weights can't be derived from that run (guaranteed overfitting), the weights from the 0.8 split however
    are perfectly suitable.
    '''
    print(" - Start.")
    model, weights = trainclf() #training
    #weights have to be saved from an 0.8 split to prevent heavy overfitting when run on full data
    weights= [0.30, 0.15, 0.40, 0.15] # RF, GBM, GBM2, RF2
    make_submission(model, weights) #testing
    print(" - Finished.")

if __name__ == '__main__':
    '''
    The main function here everything starts.
    Actually we only start (and close and print) a timer for the run time of the total program.
    Besides calling yet another (more real) main function.
    '''
    start_time = time.time()
    main()
    print("--- execution took %s seconds ---" % (time.time() - start_time))
