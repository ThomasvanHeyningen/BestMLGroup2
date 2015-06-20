'''
Main file with the classification methods from the Waterdragers group for the Pump it up challenge.
Members: Bas van Berkel, Hans-Christiaan Braun, Erik Eppenhof, Thomas van Heyningen, Senna van Iersel, and  Harmen Prins
Running this file will run our classification methods. Running the full ensemble takes 30-40 minutes on an average PC.
'''
from __future__ import division

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
        # We first try to get the data using Unix paths
        data_dir=os.path.dirname(os.path.abspath('')) + '/data/'

        #the main train file expanded with temperature features:
        train = pd.read_csv(data_dir + 'Training_set_with_temperature.csv')
        #the labels of the train rows:
        trainlabels = pd.read_csv(data_dir + 'trainlabels.csv')
        #the main test file expanded with temperature features:
        test = pd.read_csv(data_dir + 'Test_set_with_temperature.csv')
        #the extra features generated using generate_features.py:
        extratrain=pd.read_csv(data_dir + 'extratrainfeatures.csv')
        extratest=pd.read_csv(data_dir + 'extratestfeatures.csv')
        #the basic train and test data but now with entropy-optimal ordening:
        orderedtrain=pd.read_csv(data_dir + 'train_ordered.csv')
        orderedtest=pd.read_csv(data_dir + 'test_ordered.csv')
    except IOError:
        # If the Unix path fails we load it using the Windows path type
        data_dir='..\\data\\'

        train = pd.read_csv(data_dir + 'Training_set_with_temperature.csv')
        trainlabels = pd.read_csv(data_dir + 'trainlabels.csv')
        test = pd.read_csv(data_dir + 'Test_set_with_temperature.csv')
        extratrain=pd.read_csv(data_dir + 'extratrainfeatures.csv')
        extratest=pd.read_csv(data_dir + 'extratestfeatures.csv')
        orderedtrain=pd.read_csv(data_dir + 'train_ordered.csv')
        orderedtest=pd.read_csv(data_dir + 'test_ordered.csv')

    #The selection of the labels/features to use for the training/testing:
    #If a label is removed please add it to the removed labels.

    #First we load the numerical labels from the basic train file, this is the list of features we use:
    numerical_label = ['gps_height','longitude','latitude','region_code','district_code','population','construction_year'
        ,'Winter average','Spring average','Summer average','Autumn average','Average']\
    #removed labels: num_private, 'Month 1','Month 2','Month 3'
    #,'Month 4','Month 5','Month 6','Month 7','Month 8','Month 9','Month 10','Month 11' ,'Month 12']

    #Secondly we load these features from the extra-train/test files. Listed are the features we use
    extra_label=['region_num','lga_num','ward_num'
        ,'public_meeting_num','scheme_management_num','scheme_name_num','permit_num','extraction_type_num'
        ,'extraction_type_group_num' ,'month_recorded','age_of_pump','date_recorded_distance_days_20140101'
        ,'basin_freq','region_freq','lga_freq','ward_freq','scheme_name_freq', 'year_recorded'
        ,'funder_clean2_num','installer_clean2_num', 'funder_clean2_freq','installer_clean2_freq']
        #,'5_nearest_functional','5_nearest_need_repair','5_nearest_broken','10_nearest_functional' have to look at this, seems that validation data is leaking through
        #,'10_nearest_need_repair','10_nearest_broken','20_nearest_functional','20_nearest_need_repair'
        #,'20_nearest_broken','40_nearest_functional','40_nearest_need_repair','40_nearest_broken']

    #removed labels: recorded_by_num, day_recorded, wpt_name_num, subvillage_num, amount_tsh,'quality_group_num','source_type_num'
    #more removed: 'funder_num','installer_num', 'funder_freq','installer_freq'
    #after hand cleaning: 'funder_clean_num','installer_clean_num', 'funder_clean_freq','installer_clean_freq'
    #'extraction_type_class_num','management_num','management_group_num','payment_num'
    #    ,'payment_type_num','water_quality_num','quantity_num','quantity_group_num','source_num'
    #    ,'source_class_num','waterpoint_type_num','waterpoint_type_group_num', 'basin_num'

    #These are the labels of the features from the ordered files.
    ordered_label = ['extraction_type_class','management','management_group','payment','payment_type','water_quality'
        ,'quantity','quantity_group','source','source_class','waterpoint_type','waterpoint_type_group', 'basin']

    #Processing the labels of the three files into the train and test sets:
    X_train_num=train[numerical_label]
    X_test_num=test[numerical_label]
    X_extratrain_num=extratrain[extra_label]
    X_extratest_num=extratest[extra_label]
    X_orderedtrain=orderedtrain[ordered_label]
    X_orderedtest=orderedtest[ordered_label]

    #Stacking the numpy arrays to put everything in one big matrix:
    Xtrain=np.hstack((X_train_num, X_extratrain_num, X_orderedtrain))
    Xtest=np.hstack((X_test_num, X_extratest_num, X_orderedtest))

    #Add the labels for the train set:
    trainset = np.column_stack((Xtrain,trainlabels['status_group']))

    #Make the train/validation split.
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        trainset[:, 0:-1], trainset[:, -1], train_size=train_size)

    #Based on whether train or test data is requested we return the correct one. Also the size of the data is printed.
    if testdata:
        print("Testing set has {0[0]} rows and {0[1]} columns/features".format(Xtest.shape))
        return (Xtest, test['id'])
    else:
        print("Training set has {0[0]} rows and {0[1]} columns/features".format(Xtrain.shape))
        return(X_train.astype(float), X_valid.astype(float), Y_train.astype(str), Y_valid.astype(str))

def trainclf():
    '''
    Code to train a classifier. Gets the data from load_data.

    Returns: the classifiers and ensemble weights

    '''
    #loading the data from load_data:
    X_train, X_valid, y_train, y_valid = load_data(train_size=0.8, testdata=False)


    clfs = [] #Create an empty array to store the classifiers in
    print(" -- Start training.") # print that we start with the actual training.

    '''
    # K-nearest neighbor classifier .70 accuracy (too low to be considered by the ensemble).
    nn = KNeighborsClassifier(15, weights='distance')
    nn.fit(X_train, y_train)
    print('nn 1 LogLoss {score}'.format(score=log_loss(y_valid, nn.predict_proba(X_valid))))
    print('nn 1 accuracy {score}'.format(score=accuracy_score(y_valid, nn.predict(X_valid))))
    clfs.append(nn)
    '''

    '''
    #Code voor MLP, quite low performance. MLP normally works better with normalized or log data,
    #tf-idf was a quick fix for this in the ottogroup challenge, and worked well for that challenge.
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

    '''
    # ADABOOST
#   AdaBoost with RF, random_state omitted, max_depth & n_estimators lower
    ada = AdaBoostClassifier(RandomForestClassifier(n_jobs=3, n_estimators=200, max_depth=15))
    ada.fit(X_train, y_train)
    print('RFC 1 LogLoss {score}'.format(score=log_loss(y_valid, ada.predict_proba(X_valid))))
    print('RFC 1 accuracy {score}'.format(score=accuracy_score(y_valid, ada.predict(X_valid))))
    clfs.append(ada)
    '''

    # First  RandomForestClassifier
    #First we define the parameters for our classifier:
    clf = RandomForestClassifier(n_jobs=3, n_estimators=400, max_depth=19, random_state=60)
    clf.fit(X_train, y_train) #Then we fit the classifier:
    #Then we train the log-loss and accuracy score of the classifier on the validation set
    print('RFC 1 LogLoss {score}'.format(score=log_loss(y_valid, clf.predict_proba(X_valid))))
    print('RFC 1 accuracy {score}'.format(score=accuracy_score(y_valid, clf.predict(X_valid))))
    clfs.append(clf) #finally we add the classifier to the ensemble

    #First GBM:
    gbm=GradientBoostingClassifier(n_estimators=50, max_depth=13, max_features=20, min_samples_leaf=3,verbose=1, subsample=0.85, random_state=187)
    gbm.fit(X_train, y_train)
    print('GBM LogLoss {score}'.format(score=log_loss(y_valid, gbm.predict_proba(X_valid))))
    print('GBM accuracy {score}'.format(score=accuracy_score(y_valid, gbm.predict(X_valid))))
    clfs.append(gbm)

    #Second GBM
    gbm2=GradientBoostingClassifier(n_estimators=50, max_depth=15, max_features=20, min_samples_leaf=5,verbose=1, subsample=0.90, random_state=186)
    gbm2.fit(X_train, y_train)
    print('GBM 2 LogLoss {score}'.format(score=log_loss(y_valid, gbm2.predict_proba(X_valid))))
    print('GBM 2 accuracy {score}'.format(score=accuracy_score(y_valid, gbm2.predict(X_valid))))
    clfs.append(gbm2)

    #Second RF
    clf2 = RandomForestClassifier(n_jobs=3, n_estimators=400, max_depth=21, random_state=188)
    clf2.fit(X_train, y_train)
    print('RFC 2 LogLoss {score}'.format(score=log_loss(y_valid, clf2.predict_proba(X_valid))))
    print('RFC 2 accuracy {score}'.format(score=accuracy_score(y_valid, clf2.predict(X_valid))))
    clfs.append(clf2)

    print(" -- Finished training") # The actual training of the independent classifiers has finished.

    #Now it is time to train the optimal weights for the ensemble.
    #To train the optimal weights we need the probabilities to train on:
    predictions = []
    for clf in clfs:
        predictions.append(clf.predict_proba(X_valid))

    #the algorithms need a starting value, right now we chose 0.5 for all weights:
    starting_values = [0.5]*len(predictions)

    #adding constraints and a different solver. This part of the code is derived from:
    #https://kaggle2.blob.core.windows.net/forum-message-attachments/75655/2393/otto%20model%20weights.pdf?sv=2012-02-12&se=2015-05-03T21%3A22%3A17Z&sr=b&sp=r&sig=rkeA7EJC%2BiQ%2FJ%2BcMpcA4lYQLFh6ubNqs2XAkGtFsAv0%3D
    cons = ({'type':'eq','fun':lambda w: 1-sum(w)})

    #our weights are bound between 0 and 1
    bounds = [(0,1)]*len(predictions)
    #The minimization function, We also have an accuracy_func, that one however has weird behaviour, log-loss fitting worked better.
    res = minimize(log_loss_func, starting_values, (predictions, y_valid),  method='SLSQP', bounds=bounds, constraints=cons)

    #Print the final score of the ensemble on the validation sets. Also print the weights as we copy these manually for
    # a later run on the test-set
    print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
    print('Best Weights: {weights}'.format(weights=res['x']))

    ## This will combine the model probabilities using the optimized weights
    y_prob = 0
    for i in range( len(predictions)):
        y_prob += predictions[i]*res['x'][i]
    y_prob=np.array(y_prob)
    max_index=np.argmax(y_prob, axis=1)

    #This code will enable us to get the accuracy of the ensemble:
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
    #Print the accuracy:
    print ('Ensemble accuracy: {accuracy}'.format(accuracy=accuracy_score(y_valid, y_compare)))

    #print clf.feature_importances_ # to print the importance of all features.

    #Print a classification report,seeing which classes have worse performance:
    #y_pred = clf.predict(X_valid)
    #print classification_report(y_valid, y_pred)

    return clfs, res['x']

def log_loss_func(weights, predictions, y_valid):
    '''
    Function to optimize the weights based on log_loss. Which is not ideal because we want to optimize accuracy.
    It does however provide some functionality, and the accuracy function still has some issues.
    '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction

    return log_loss(y_valid, final_prediction)

def accuracy_func(weights, predictions, y_valid):
    '''
    Function to optimize the weights based on accuracy. This function has some issues in that it only outputs equal
    chances (1/classifiers). Thus I doubt that optimizing the weights using accuracy like this works.
    '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
        final_prediction += weight*prediction
    max_index=np.argmax(final_prediction, axis=1)
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

    return accuracy_score(y_valid, y_pred)

def make_submission(clfs, weights):
    '''
    Code to make a submission:
    Gets a list of classifiers and uses these to classify the test-set which is loaded using load_data
    also gets a list of weights to use the probabilities proposed by the classifiers with those weights.
    '''
    #path to store the submission to:
    #path = ('..\submissions\my_submission_{date}.csv'.format(date=time.strftime("%y%m%d%H%M"))) # alternative path
    path = ('githubsubmission.csv') # quick hack for some git issues on remote Unix system.
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
    weights= [0.21901258,  0.1913049,   0.30512143,  0.28456109] # RF, GBM, GBM2, RF2
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
