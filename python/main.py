
from __future__ import division
import sys
import time

import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer

def load_data(train_size=0.8, testdata=False):
    data_dir='..\\data\\'
    train = pd.read_csv(data_dir + 'trainset.csv')
    trainlabels = pd.read_csv(data_dir + 'trainlabels.csv')
    test = pd.read_csv(data_dir + 'testset.csv')
    extratrain=pd.read_csv('extratrainfeatures.csv')
    extratest=pd.read_csv('extratestfeatures.csv')
    vec = DictVectorizer()

    #names_cat = []
    #for name in train.columns[1:-1]:
    #    names_cat.append(name)
    #    print name, len(np.unique(train[name]))

    numerical_label = ['amount_tsh','gps_height','longitude','latitude','region_code','district_code','population','construction_year']
    #removed labels: num_private
    extra_label=['funder_num','installer_num','wpt_name_num','basin_num','subvillage_num','region_num','lga_num','ward_num'
        ,'public_meeting_num','scheme_management_num','scheme_name_num','permit_num','extraction_type_num'
        ,'extraction_type_group_num','extraction_type_class_num','management_num','management_group_num','payment_num'
        ,'payment_type_num','water_quality_num','quality_group_num','quantity_num','quantity_group_num','source_num'
        ,'source_type_num','source_class_num','waterpoint_type_num','waterpoint_type_group_num'
        ,'month_recorded','age_of_pump','date_recorded_distance_days_20140101']
    #removed labels: recorded_by_num, day_recorded, year_recorded
    X_train_num=train[numerical_label]
    X_test_num=test[numerical_label]
    X_extratrain_num=extratrain[extra_label]
    X_extratest_num=extratest[extra_label]

    Xtrain=np.hstack((X_train_num, X_extratrain_num))
    Xtest=np.hstack((X_test_num, X_extratest_num))
    trainset = np.column_stack((Xtrain,trainlabels['status_group']))
    print trainset.shape
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        trainset[:, 0:-1], trainset[:, -1], train_size=train_size,
    )

    if testdata:
        return (Xtest, test['id'])
    else:
        return(X_train, X_valid, Y_train, Y_valid)


def trainrf():
    transformer = TfidfTransformer(norm='l2', smooth_idf=True, sublinear_tf=False, use_idf=True)
    X_train, X_valid, y_train, y_valid = load_data(train_size=0.8, testdata=False)

    # Number of trees, increase this to beat the benchmark ;)
    n_estimators = 100
    clf = RandomForestClassifier(n_jobs=3, n_estimators=n_estimators, max_depth=23)
    print(" -- Start training.")
    clf.fit(X_train, y_train)
    print clf.feature_importances_
    y_prob = clf.predict_proba(X_valid)
    print(" -- Finished training 1")

    y_pred = clf.predict(X_valid)
    print classification_report(y_valid, y_pred)
    print accuracy_score(y_valid, y_pred)

    ''' #code to print predictions, pretty useless without original labels
    predictdata=np.column_stack((X_valid, y_pred))
    print predictdata.shape
    print y_prob.shape
    predictdata=np.hstack((predictdata, y_prob))
    predictdata=np.column_stack((predictdata, y_valid))
    filedata=pd.DataFrame(data=predictdata)
    filedata.to_csv('predictions.csv', index=False)
    '''
    encoder = LabelEncoder()
    y_true = encoder.fit_transform(y_valid)
    assert (encoder.classes_ == clf.classes_).all()


    return clf, encoder, transformer

def make_submission(clf, encoder, transformer, path='my_submission.csv'):
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
    model, encoder, transformer = trainrf()
    #make_submission(model, encoder, transformer)
    print(" - Finished.")
if __name__ == '__main__':
    start_time = time.time()
    main()
    #statist()
    print("--- execution took %s seconds ---" % (time.time() - start_time))
