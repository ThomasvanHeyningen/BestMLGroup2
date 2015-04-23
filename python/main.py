
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
    vec = DictVectorizer()

    #names_cat = []
    #for name in train.columns[1:-1]:
    #    names_cat.append(name)
    #    print name, len(np.unique(train[name]))

    names_categorical = ['source', 'quantity', 'waterpoint_type', 'water_quality', 'payment', 'management', 'extraction_type'] #'permit', 'public_meeting', 'basin'
    #'funder','installer','wpt_name','basin','subvillage','region','lga','ward','public_meeting','recorded_by'
    #    ,'scheme_management','scheme_name','permit','extraction_type','extraction_type_group','extraction_type_class','management','management_group','payment',
    #                     'payment_type','water_quality','quality_group','quantity','quantity_group','source','source_type','source_class',
    #                     'waterpoint_type','waterpoint_type_group'
    numerical_label = ['amount_tsh','gps_height','longitude','latitude','num_private','region_code','district_code','population','construction_year']
    X_train_cat=vec.fit_transform(train[names_categorical].T.to_dict().values()).todense()
    X_test_cat = vec.transform(test[names_categorical].T.to_dict().values()).todense()
    X_train_num=train[numerical_label]
    X_test_num=test[numerical_label]

    Xtrain=np.hstack((X_train_cat, X_train_num))
    Xtest=np.hstack((X_test_cat,X_test_num))
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
