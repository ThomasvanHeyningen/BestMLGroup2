import sys
import time
import re
import numpy as np
import pandas as pd
import datetime as dt

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

def store_data(ids, data_dir, newdata, train=True, labels=('')):
    if train:
        trainfile = pd.read_csv('extratrainfeatures.csv')
        #trainfile=pd.DataFrame(data=ids)  # only needed to generate the file the first time
        trainfile.set_index('id')
        index=0
        for label in labels:
            trainfile[label]=newdata[:,index]
            index=index+1
        trainfile.to_csv('extratrainfeatures.csv', index_label='id', index=False)
    else:
        testfile = pd.read_csv('extratestfeatures.csv')
        #testfile=pd.DataFrame(data=ids) # only needed to generate the file the first time
        testfile.set_index('id')
        index=0
        for label in labels:
            testfile[label]=newdata[:,index]
            index=index+1
        testfile.to_csv('extratestfeatures.csv', index_label='id', index=False)


def main():
    print(" - Start.")
    data_dir='..\\data\\'
    train = pd.read_csv(data_dir + 'trainset.csv')
    trainlabels = pd.read_csv(data_dir + 'trainlabels.csv')
    test = pd.read_csv(data_dir + 'testset.csv')
    newtrain = date_features(train)
    newtest = date_features(test)
    store_data(train['id'],data_dir, newtrain, train=True,labels=('year_recorded','month_recorded','day_recorded','age_of_pump','date_recorded_distance_days_20140101'))
    store_data(test['id'],data_dir, newtest, train=False,labels=('year_recorded','month_recorded','day_recorded','age_of_pump','date_recorded_distance_days_20140101'))
    print(" - Finished.")

if __name__ == '__main__':
    start_time = time.time()
    main()
    #statist()
    print("--- execution took %s seconds ---" % (time.time() - start_time))