import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Orange

def df2table(df):
    tdomain = df2domain(df)
    ttables = [series2table(df.icol(i), tdomain[i]) for i in xrange(len(df.columns))]
    return Orange.data.Table(ttables)

def table2df(tab):
    series = [column2df(tab.select(i)) for i in xrange(len(tab.domain))]
    series_name = [i[0] for i in series]  # To keep the order of variables unchanged
    series_data = dict(series)
    print series_data
    return pd.DataFrame(series_data, columns=series_name)

def impute(tab):
	if type(tab) is pd.core.frame.DataFrame:
		tab = df2table(tab)
	imputer = Orange.feature.imputation.ModelConstructor()
	imputer.learner_continuous = Orange.regression.mean.MeanLearner()
	imputer.learner_discrete   = Orange.classification.bayes.NaiveLearner()
	tab = imputer(tab)
	return table2df(tab)

if __name__ == "__main__":
	data = impute(Orange.data.Table('../data/trainset.csv'))
	data.to_csv('../data/trainset.csv')
