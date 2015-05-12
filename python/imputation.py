import numpy as np
import pandas as pd
import Orange
from sklearn.preprocessing import Imputer

cols = ['longitude']#, 'latitude', 'num_private', 'population', 'construction_year']

def setNaN(df):
	df[cols] = df[cols].replace(0, np.nan)
	return df

def impute(df):
	print "Setting NaNs"
	df = setNaN(df)
	imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
	print "Imputing"
	df[cols] = imp.fit_transform(df[cols])
	return df

if __name__ == "__main__":
	data = impute(pd.read_csv('../data/trainset.csv'))
	data.to_csv('../data/imp_trainset.csv', index=False)
	data = impute(pd.read_csv('../data/orig_testset.csv'))
	data.to_csv('../data/imp_testset.csv', index=False)
