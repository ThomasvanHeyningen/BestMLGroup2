import numpy as np
import pandas as pd
import Orange
from sklearn.preprocessing import Imputer

def setNaN(df):
	cols = ['longitude']#, 'latitude', 'num_private', 'population', 'construction_year']
	df[cols].replace(0, np.nan)
	return df

def impute(df):
	print "Setting NaNs"
	df = setNaN(df)
	imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
	df = pd.DataFrame(imp.fit_transform(df))
	return df

if __name__ == "__main__":
	data = impute(pd.read_csv('../data/trainset.csv'))
	data.to_csv('../data/imp_trainset.csv')
