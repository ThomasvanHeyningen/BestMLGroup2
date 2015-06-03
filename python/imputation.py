import numpy as np
import pandas as pd
import Orange
from sklearn.linear_model import LinearRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import LabelEncoder as LE
from scipy.stats import mode

cols = ['num_private', 'latitude', 'population', 'construction_year', 'longitude', 'funder', 'installer', 'subvillage', 'public_meeting', 'scheme_management', 'scheme_name']
continuous = [True,    True,       True,         True,                True,         False,    False,      False,        False,            False,               False]

def setNaN(df):
	pass#df[cols] = df[cols].replace(0, np.nan)

def sans(group, sub):
	try: sub.__iter__
	except AttributeError: sub = [sub] #	Convert single objects (incl. strings) to a list
	return list(set(group).difference(set(sub)))

def impute_cont(df, colname):
	df.loc[:, colname] = df.loc[:, colname].replace(0, np.nan)
	colnames = df.columns.values
	y = df.loc[:,colname] # y requires imputation
	idc = list(np.isnan(y).nonzero()[0])

	X = df.loc[:,sans(colnames, colname)] # X used to predict y
	full = np.all(df.notnull(), axis=1).nonzero()[0] # Rows that do not contain NaN
	train = X.iloc[sans(full, idc)]
	ytr = y.iloc[sans(full, idc)] # Known labels
	test  = X.iloc[idc]

	model = LR()
	model.fit(train,ytr)
	imput = np.array(model.predict(test))
	imput[np.isnan(list(imput))] = np.nanmean(list(imput))
	df.loc[idc, colname] = imput
	return df

def impute_disc(df, colname, encoding):
	nan = encoding.transform(['unknown'])[0]
	colnames = df.columns.values
	y = df.loc[:,colname] # y requires imputation
	idc = list(y==nan)
	
	X = df.loc[:,sans(colnames, colname)] # X used to predict y
	full = np.all(df.notnull(), axis=1).nonzero()[0] # Rows that do not contain NaN
	train = X.iloc[sans(full, idc)]
	ytr = y.iloc[sans(full, idc)] # Known labels
	test  = X.iloc[idc]
	
	model = KNN()
	model.fit(train,ytr)
	imput = np.array(model.predict(test))
	imput[np.isnan(list(imput))] = mode(imput)
	df.loc[idc, colname] = imput
	return df
	

# Info for en- and decoding categories
categorical_labels = ['funder', 'installer', 'wpt_name', 'basin', 'subvillage', 'region', 'lga', 'ward', 'recorded_by', 'scheme_management', 'scheme_name', 'extraction_type', 'extraction_type_group', 'extraction_type_class', 'management_group', 'management', 'payment', 'payment_type', 'water_quality', 'quality_group', 'quantity', 'quantity_group', 'source', 'source_type', 'source_class', 'waterpoint_type', 'waterpoint_type_group', 'public_meeting', 'permit', ]
def encode(df):
	df.loc[:, categorical_labels] = df.loc[:, categorical_labels].fillna('unknown')
	les = {}
	for label in categorical_labels:
		le = LE()
		les.update({label: le})
		df.loc[:,label] = le.fit_transform(df[label])
	return les

def decode(df, les):
	for label in categorical_labels:
		le = les[label]
		df.loc[:,label] = le.inverse_transform(df[label].astype('int32'))

def impute(df, cols):
	setNaN(df)
	les = encode(df)
	for cont, col in zip(continuous, cols):
		if cont:
			df = impute_cont(df, col)
		else:
			df = impute_disc(df, col, les[col])
	decode(df, les)

if __name__ == "__main__":
	print "Starting train set"
	data = pd.read_csv('../data/orig_trainset.csv')
	data = data.drop('date_recorded', axis=1)
	impute(data, cols)
	data.to_csv('../data/trainset.csv', index=False)
	print "Starting test set"
	data = pd.read_csv('../data/orig_testset.csv')
	data = data.drop('date_recorded', axis=1)
	impute(data, cols)
	data.to_csv('../data/testset.csv', index=False)
