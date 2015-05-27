import numpy as np
import pandas as pd
import Orange
from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import LabelEncoder as LE

cols = ['longitude']#, 'latitude', 'num_private', 'population', 'construction_year']

def setNaN(df):
	df[cols] = df[cols].replace(0, np.nan)

def sans(group, sub):
	try: sub.__iter__
	except AttributeError: sub = [sub] #	Convert single objects (incl. strings) to a list
	return list(set(group).difference(set(sub)))

def impute_one(df, colname):
	colnames = df.columns.values
	y = df.loc[:,colname] # y requires imputation
	idc = np.isnan(y)

	X = df.loc[:,sans(colnames, colname)] # X used to predict y
	full = np.all(df.notnull(), axis=1).nonzero()[0] # Rows that do not contain NaN
	train = X.iloc[sans(full, idc)]
	ytr = y.loc[sans(full, idc)] # Known labels
	test  = X.loc[idc]

	model = LR()
	model.fit(train,ytr)
	imput = model.predict(test)
	for i,imp in zip(idc, imput):
		y.loc[i] = imp # Put the predictions into the column

# Info for en- and decoding categories
categorical_labels = ['funder', 'installer', 'wpt_name', 'basin', 'subvillage', 'region', 'lga', 'ward', 'recorded_by', 'scheme_management', 'scheme_name', 'extraction_type', 'extraction_type_group', 'extraction_type_class', 'management_group', 'management', 'payment', 'payment_type', 'water_quality', 'quality_group', 'quantity', 'quantity_group', 'source', 'source_type', 'source_class', 'waterpoint_type', 'waterpoint_type_group']
def encode(df):
	les = {}
	for label in categorical_labels:
		le = LE()
		les.update({label: le})
		df[label] = le.fit_transform(df[label])
	return les

def decode(df, les):
	for label in categorical_labels:
		le = les[label]
		df[label] = le.inverse_transform(df[label])

def impute(df, cols):
	setNaN(df)
	les = encode(df)
	for col in cols:
		impute_one(df, col)
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
