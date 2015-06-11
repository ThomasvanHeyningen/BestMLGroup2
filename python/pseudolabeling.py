""" This file defines the pseudolabeling feature.
	To do pseudolabeling, unlabeled data, predicted labels,
	prediction confidence and a threshold are given.
"""
import pandas as pd
import main

def selectExamples(confidence, threshold=0.95):
	""" Expects examples as a df, confidence as np.ndarray. """
	selection = confidence >= threshold
	return selection

def addExamples(labels, ids):
	data_dir = '../data/'
	# Load test set to grab confident examples
	testname = 'testset.csv'
	test = pd.read_csv(data_dir + testname)
	
	# Load train set to add examples to
	trainname = 'trainset.csv'
	train = pd.read_csv(data_dir + trainname)
	print "Train before ", train.shape
	# Add examples to train set
	examples = test[ids, :]
	#SOMEHOW EXAMPLES ARE NOT ADDED TO TRAIN BUT ARE EMPTY
	1/0
	train = pd.concat([train, examples])
	train.to_csv(data_dir + trainname, index_label='id', index=False)
	print "Examples ", examples.shape
	
	# Add labels to label file
	lblname = 'trainlabels.csv'
	labels = pd.DataFrame(labels, index=ids, columns=['status_group'])
	lbls = pd.read_csv(data_dir + lblname).set_index('id')
	lbls = pd.concat([lbls, labels])
	lbls.to_csv(data_dir + lblname, index_label='id', index=True)
	
	print "Train after ", train.shape
	print "Indexes", len(ids)

	# Remove examples from test set
	# Actually, don't
	#test.drop(ids)
	#test.to_csv(data_dir + testname, index_label='id', index=True)

if __name__ == '__main__':
	main.pseudolabel()
