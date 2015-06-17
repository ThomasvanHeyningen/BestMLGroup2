""" This file defines the pseudolabeling feature.
	To do pseudolabeling, unlabeled data, predicted labels,
	prediction confidence and a threshold are given.
"""
import pandas as pd
import numpy as np
import main

def selectExamples(confidence, labels, threshold=0.95, class_ratios=None):
	""" Expects labels, confidence as np.ndarray. """
	selection = confidence >= threshold
	if class_ratios is not None:
		subselection = preserve_ratios(labels[selection], class_ratios)
	
	j = 0
	for i,b in enumerate(selection):
		if b:
			selection[i] = subselection[j]
			j+=1
	return selection

def preserve_ratios(classes, true_ratios):
	curr_ratios = np.array([np.sum(classes == i) for i in range(np.max(classes))])
	curr_ratios /= len(classes)
	r = np.array(true_ratios) / curr_ratios  # Meta-ratio: ratio between ratios
	r = np.min(r)
	curr_ratios = curr_ratios * r
	max_counts = curr_ratios * len(classes)
	counts = np.zeros_likes(max_counts)
	preserved = []
	for c in classes:
		if counts[c] <= max_counts:
			preserved.append(True)
			counts[c] += 1
		else:
			preserved.append(False)
	return preserved
			

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
	print "EXAMPLES: ", ids, examples
	#SOMEHOW EXAMPLES ARE NOT ADDED TO TRAIN BUT ARE EMPTY
	1/0
	print "PRETRAIN ", train
	train = pd.concat([train, examples])
	print "POSTTRAIN ", train
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
