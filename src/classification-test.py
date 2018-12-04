# Simple Python script to perform cross-validation on the TCGA dataset,
# comparing results with the reduced dataset (selected features only)

# by Alejandro Lopez and Alberto Tonda, 2018 <alberto.tonda@gmail.com>

import copy
import numpy as np
import os
import sys

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier

from sklearn.multiclass import OneVsOneClassifier 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import RadiusNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier

# used for normalization
from sklearn.preprocessing import StandardScaler

# used for cross-validation
from sklearn.model_selection import StratifiedKFold

# used to compute accuracy
from sklearn.metrics import accuracy_score

# this is an incredibly useful function
from pandas import read_csv

# local functions
import genericFunctions

def main() :
	
	# a few hard-coded variables, change here if you want to modify random seed or number of folds in the cross-validation
	nFolds = 10
	randomSeed = 42
	
	# here the feature file is selected 
	featureFile = "../results/feature-importance-efs.csv"
	#featureFile = "../results/feature-importance-elastic-net.csv"
	#featureFile = "../results/feature-importance-recursive-feature-elimination-svc.csv"
	#featureFile = "../results/feature-importance-univariate.csv"
	
	# load dataset
	X, y, featureNames = genericFunctions.loadTCGADataset()
	print("Training dataset (original):", X.shape)
	
	# load selected features
	selectedFeatures = genericFunctions.loadFeatures(featureFile)
	
	# create reduced dataset
	print("Reading feature file \"" + featureFile + "\"...")
	featureIndexes = [ i for i in range(0, len(featureNames)) if featureNames[i] in selectedFeatures ]
	X_reduced = X[:,featureIndexes] 
	print("Training dataset (reduced):", X_reduced.shape)
	
	print("Normalizing by samples...")
	normalizeBySample = True
	if normalizeBySample :
		from sklearn.preprocessing import normalize
		X = normalize(X)
		X_reduced = normalize(X_reduced)
	
	# FINALLY, WE CAN CLASSIFY AWAY!
	classifierList = [

			#[RandomForestClassifier(), "RandomForestClassifier()"],
			[BaggingClassifier(n_estimators=300), "BaggingClassifier(n_estimators=300)"],
			[GradientBoostingClassifier(n_estimators=300), "GradientBoostingClassifier(n_estimators=300)"],
			[RandomForestClassifier(n_estimators=300), "RandomForestClassifier(n_estimators=300)"],
			[LogisticRegression(), "LogisticRegression"], # coef_
			[PassiveAggressiveClassifier(), "PassiveAggressiveClassifier"], # coef_
			[RidgeClassifier(), "RidgeClassifier"], # coef_
			[SGDClassifier(), "SGDClassifier"], # coef_
			[SVC(kernel='linear'), "SVC(linear)"], # coef_, but only if the kernel is linear...the default is 'rbf', which is NOT linear

			]
	
	# 10-fold cross-validation
	from sklearn.model_selection import StratifiedKFold 
	skf = StratifiedKFold(n_splits = nFolds, shuffle=True, random_state=randomSeed) 
	foldIndexes = [ (training, test) for training, test in skf.split(X, y) ]

	for originalClassifier, classifierName in classifierList :
		
		classifierPerformance = []
		classifierPerformanceReduced = []

		# iterate over all folds
		print("\nClassifier " + classifierName + " on original dataset...")
		for fold, indexes in enumerate(foldIndexes) :
			
			train_index, test_index = indexes
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			
			# let's normalize by feature
			scaler = StandardScaler()
			X_train = scaler.fit_transform(X_train)
			X_test = scaler.transform(X_test)

			classifier = copy.deepcopy(originalClassifier)
			classifier.fit(X_train, y_train)
			scoreTraining = classifier.score(X_train, y_train)
			scoreTest = classifier.score(X_test, y_test)
			
			print("\tFold #%d: training: %.4f, test: %.4f" % (fold, scoreTraining, scoreTest))
			classifierPerformance.append( scoreTest )
		
		# iterate again over all folds, this time on the reduced dataset
		print("Classifier " + classifierName + " on reduced dataset...")
		for fold, indexes in enumerate(foldIndexes) :

			train_index, test_index = indexes
			X_train, X_test = X_reduced[train_index], X_reduced[test_index]
			y_train, y_test = y[train_index], y[test_index]
			
			# let's normalize by feature
			scaler = StandardScaler()
			X_train = scaler.fit_transform(X_train)
			X_test = scaler.transform(X_test)

			classifier = copy.deepcopy(originalClassifier)
			classifier.fit(X_train, y_train)
			scoreTraining = classifier.score(X_train, y_train)
			scoreTest = classifier.score(X_test, y_test)
			
			print("\tFold %d: training: %.4f, test: %.4f" % (fold, scoreTraining, scoreTest))
			classifierPerformanceReduced.append( scoreTest )
		

		print("Classifier %s, performance on original dataset: %.4f (+/- %.4f)" % (classifierName, np.mean(classifierPerformance), np.std(classifierPerformance)))
		print("Classifier %s, performance on reduced dataset: %.4f (+/- %.4f)" % (classifierName, np.mean(classifierPerformanceReduced), np.std(classifierPerformanceReduced)))

	return


if __name__ == "__main__" :
	sys.exit( main() )
