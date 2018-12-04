# Simple Python script to compute feature importance using univariate statistical analysis, recursive feature elimination, and elastic net
# by Alberto Tonda and Alejandro Lopez, 2018

import numpy as np
import sys

# feature selection methods
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE

# this is a common feature selection method from bio-informatics that exploits ElasticNet
from sklearn.linear_model import ElasticNetCV

# other functions from sklearn
from sklearn.svm import SVC

# these are a few useful functions
from pandas import read_csv # incredibly useful!
from datetime import datetime

# local functions
import genericFunctions

def main() :
	
	# hard-coded constants
	methodologies = dict()
	methodologies["univariate"] = SelectKBest(k=100)
	# WARNING: RFE can take A LOT of time to complete. Be patient (or comment the following line)
	methodologies["recursive-feature-elimination-svc"] = RFE( SVC(kernel='linear'), n_features_to_select=100, verbose=1 )
	methodologies["elastic-net"] = ElasticNetCV()

	featuresEFSFile = "../results/feature-importance-efs.csv"
	
	print("Loading dataset...")
	X, y, biomarkerNames = genericFunctions.loadTCGADataset()
	
	for methodName in methodologies :
		
		start_time = datetime.now() 
		
		print("\nComputing most relevant features using methodology \"" + methodName + "\"...")
		featureSelectionMethod = methodologies[methodName]
		featureSelectionMethod.fit(X, y)
		
		delta_time = datetime.now() - start_time
		
		# create list of tuples
		sortedFeatures = None
		if methodName.find("select-from-model") != -1 or methodName.find("recursive-feature-elimination") != -1 :
			featureIndices = featureSelectionMethod.get_support(indices=True)
			sortedFeatures = [ (1.0, biomarkerNames[i]) for i in featureIndices ]
		elif methodName.find("elastic-net") != -1 :
			coefficients = featureSelectionMethod.coef_
			sortedFeatures = list( zip( list(coefficients), biomarkerNames ) )
		else :
			sortedFeatures = list( zip( list(featureSelectionMethod.scores_), biomarkerNames ) )
		
		# remove all 'nan' values and sort on first element
		sortedFeatures = [ x for x in sortedFeatures if not np.isnan(x[0]) ]
		sortedFeatures = sorted( sortedFeatures, key=lambda x : x[0], reverse=True )
		
		# save everything to file
		outputFile = "feature-importance-" + methodName + ".csv"
		with open(outputFile, "w") as fp :
			for score, feature in sortedFeatures :
				print(feature + ": " + str(score))
				fp.write(feature + "," + str(score) + "\n")
		
		# also, try a comparison with the features obtained through ML
		featuresML = []
		with open(featuresEFSFile, "r") as fp :
			lines = fp.readlines()
			lines.pop(0)
			featuresML = [ lines[i].rstrip().split(',')[0] for i in range(0,100) ]
		
		logFile = "feature-importance-" + methodName + ".log"
		with open(logFile, "w") as fp :
			commonFeatures = 0
			for f in sortedFeatures[:100] :
				if f[1] in featuresML :
					commonFeatures += 1
					string = "Feature \"" + f[1] + "\" is common to both ML and univariate feature selection."
					print(string)
					fp.write(string + "\n")
			string = "\nA total of " + str(commonFeatures) + " features are common to method \"" + methodName + "\" and ensemble ML feature selection." 
			print(string)
			fp.write(string + "\n")
			
			string = "Total time taken by method \"" + methodName + "\": " + str(delta_time)
			print(string)
			fp.write(string + "\n")

	return
	

if __name__ == "__main__" :
	sys.exit( main() )
