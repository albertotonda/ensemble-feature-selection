import numpy as np
from pandas import read_csv

def loadTCGADataset() :
	
	# data used for the predictions
	print("Reading data...")
	df = read_csv("../data/tcga_dataset.csv")
	
	# names of the biomarkers
	biomarkers = list(df)
	biomarkers.remove("class")
	
	data = df.as_matrix()
	X = data[:,1:]
	y = data[:,0].ravel() # to have it in the format that the classifiers like

	return X, y, biomarkers 
