# Backprop on the Seeds Dataset
from random import seed
import datetime
import random
import numpy as np

from csv import reader

def check_valid_row(row):
    for c in row:
        if len(c.strip())==0:
            return False
    return True

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            if check_valid_row(row):
                dataset.append(row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup


# Find the min, max, mean values for each column
def dataset_minmaxmean(dataset):
	minmax=[]
	for col in range(len(dataset[0])):
		columnvalues=[]
		for row in range(len(dataset)):
			columnvalues.append(dataset[row][col])
		minmax.append([min(columnvalues), max(columnvalues), np.mean(columnvalues)])
	return minmax

# Rescale dataset columns to the range 0-1
def normalize_inputdataset(dataset, minmaxmean):
	for row in dataset:
		for col in range(len(row)-1):
			range_value=(minmaxmean[col][1] - minmaxmean[col][0])
			row[col] = (row[col] - minmaxmean[col][0]) / range_value


def split_into_training_test_set(dataset, split):
	trainingSet = []
	testSet = []
	for d in dataset:
		if random.random() < split:
			trainingSet.append(d)
		else:
			testSet.append(d)

	return trainingSet, testSet


def split_into_x_and_y(data):
    x=map(lambda item: item[:len(item)-1], data)
    y=map(lambda item: [item[len(item)-1]], data)
    return np.array(list(x)), np.array(list(y))


if __name__ == "__main__":
	seed(datetime.datetime.utcnow())
	# load and prepare data
	filename = 'airfoil_self_noise.csv'
	dataset = load_csv(filename)
	for i in range(len(dataset[0])):
		str_column_to_float(dataset, i)
	print('transformed dataset')
	print(dataset[0])
	minmax = dataset_minmaxmean(dataset)
	print('min, max, mean')
	print(minmax)
	normalize_inputdataset(dataset, minmax)
	print('normalize dataset')
	print(dataset[0])
	split=0.8
	trainingset, testset=split_into_training_test_set(dataset, split)
	print('size trainingset', len(trainingset), 'size testset', len(testset))
	train_x, train_y=split_into_x_and_y(trainingset)
	print('train x',train_x[0])
	print('train y',train_y[0])
