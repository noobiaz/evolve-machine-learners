import csv
import random
import math
import operator
from minist_dataset import generate_training_test, split_into_x_and_y_minist, subsample

def euclideanDistance(instance1, instance2):
    distance = 0
    for i in range(len(instance1)-1):
        distance += pow((instance1[i] - instance2[i]), 2)
    return math.sqrt(distance)

def getAccuracy(y_set, predictions):
    correct = 0
    for i in range(len(y_set)):
        print('predicted = '+str(predictions[i])+'   actual = '+str(y_set[i]))
        if y_set[i] == predictions[i]:
            correct += 1
    return (correct / float(len(y_set))) * 100.0


def getNeighbors(x_train, y_train, x_test_Instance, k):
    distances = []
    for i in range(len(x_train)):
        dist = euclideanDistance(x_test_Instance, x_train[i])
        distances.append((y_train[i], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for response in neighbors:
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def predict(x_test, x_train, y_train, k):
    predictions = []
    for i in range(len(x_test)):
        neighbors = getNeighbors(x_train, y_train, x_test[i], k)
        result = getResponse(neighbors)
        predictions.append(result)

    return predictions


def main_minist():
    # prepare data
    trainingSet, testSet=generate_training_test()
    subsampledtrainset=subsample(trainingSet, 0.01)
    subsampledtestset=subsample(testSet, 0.01)

    x_train, y_train=split_into_x_and_y_minist(subsampledtrainset)
    x_test, y_test=split_into_x_and_y_minist(subsampledtestset)


    print(f'Subsample Train set: {str(len(subsampledtrainset))}')
    print(f'Subsample Test set: {str(len(subsampledtestset))}')
    # generate predictions
    k=3
    predictions = predict(x_train, x_train, y_train, k)
    accuracy_train = getAccuracy(y_train, predictions)
    print('Accuracy train: ' + str(accuracy_train) + '%')
    print('============================')
    predictions = predict(x_test, x_train, y_train, k)
    accuracy_test = getAccuracy(y_test, predictions)
    print('Accuracy test: ' + str(accuracy_test) + '%')
    
main_minist()