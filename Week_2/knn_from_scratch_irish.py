import csv
import random
import math
import operator


def loadDataset(filename, split):
    trainingSet = []
    testSet = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for l in csvreader:
            for y in range(4):
                l[y] = float(l[y])
            if random.random() < split:
                trainingSet.append(l)
            else:
                testSet.append(l)

    return trainingSet, testSet

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for i in range(length):
        distance += pow((instance1[i] - instance2[i]), 2)
    return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for i in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[i], length)
        distances.append((trainingSet[i], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for i in range(len(neighbors)):
        response = neighbors[i][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def predict(testSet, trainingSet, k):
    predictions = []
    for i in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[i], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('predicted=' + result + ', actual=' + testSet[i][-1])

    return predictions


def main():
    # prepare data
    split = 0.5
    trainingSet, testSet=loadDataset('iris.data', split)
    print('Train set: ' + str(len(trainingSet)))
    print('Test set: ' + str(len(testSet)))
    # generate predictions
    k=3
    predictions = predict(trainingSet, trainingSet,k)
    accuracy_train = getAccuracy(trainingSet, predictions)
    print('Accuracy train: ' + str(accuracy_train) + '%')
    print('============================')
    predictions = predict(testSet, trainingSet,k)
    accuracy_test = getAccuracy(testSet, predictions)
    print('Accuracy test: ' + str(accuracy_test) + '%')


main()