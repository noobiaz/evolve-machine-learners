from random import seed
from csv_dataset import load_csv, str_column_to_float, str_column_to_int
from csv_dataset import dataset_minmaxmean, normalize_inputdataset
from csv_dataset import split_into_training_test_set, split_into_x_and_y
from sknn.mlp import Classifier, Layer
import datetime

def create_network(niter, lr,
    verboseflag):
    nn = Classifier(
        layers=[
            Layer("Sigmoid", units=5),
            Layer("Softmax")],
        learning_rate=lr,
        n_iter=niter, verbose=verboseflag)
    return nn

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

if __name__ == "__main__":
    seed(datetime.datetime.utcnow())
    filename = 'wheat-seeds.csv'
    dataset = load_csv(filename)
    for i in range(len(dataset[0]) - 1):
        str_column_to_float(dataset, i)
    str_column_to_int(dataset, len(dataset[0]) - 1)
    minmax = dataset_minmaxmean(dataset)
    print('stats dataset', minmax)
    #normalize_inputdataset(dataset, minmax)
    split=0.8
    trainingset, testset = split_into_training_test_set(dataset, split)
    train_x, train_y = split_into_x_and_y(trainingset)
    niter=100
    lr=0.01
    verboseflag=True
    model=create_network(niter, lr, verboseflag)
    model.fit(train_x, train_y)
    predicted_train_y=model.predict(train_x)
    acc_train=accuracy_metric(train_y, predicted_train_y)
    print('accuracy on training set', acc_train)
    test_x, test_y = split_into_x_and_y(testset)
    predicted_test_y=model.predict(test_x)
    acc_test = accuracy_metric(test_y, predicted_test_y)
    print('accuracy on test set', acc_test)

if __name__ == "__main__":
    seed(datetime.datetime.utcnow())
    filename = 'wheat-seeds.csv'
    dataset = load_csv(filename)
    for i in range(len(dataset[0]) - 1):
        str_column_to_float(dataset, i)
    str_column_to_int(dataset, len(dataset[0]) - 1)
    minmax = dataset_minmaxmean(dataset)
    print('stats dataset', minmax)
    normalize_inputdataset(dataset, minmax)
    split=0.8
    trainingset, testset = split_into_training_test_set(dataset, split)
    train_x, train_y = split_into_x_and_y(trainingset)
    niter=100
    lr=0.01
    verboseflag=True
    model=create_network(niter, lr, verboseflag)
    model.fit(train_x, train_y)
    predicted_train_y=model.predict(train_x)
    acc_train=accuracy_metric(train_y, predicted_train_y)
    print('accuracy on training set', acc_train)
    test_x, test_y = split_into_x_and_y(testset)
    predicted_test_y=model.predict(test_x)
    acc_test = accuracy_metric(test_y, predicted_test_y)
    print('accuracy on test set', acc_test)