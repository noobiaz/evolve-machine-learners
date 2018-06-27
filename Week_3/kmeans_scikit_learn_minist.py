import csv
import random
import math
import operator
from minist_dataset import generate_training_test, split_into_x_and_y_minist, subsample, group_by_label, find_centroid_for_label
import math
from sklearn.cluster import KMeans

def predict(kmeans, data):
    result=kmeans.predict(data)
    return result


def main_minist():
    # prepare data
    train, test=generate_training_test()

    x_train, y_train=split_into_x_and_y_minist(train)
    x_test, y_test=split_into_x_and_y_minist(test)

    k=20
    kmeans = KMeans(n_clusters=k, random_state=1, init='random')
    kmeans.fit(x_train)
    y_train_hat=predict(kmeans, x_train)
    y_test_hat=predict(kmeans, x_test)

    groupbylabel = group_by_label(x_train, y_train)
    map_centerid_to_label=find_centroid_for_label(groupbylabel, kmeans)
    print(map_centerid_to_label)

main_minist()