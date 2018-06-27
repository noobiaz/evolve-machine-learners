import random
from base64 import b64decode
from json import loads
import numpy as np
import matplotlib.pyplot as plt
import operator


def parse(x):
    """
    to parse the digits file into tuples of
    (labelled digit, numpy array of vector representation of digit)
    """
    digit = loads(x)
    array = np.fromstring(b64decode(digit["data"]),dtype=np.ubyte)
    array = array.astype(np.float64)
    return (digit["label"], array)


def generate_training_test():
    # read in the digits file. Digits is a list of 60,000 tuples,
    # each containing a labelled digit and its vector representation.
    digits=[]
    with open("digits.base64.json", "r") as f:
        line=f.readline()
        while line:
            digits.append(parse(line))
            line = f.readline()

    # pick a ratio for splitting the digits list into a training and a validation set.
    print('len digits', len(digits))
    ratio = int(len(digits)*0.99)
    test = digits[:ratio]
    training = digits[ratio:]

    return training, test


def subsample(data, take_n_percent):
    upperbound = int(len(data)*take_n_percent)
    result=data[:upperbound]
    return result



def display_digit(digit, labeled = True, title = ""):
    """
    graphically displays a 784x1 vector, representing a digit
    """
    if labeled:
        digit = digit[1]
    image = digit
    plt.figure()
    fig = plt.imshow(image.reshape(28,28))
    fig.set_cmap('gray_r')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    if title != "":
        plt.title("Inferred label: " + str(title))


def split_into_x_and_y_minist(data):
    x=map(lambda item: item[1], data)
    y=map(lambda item: item[0], data)
    return list(x), list(y)

# def group_by_label(x_train, y_train):
#     result={}
#     for i in range(len(x_train)):
#         if y_train[i] in result:
#             result[y_train[i]].append(x_train[i])
#         else:
#             result[y_train[i]]=[x_train[i]]

#     return result


# def find_centroid_for_label(group_by_label_data, kmeans):
#     result={}
#     for label in group_by_label_data:
#         data=group_by_label_data[label]
#         predictions=kmeans.predict(data)
#         most_centroidnumber_for_label={}
#         for p in predictions:
#             if p in most_centroidnumber_for_label:
#                 most_centroidnumber_for_label[p]=most_centroidnumber_for_label[p]+1
#             else:
#                 most_centroidnumber_for_label[p] = 1

#         sortedmost_centroidnumber_for_label=sorted(most_centroidnumber_for_label.iteritems(), key=operator.itemgetter(1), reverse=True)

#         result[sortedmost_centroidnumber_for_label[0][0]]=label

#     return result


# if __name__ == "__main__":
#     train, test=generate_training_test()

#     print('train',len(train))
#     subsampletrain=subsample(train, 0.1)
#     print('subsampletrain',len(subsampletrain))

#     x_train, y_train=split_into_x_and_y_minist(subsampletrain)
#     grp=group_by_label(x_train, y_train)
#     print(grp)

#     testmap={1:10, 2:20, 3:30}
#     for i in testmap:
#         print(i)

