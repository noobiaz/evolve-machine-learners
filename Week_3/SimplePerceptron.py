import random
import math

w=[0,0]
b=[0]

def create_perceptron():
    w[0] = random.random()
    w[1] = random.random()
    b[0] = random.random()

def activate(x):
    sum= (w[0] * x[0]) + (w[1] * x[1]) + b[0]
    if sum > 0 :
        return 1
    else:
        return -1

def compute_error(target,predicted):
    error = target - predicted
    return error

def learn(error, learning_rate, x):
    w[0] = w[0] + learning_rate*error*x[0]
    w[1] = w[1] + learning_rate*error*x[1]
    b[0] = b[0] + learning_rate*error*1


def one_step_learn(epoch, x_sample, y_sample, learning_rate):
    predicted = activate(x_sample)
    error = compute_error(y_sample, predicted)
    #print('error at iter', epoch, error)
    learn(error, learning_rate, x_sample)

def one_epoch_learn(epoch, x_dataset, y_dataset, learning_rate):
    for i in range(len(x_dataset)):
        rn = random.randint(0, len(x_dataset) - 1)
        x_sample = x_dataset[rn]
        y_sample = y_dataset[rn]
        one_step_learn(i, x_sample, y_sample, learning_rate)


def train(x_dataset, y_dataset, learning_rate, n_iteration):
    for epoch in range (n_iteration):
        one_epoch_learn(epoch, x_dataset, y_dataset, learning_rate)
        total_error=comp_total_error(x_dataset, y_dataset)
        print('total error at epoch', epoch, total_error)
        if total_error<0.1:
            break


def comp_total_error(x_dataset, y_dataset):
    total_error=0.0
    for i in range(len(x_dataset)):
        predict=activate(x_dataset[i])
        error=compute_error(y_dataset[i], predict)
        total_error=total_error+math.fabs(error)

    return total_error



if __name__ == "__main__":

    x_dataset = [[0,0],
                 [0,1],
                 [1,0],
                 [1,1]]

    y_dataset = [-1,1,1,1]
    #y_dataset = [-1,-1,-1,1]

    print('activation before learning')

    for x in x_dataset:
        print(activate(x))


    learning_rate = 0.01

    create_perceptron()
    train(x_dataset, y_dataset, learning_rate, 500)

    print('activation after learning')
    for x in x_dataset:
        print(activate(x))

    #for i in range(100):
    #    rn = random.randint(0, len(x_dataset)-1)
    #    print rn




