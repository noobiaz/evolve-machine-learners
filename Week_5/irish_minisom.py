from minisom import MiniSom

import numpy as np
import matplotlib.pyplot as plt

def load_irish():
    data = np.genfromtxt('iris.csv', delimiter=',', usecols=(0, 1, 2, 3))
    # data normalization
    data = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, data)
    return data

def create_som(data):
    som = MiniSom(10, 10, 4, sigma=2.0, learning_rate=0.2)
    som.random_weights_init(data)
    return som


def plot_irish(som, data):
    # Plotting the response for each pattern in the iris dataset
    plt.bone()
    plt.pcolor(som.distance_map().T)  # plotting the distance map as background
    plt.colorbar()

    target = np.genfromtxt('iris.csv', delimiter=',', usecols=(4), dtype=str)
    t = np.zeros(len(target), dtype=int)
    t[target == 'setosa'] = 0
    t[target == 'versicolor'] = 1
    t[target == 'virginica'] = 2

    # use different colors and markers for each label
    markers = ['o', 's', 'D']
    colors = ['r', 'g', 'b']
    for cnt, xx in enumerate(data):
        w = som.winner(xx)  # getting the winner
        # palce a marker on the winning position for the sample xx
        plt.plot(w[0] + .5, w[1] + .5, markers[t[cnt]], markerfacecolor='None',
                 markeredgecolor=colors[t[cnt]], markersize=12, markeredgewidth=2)
    plt.axis([0, 10, 0, 10])
    plt.show()


if __name__ == '__main__':
    data=load_irish()
    som=create_som(data)

    print("Training...")
    som.train_random(data, 100)  # random training
    print("\n...ready!")


    plot_irish(som, data)
