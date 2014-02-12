import numpy as np


def f2(a, b):
    f1(a)

@profile
def f1(a):
    a = np.array([1, 0, 1])
    t = 1



def plot_test_pylab():

    plot(np.arange(12))

def plot_test_pyplot():

    import matplotlib.pyplot as plt

    plt.plot(np.arange(12))


if __name__ == '__main__':

    a = 2

    print(a)

    f2(a, 2)

    #plot_test_pylab()
    #plot_test_pyplot()

