import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # x = np.arange(10, 110, 10)

    x = np.arange(200, 2200, 200)
    y1 = [3, 4, 6, 9, 10, 10, 10, 10, 10, 11]
    y2 = [10, 19, 24, 33, 35, 35, 36, 40, 42, 48]
    y3 = [0.22, 0.40, 0.77, 1.31, 1.48, 1.74, 2.16, 2.67, 2.71, 2.95]
    y4 = np.arange(0.3, 3.3, 0.3)

    plt.figure()
    # plt.xlabel("Particles")
    # plt.ylabel("Time(s)")
    # plt.ylim(0, 60)
    #
    # plt.plot(x, y1, '-or', label="Particles")
    # plt.plot(x, y2, '-^b', label="Dataset")

    plt.xlabel("Dataset(Million Lines)")
    plt.ylabel("Time(s)")
    plt.ylim(0, 3)
    plt.plot(x, y3, '-or', label="Real")
    plt.plot(x, y4, '-^b', label="Expect")

    plt.legend(loc="lower right")
    plt.show()
