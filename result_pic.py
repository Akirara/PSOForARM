import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # x = np.arange(10, 110, 10)

    x = np.arange(2, 22, 2)
    y1 = [3, 4, 6, 8, 9, 9, 10, 10, 10, 11]
    y2 = [10, 19, 24, 33, 35, 35, 36, 40, 42, 48]
    y3 = [0.22, 0.40, 0.77, 1.31, 1.48, 1.74, 2.16, 2.67, 2.71, 2.95]
    y4 = np.arange(0.22, 2.42, 0.22)
    y5 = [15, 25, 34, 40, 49, 56, 61, 69, 80, 89]

    # plt.figure()
    # plt.xlabel("Particle Number")
    # plt.ylabel("Time(sec)")
    # plt.ylim(0, 120)
    #
    # plt.plot(x, y5, '-*k', label="Serial")
    # plt.plot(x, y1, '-or', label="Particle-oriented")
    # plt.plot(x, y2, '-^b', label="Data-oriented")
    # plt.legend()

    plt.xlabel("Dataset Size(million lines)")
    plt.ylabel("Time(sec)")
    plt.ylim(0, 3.5 )
    plt.plot(x, y3, '-or', label="Real")
    plt.plot(x, y4, '--^b', label="Ideal")

    plt.legend()
    # plt.legend(loc="lower right")
    plt.show()
