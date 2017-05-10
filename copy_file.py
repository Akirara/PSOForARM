import numpy as np


if __name__ == "__main__":
    origin = np.loadtxt("C:/Users/Zhaoxuan/Desktop/pso_test_data_200.txt", dtype=np.float64, delimiter=',')

    outData = []

    count = 0
    for row in origin:
        count += 1
        for i in range(9):
            print("Row: ", count, "  times: ", i)
            outData.append(row)

    outfile = np.savetxt("C:/Users/Zhaoxuan/Desktop/pso_test_data_1800.txt", outData, fmt='%.2f', delimiter=',')
