import time
import copy
import InsertSort
import MergeSort
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x, y1, y2 = [], [], []
    for i in range(1000, 10001, 1000):
        print(i)
        x.append(i)
        arr1 = np.random.randint(1, 10000, i)
        arr2 = copy.deepcopy(arr1)
        st = time.clock()
        InsertSort.insert_sort(arr1)
        ed = time.clock()
        y1.append(ed - st)
        st = time.clock()
        MergeSort.merge_sort(arr2)
        ed = time.clock()
        y2.append(ed - st)
    plt.figure('Comparision')
    plt.plot(x, y1, color='black', linewidth=1.0, linestyle='-', label='Insert Sort')
    plt.plot(x, y2, color='red', linewidth=1.0, linestyle='--', label='Merge Sort')
    plt.xlabel('size of array')
    plt.ylabel('cost time')
    plt.legend(loc='best')
    plt.show()
    file = open('Result.txt', 'w')
    for i in range(len(x)):
        file.write('size:' + str('%d' % x[i]) + ' time1:' + str('%.6f' % y1[i]) + ' time2:' + str('%.6f' % y2[i]) + '\n')
    file.close()


