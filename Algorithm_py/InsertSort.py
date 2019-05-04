import numpy as np
import time
import random


def insert_sort(lists):
    for i in range(1, len(lists)):
        key = lists[i]
        j = i - 1
        while j >= 0 and lists[j] > key:
            lists[j + 1] = lists[j]
            j -= 1
        lists[j + 1] = key
    return lists


if __name__ == '__main__':
    nn = []
    for i in range(10000):
        nn.append(random.randint(1, 10000))
    start_time = time.time()
    insert_sort(nn)
    end_time = time.time()
    print(end_time - start_time)
