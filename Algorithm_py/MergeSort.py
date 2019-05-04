import numpy as np
import time


def merge(left, right):
    i, j = 0, 0
    result = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    for k in range(i, len(left)):
        result.append(left[k])
    for k in range(j, len(right)):
        result.append(right[k])
    return result


def merge_sort(lists):
    if len(lists) <= 1:
        return lists
    num = int(len(lists) / 2)
    left = merge_sort(lists[:num])
    right = merge_sort(lists[num:])
    return merge(left, right)


if __name__ == '__main__':
    nn = np.random.randint(1, 10000, 10000)
    start_time = time.clock()
    nn = merge_sort(nn)
    end_time = time.clock()
    print(end_time - start_time)
