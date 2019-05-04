class Node:
    def __init__(self, elem, nex):
        self.elem = elem
        self.next = nex


class List:
    def __init__(self):
        self.head = None

    def append(self, data):
        if self.head is None:
            self.head = Node(data, None)
        else:
            self.head = Node(data, self.head)


if __name__ == "__main__":
    li = List()
    for i in range(5):
        li.append(i)
    tmp = li.head
    while tmp:
        print(tmp.elem)
        tmp = tmp.next
