//
// Created by LDNN97 on 2018/3/5.
//
#include <iostream>
using namespace std;
class Node {
public:
    Node(int elem=0, Node *next=NULL) {
        this->elem = elem;
        this->next = next;
    }
    int elem;
    Node *next;
};
class List{
public:
    List() {
        head = NULL;
        tail = NULL;
    }
    void append(int elem) {
        if (head == NULL) {
            head = new Node(elem);
            tail = head;
        } else{
            tail->next = new Node(elem);
            tail = tail->next;
        }
    }
    Node *head, *tail;
};

void merge(List &list1, List &list2, List &merged) {
    Node * l1 = list1.head;
    Node * l2 = list2.head;
    while (l1 != NULL || l2 != NULL) {
        if (l1 == NULL) {
            merged.append(l2->elem);
            l2 = l2->next;
        } else
        if (l2 == NULL) {
            merged.append(l1->elem);
            l1 = l1->next;
        } else
        if (l1->elem <= l2->elem) {
            merged.append(l1->elem);
            l1 = l1->next;
        } else {
            merged.append(l2->elem);
            l2 = l2->next;
        }
    }
}
int main() {
    int M;
    cin>>M;
    while(M--) {
        int N1, N2;
        cin>>N1>>N2;
        //读取输入
        List list1, list2;
        for(int i=0; i<N1; i++) {
            int temp;
            cin>>temp;
            list1.append(temp);
        }
        for(int i=0; i<N2; i++) {
            int temp;
            cin>>temp;
            list2.append(temp);
        }
        //合并两个链表
        List merged;
        merge(list1, list2, merged);
        //输出合并结果
        Node *node = merged.head;
        while(node) {
            cout << node->elem << (node->next?' ':'\n');
            node = node->next;
        }
    }
    return 0;
}

