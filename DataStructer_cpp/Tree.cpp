//
// Created by LDNN97 on 2018/3/5.
//
#include <iostream>
using namespace std;
class Node {
public:
    Node(int elem, Node *left=NULL, Node *right=NULL) {
        this->elem = elem;
        this->left = left;
        this->right = right;
    }
    int elem;
    Node *left, *right;
};
class Tree {
public:
    Tree() {
        root = NULL;
        cnt = 0;
    }
    void insert(Node * &now, int elem) {
        if (now == NULL) {
            now = new Node(elem);
            return;
        } else {
            if (elem < now->elem)
                insert(now->left, elem);
            else
                insert(now->right, elem);
        }
    }
    void traverse(Node *now, int *result) {
        if (now == NULL) return;
        traverse(now->left, result);
        traverse(now->right, result);
        result[cnt++] = now->elem;
    }
    int cnt;
    Node *root;
};

int main() {
    int M;
    cin>>M;
    while(M--) {
        int N;
        cin>>N;
        Tree tree;
        for(int i=0;i<N;i++) {
            int temp;
            cin>>temp;
            tree.insert(tree.root, temp);
        }
        int *result = new int[N];
        tree.traverse(tree.root, result);
        for(int i=0; i<N; i++)
            cout << result[i] << (i==N-1?'\n':' ');
        delete [] result;
    }
    return 0;
}