//
// Created by LDNN97 on 2018/3/5.
//
#include <iostream>
using namespace std;

class Node {
public:
    Node(int elem = 0, Node * next = NULL) {
        this->elem = elem;
        this->next = next;
    }
    int elem;
    Node * next;
};

class Graph {
public:
    Graph(int num) {
        N = num;
        ans = 0;
        inf = new Node*[num];
        for (int i = 0; i < num; i++)
            inf[i] = NULL;
    }
    void add_edge(int v1, int v2) {
        v1--; v2--;
        inf[v1] = new Node(v2, inf[v1]);
        inf[v2] = new Node(v1, inf[v2]);
    }
    void dfs(int now) {
        Node * ne = inf[now];
        while (ne) {
            if (!vis[ne->elem]) {
                vis[ne->elem] = 1;
                dfs(ne->elem);
            }
            ne = ne->next;
        }
    }

    unsigned int num_connected_components() {
        vis = new bool[N];
        for (int i = 0; i < N; i++)
            vis[i] = 0;
        for (int i = 0; i < N; i++)
            if (!vis[i]) {
                ans++;
                dfs(i);
            }
        return ans;
    }
    int N, ans;
    bool *vis;
    Node ** inf;
};

int main() {
    int M;
    cin>>M;
    while(M--) {
        int N, C;
        cin>>N>>C;

        Graph graph(N);
        for(int i=0;i<C;i++) {
            int v1, v2;
            cin>>v1>>v2;
            graph.add_edge(v1, v2);
        }
        cout << graph.num_connected_components() << endl;

    }
    return 0;
}

