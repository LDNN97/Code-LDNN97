#include<bits/stdc++.h>

using namespace std;

struct edge {
    int v, c;
    edge *ne;
};

edge **nn;
int n, m, s, t;
int *dis;

void link(int u, int v, int c) {
    edge *newedge = new edge;
    newedge->ne = nn[u];
    newedge->v = v;
    newedge->c = c;
    nn[u] = newedge;
}

queue<int> que;

void spfa() {
    bool *bol = new bool[n];
    memset(dis, 0x3f3f3f, sizeof(int) * n);
    memset(bol, 0, sizeof(bol));
    que.push(s);
    dis[s] = 0;
    bol[s] = true;
    while (que.size()) {
        int u = que.front();
        que.pop();
        bol[u] = false;
        for (edge *now = nn[u]; now != NULL; now = now->ne) {
            int v = now->v;
            if (dis[v] >= dis[u] + now->c) {
                dis[v] = dis[u] + now->c;
                if (!bol[v]) {
                    que.push(v);
                    bol[v] = true;
                }
            }
        }
    }
}

int main() {
    fstream file;
    file.open("Test.txt", ios::in | ios::out);
    file >> n >> m;
    file >> s >> t;
    nn = new edge *[n];
    for (int i = 0; i < n; i++) nn[i] = NULL;
    for (int i = 0; i < m; i++) {
        int u, v, c;
        file >> u >> v >> c;
        link(u, v, c);
        link(v, u, c);
    }
    dis = new int[n];
    spfa();
    for (int i = 0; i < n; i++)
        cout << i << " " << dis[i] << endl;
    file.close();
    return 0;
}
