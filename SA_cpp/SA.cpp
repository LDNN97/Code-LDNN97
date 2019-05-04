#include<iostream>
#include<cstdlib>
#include<cstdio>
#include<cmath>
#include<ctime>

using namespace std;

const double eps = 1e-6;
const double pi = acos(-1.0);
const double INF = 1e20;

int T, m;
double X, Y, dis[1100];
struct xy{
	double x, y;
}nn[1100], mm[1100];

double Rnd() {return ((rand() % 10000 + 1) / 10000.0);}

void input() {
	scanf("%lf %lf %d", &X, &Y, &m);
	for (int i = 1; i <= m; i++) 
		scanf("%lf %lf", &nn[i].x, &nn[i].y);
}

double cal_dis(double x1, double y1, double x2, double y2) {
	return (sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)));
}

void sa() {
	for (int i = 1; i <= 30; i++) {
		mm[i].x = Rnd() * X;
		mm[i].y = Rnd() * Y;
		dis[i] = INF;
		for (int j = 1; j <= m; j++) 
			dis[i] = min(dis[i], cal_dis(nn[j].x, nn[j].y, mm[i].x, mm[i].y));
	}
	double delta = 1e6;
	while (delta > eps) {
		for (int i = 1; i <= 30; i++) {
			double tx = mm[i].x, ty = mm[i].y;
			for (int j = 1; j <= 30; j++) {
				double R = Rnd() * 2 * pi;
				double dx = delta * cos(R);
				double dy = delta * sin(R);
				tx += dx; ty += dy;
				if (tx < 0 || tx > X || ty < 0 || ty > Y) continue;
				double tdis = INF;
				for (int k = 1; k <= m; k++) 
					tdis = min(tdis, cal_dis(tx, ty, nn[k].x, nn[k].y));
				if (tdis > dis[i]) {
					dis[i] = tdis;
					mm[i].x = tx; mm[i].y = ty;
				}
			}
		}
		delta *= 0.6;
	}
}

void output() {
	double ans = 0; 
	int ansm;
	for (int i = 1; i <= 30; i++) 
		if (dis[i] > ans) {
			ansm = i;
			ans = dis[i];
		}
	printf("The safest point is (%.1f, %.1f).\n", mm[ansm].x, mm[ansm].y);
}

int main() {
	scanf("%d", &T);
	while (T--) {
		input();
		sa();
		output();
	}
}