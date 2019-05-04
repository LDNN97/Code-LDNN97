#include<bits/stdc++.h>

using namespace std;

const int Dimen = 50;
const int PopulationSize = 100;

struct Individual {
    double x[Dimen];
    double fitness;
}Ind[PopulationSize];

double Random(double L, double R) {
    return L + (R - L) * rand() / double(RAND_MAX);
}

double objective(Individual &ind) {
    double fitness = 418.9829 * 50;
    for (int i = 0; i < Dimen; i++) {
        fitness -= ind.x[i] * sin(sqrt(fabs(ind.x[i])));
    }
    return fitness;
}

void initialize() {
    for (int i = 0; i < PopulationSize; i++) {
        for (int j = 0; j < Dimen; j++)
            Ind[i].x[j] = Random(-500, 500);
        Ind[i].fitness = objective(Ind[i]);
    }
}

void DE() {
    double GlobalOpi = 1e20;                                                 //对试验个体测试，选择
    int count = 3000, gen = 0;
    while (gen < count) {                                                           //不断进化
        double CR = Random(0, 1), F = Random(0, 1);
        Individual NewInd[PopulationSize];
        for (int i = 0; i < PopulationSize; i++) {                                         //每个个体变异
            int a, b, c;
            do a = int(Random(0, 1) * PopulationSize); while (a == i);
            do b = int(Random(0, 1) * PopulationSize); while (b == i || b == a);
            do c = int(Random(0, 1) * PopulationSize); while (c == i || c == a || c == b);
            int j = int(Random(0, 1) * Dimen);
            Individual trial;
            for (int k = 0; k < Dimen; k++) {                                   //个体杂交得到试验个体
                if ((Random(0, 1) < CR) || (k == Dimen)) {
                    trial.x[j] = Ind[c].x[j] + F * (Ind[a].x[j] - Ind[b].x[j]);
                    if (trial.x[j] > 500 || trial.x[j] < -500)
                    trial.x[j] = Random(-500, 500);
                }
                else
                    trial.x[j] = Ind[i].x[j];
                j = (j + 1) % Dimen;
            }
            trial.fitness = objective(trial);
            //cout << trial.fitness << endl;
            GlobalOpi = min(GlobalOpi, trial.fitness);
            if (trial.fitness <= Ind[i].fitness) {
//                for (int j = 0; j < Dimen; j++)
//                    NewInd[i].x[j] = trial.x[j];
//                NewInd[i].fitness = trial.fitness;
                memcpy(&NewInd[i], &trial, sizeof(trial));
            } else {
//                for (int j = 0; j < Dimen; j++)
//                    NewInd[i].x[j] = Ind[i].x[j];
//                NewInd[i].fitness = Ind[i].fitness;
                memcpy(&NewInd[i], &Ind[i], sizeof(Ind[i]));
            }
        }
        cout << gen << " " << GlobalOpi << endl;
//        for (int i = 0; i < PopulationSize; i++) {
//            for (int j = 0; j < Dimen; j++)
//                Ind[i].x[j] = NewInd[i].x[j];
//            Ind[i].fitness = NewInd[i].fitness;
//        }
        memcpy(&Ind, &NewInd, sizeof(NewInd));
        gen++;
    }
}

int main() {
    initialize();
    DE();
}
