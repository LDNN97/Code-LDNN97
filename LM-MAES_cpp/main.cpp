//
// plain C reference implementation of LM-MA-ES by Ilya Loshchilov 2017-2018
//
// This file is part of the supplementary material for the paper
// "Large Scale Black-box Optimization by Limited-Memory Matrix Adaptation",
// IEEE Transactions on Evolutionary Computation,
// by Ilya Loshchilov, Tobias Glasmachers and Hans-Georg Beyer.
//


#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>


#define ALLOC(TYPE, NUMBER) (TYPE*)(void*)malloc(sizeof(TYPE) * (NUMBER))


typedef struct
/* random_t
* sets up a pseudo random number generator instance
*/
{
	/* Variables for Uniform() */
	long int startseed;
	long int aktseed;
	long int aktrand;
	long int *rgrand;

	/* Variables for Gauss() */
	short flgstored;
	double hold;
} random_t;

typedef struct
{
	double value;
	int id;
} sortedvals;

typedef struct
{
	random_t ttime;
	double*	func_tempdata;
	double*	x_tempdata;
	double*	rotmatrix;
	double* func_shiftxi;
	time_t	time_tic_t;
	clock_t	time_tic_c;
	time_t	time_toc_t;
	clock_t	time_toc_c;
} global_t;

void init_gt(global_t* gt)
{
	gt->func_tempdata = NULL;
	gt->x_tempdata = NULL;
	gt->rotmatrix = NULL;
	gt->func_shiftxi = NULL;
}

void free_gt(global_t* gt)
{
	if (gt->func_tempdata) { free(gt->func_tempdata);  gt->func_tempdata = NULL; }
	if (gt->x_tempdata)    { free(gt->x_tempdata);     gt->x_tempdata = NULL; }
	if (gt->rotmatrix)     { free(gt->rotmatrix);      gt->rotmatrix = NULL; }
	if (gt->func_shiftxi)  { free(gt->func_shiftxi);   gt->func_shiftxi = NULL; }
}

enum FunctionId
{
	fid_Sphere = 1,
	fid_Ellipsoid = 2,
	fid_Rosenbrock = 3,
	fid_Discus = 4,
	fid_Cigar = 5,
	fid_DiffPowers = 6,
	fid_rotEllipsoid = 7,
	fid_rotRosenbrock = 8,
	fid_rotDiscus = 9,
	fid_rotCigar = 10,
	fid_rotDiffPowers = 11,
};


/* random_Start(), random_init(), random_exit(), random_Uniform(), random_Gauss(), time_tic(), time_tictoc(), time_toc() are adopted
from Nikolaus Hansen's source code for CMA-ES
*/

unsigned int m_z = 1;
unsigned int m_w = 123;

unsigned int GetUint()
{
	m_z = 36969 * (m_z & 65535) + (m_z >> 16);
	m_w = 18000 * (m_w & 65535) + (m_w >> 16);
	return (m_z << 16) + m_w;
}

double GetUniform()
{
	// 0 <= u < 2^32
	unsigned int u = GetUint();
	// The magic number below is 1/(2^32 + 2).
	// The result is strictly between 0 and 1.
	return (u + 1.0) * 2.328306435454494e-10;
}

int ibit = 31;
int myrand = 0;
int mask = 1;

int GetRademacherCheap()
{
	//if (ibit == 31)
	{
		mask = 1; myrand = GetUint();
	}
	int res = -1;
	if ((mask&myrand) >= 1)	res = 1;
	mask <<= 1;
	return res;
}

double random_Uniform(random_t *t)
{
	return GetUniform();
	long tmp;

	tmp = t->aktseed / 127773;
	t->aktseed = 16807 * (t->aktseed - tmp * 127773)
		- 2836 * tmp;
	if (t->aktseed < 0)
		t->aktseed += 2147483647;
	tmp = t->aktrand / 67108865;
	t->aktrand = t->rgrand[tmp];
	t->rgrand[tmp] = t->aktseed;
	return (double)(t->aktrand) / (2.147483647e9);
}

/* --------------------------------------------------------- */
double random_Gauss(random_t *t)
{
	double x1, x2, rquad, fac;

	if (t->flgstored)
	{
		t->flgstored = 0;
		return t->hold;
	}
	do
	{
		x1 = 2.0 * random_Uniform(t) - 1.0;
		x2 = 2.0 * random_Uniform(t) - 1.0;
		rquad = x1*x1 + x2*x2;
	} while (rquad >= 1 || rquad <= 0);
	fac = sqrt(-2.0*log(rquad) / rquad);
	t->flgstored = 1;
	t->hold = fac * x1;
	return fac * x2;
}

void	time_tic(global_t* t)
{
	t->time_tic_t = time(NULL);	// measure time in seconds
	t->time_tic_c = clock();	// measure time in microseconds up to ~2k seconds
}

double	time_tictoc(global_t* t)
{
	double dt = difftime(t->time_toc_t, t->time_tic_t);
	if (dt < 1000)
		dt = (double)(t->time_toc_c - t->time_tic_c) / CLOCKS_PER_SEC;
	return dt;
}

double	time_toc(global_t* t)
{
	t->time_toc_t = time(NULL);
	t->time_toc_c = clock();
	return time_tictoc(t);
}

// vector res = matrix a X vector b
void matrix_mult_vector(double* res, double* a, double* b, int m)
{
	double val = 0.0;
	for (int i = 0; i<m; i++)
	{
		val = 0.0;
		for (int j = 0; j<m; j++)
			val += a[i*m + j] * b[j];
		res[i] = val;
	}
}

// matrix res = matrix a X matrix b
void matrix_mult_matrix(double* res, double* a, double* b, int m)
{
	double val;
	for (int i = 0; i<m; i++)
		for (int j = 0; j<m; j++)
		{
			val = 0;
			for (int k = 0; k<m; k++)
				val += a[i*m + k] * b[k*m + j];
			res[i*m + j] = val;
		}
}

void generateRotationMatrix(double* B, int N, double* tmp1, random_t* rnd)
{
	for (int i = 0; i<N; i++)
		for (int j = 0; j<N; j++)
			B[i*N + j] = random_Gauss(rnd);
	for (int i = 0; i<N; i++)
	{
		for (int j = 0; j<i; j++)
		{
			double ariarj = 0;
			for (int k = 0; k<N; k++)
				ariarj = ariarj + B[k*N + i] * B[k*N + j];

			for (int k = 0; k<N; k++)
				B[k*N + i] = B[k*N + i] - ariarj * B[k*N + j];
		}
		double normv = 0;
		for (int k = 0; k<N; k++)
			normv = normv + B[k*N + i] * B[k*N + i];

		normv = sqrt(normv);
		for (int k = 0; k<N; k++)
			B[k*N + i] = B[k*N + i] / normv;
	}
}

double minv(double a, double b)
{
	if (a < b)	return a;
	else		return b;
}

double maxv(double a, double b)
{
	if (a > b)	return a;
	else		return b;
}

double fsphere(double* x, int N)
{
	double Fit = 0;
	for (int i = 0; i<N; i++)
		Fit += x[i] * x[i];
	return Fit;
}

double felli(double* x, int N)
{
	double Fit = 0;
	double alpha = pow(10, 6.0);
	for (int i = 0; i<N; i++)
		Fit += pow(alpha, (double)i / (double)(N - 1)) * x[i] * x[i];
	return Fit;
}

double felli_fast(double* x, int N, global_t* t)
{
	double Fit = 0;
	if (t->func_tempdata == NULL)
	{
		t->func_tempdata = ALLOC(double, N);
		double alpha = pow(10, 6.0);
		for (int i = 0; i<N; i++)
			t->func_tempdata[i] = pow(alpha, (double)i / (double)(N - 1));
	}

	for (int i = 0; i<N; i++)
		Fit += t->func_tempdata[i] * x[i] * x[i];
	return Fit;
}

double fdiscus(double* x, int N)
{
	double Fit = 0;
	Fit = 1e+6 * (x[0] * x[0]);
	for (int i = 1; i<N; i++)
		Fit += x[i] * x[i];
	return Fit;
}

double fcigar(double* x, int N)
{
	double Fit = 0;
	for (int i = 1; i<N; i++)
		Fit += x[i] * x[i];
	Fit = Fit * 1e+6;
	Fit += x[0] * x[0];
	return Fit;
}

double fdiffpowers_fast(double* x, int N, global_t* t)
{
	double Fit = 0;
	if (t->func_tempdata == NULL)
	{
		t->func_tempdata = ALLOC(double, N);
		for (int i = 0; i<N; i++)
			t->func_tempdata[i] = (double)(2.0 + (4.0 * ((double)i) / (N - 1)));
	}
	for (int i = 0; i<N; i++)
		Fit += pow(fabs(x[i]), t->func_tempdata[i]);
	//Fit = sqrt(Fit);
	return Fit;
}

void getRotatedX(double* x, int N, global_t* t)
{
	if (t->x_tempdata == NULL)
		t->x_tempdata = ALLOC(double, N);
	if (t->rotmatrix == NULL)
	{
		t->rotmatrix = ALLOC(double, N*N);
		generateRotationMatrix(t->rotmatrix, N, t->x_tempdata, &t->ttime);
	}
	matrix_mult_vector(t->x_tempdata, t->rotmatrix, x, N);
}

double frosen(double* x, int N)
{
	double Fit = 0;
	double tmp1, tmp2;
	double Fit1 = 0;
	double Fit2 = 0;
	for (int i = 0; i<N - 1; i++)
	{
		tmp1 = x[i] * x[i] - x[i + 1];
		tmp2 = x[i] - 1.0;
		Fit1 += tmp1*tmp1;
		Fit2 += tmp2*tmp2;
	}
	Fit = 100 * Fit1 + Fit2;
	return Fit;
}

double MyFunc(enum FunctionId FuncId, int N, double* x, global_t* t)
{
	double Fit = 0;
	if (FuncId == fid_Sphere)		Fit = fsphere(x, N);
	if (FuncId == fid_Ellipsoid)	Fit = felli_fast(x, N, t);
	if (FuncId == fid_Rosenbrock)	Fit = frosen(x, N);
	if (FuncId == fid_Discus)		Fit = fdiscus(x, N);
	if (FuncId == fid_Cigar)		Fit = fcigar(x, N);
	if (FuncId == fid_DiffPowers)	Fit = fdiffpowers_fast(x, N, t);
	if (FuncId == fid_rotEllipsoid)
	{
		getRotatedX(x, N, t);
		Fit = felli_fast(t->x_tempdata, N, t);
	}
	if (FuncId == fid_rotRosenbrock)
	{
		getRotatedX(x, N, t);
		Fit = frosen(t->x_tempdata, N);
	}
	if (FuncId == fid_rotDiscus)
	{
		getRotatedX(x, N, t);
		Fit = fdiscus(t->x_tempdata, N);
	}
	if (FuncId == fid_rotCigar)
	{
		getRotatedX(x, N, t);
		Fit = fcigar(t->x_tempdata, N);
	}
	if (FuncId == fid_rotDiffPowers)
	{
		getRotatedX(x, N, t);
		Fit = fdiffpowers_fast(t->x_tempdata, N, t);
	}
	return Fit;
}

int compare(const void * a, const void * b)
{
	if      (((sortedvals*)a)->value <  ((sortedvals*)b)->value) return -1;
	else if (((sortedvals*)a)->value == ((sortedvals*)b)->value) return 0;
	else                                                         return 1;
}

void myqsort(int sz, double* arfitness, int* arindex, sortedvals* arr)
{
	for (int i = 0; i<sz; i++)
	{
		arr[i].value = arfitness[i];
		arr[i].id = i;
	}

	qsort(arr, sz, sizeof(sortedvals), compare);
	for (int i = 0; i<sz; i++)
	{
		arfitness[i] = arr[i].value;
		arindex[i] = arr[i].id;
	}
}

// vector res = vector a X matrix b
void vector_mult_matrix(double* res, double* a, double* b, int m)
{
	double val;
	for (int i = 0; i<m; i++)
	{
		val = 0;
		for (int j = 0; j<m; j++)
			val += a[j] * b[j*m + i];
		res[i] = val;
	}
}

double vector_prod(double* a, double* b, int m)
{
	double res = 0.0;
	for (int i = 0; i<m; i++)
		res += a[i] * b[i];
	return res;
}


void LMMAES(int N, enum FunctionId FuncId, int inseed, int printToFile, double* output)
{
	// set boring parameters
	double xmin = -5.0;						// x parameters lower bound
	double xmax = 5.0;						// x parameters upper bound
	int lambda = 4 + 3 * log((double)N);	// population size, e.g., 4+floor(3*log(N));
	int mu = lambda / 2;					// number of parents, e.g., floor(lambda/2);
	double sigma = 0.3 * (xmax - xmin);		// initial step-size
	double target_f = 1e-10;				// target fitness function value, e.g., 1e-10
	double maxevals = 50000 * N;			// maximum number of function evaluations allowed, e.g., 1e+6
	int sample_symmetry = 1;				// 1 or 0, to sample symmetrical solutions to save 50% time and sometimes evaluations
	double minsigma = 1e-20;				// stop if sigma is smaller than minsigma

	// set interesting (to be tuned) parameters
	int sample_type = 0;	 				// 0 - Gaussian, 1 - Rademacher
	int nvectors = 4 + 3 * log((double)N);	// number of stored direction vectors, e.g., nvectors = 4+floor(3*log(N))
	double cs = 2.0*(double)lambda / N;		// nvectors;			
	double damps = 1.0;						// when N is large, a slightly greater value should be used for the Rademacher
	
	// memory allocation
	ibit = 31;								// for the fast Rademacher sampling

	double* arx = ALLOC(double, N*lambda);
	double* arz = ALLOC(double, N*lambda);
	double* ps_arr = ALLOC(double, N*nvectors);
	double* ps = ALLOC(double, N);
	double* xmean = ALLOC(double, N);
	double* zmean = ALLOC(double, N);
	double* xold = ALLOC(double, N);
	double* z = ALLOC(double, N);
	double* dz = ALLOC(double, N);
	double* Az = ALLOC(double, N);
	double* weights = ALLOC(double, mu);
	double* arfitness = ALLOC(double, lambda);
	int* arindex = ALLOC(int, lambda);
	sortedvals* arr_tmp = ALLOC(sortedvals, 2 * lambda);

	global_t gt;
	init_gt(&gt);
	m_z = inseed + 2345;
	m_w = inseed + 1234;

	gt.ttime.flgstored = 0;

	double sum_weights = 0;
	for (int i = 0; i<mu; i++)
	{
		weights[i] = pow(log((double)(mu + 0.5)) - log((double)(1 + i)), 1.0);
		sum_weights = sum_weights + weights[i];
	}
	double mueff = 0;
	for (int i = 0; i<mu; i++)
	{
		weights[i] = weights[i] / sum_weights;
		mueff = mueff + weights[i] * weights[i];
	}
	mueff = 1.0 / mueff;

	for (int i = 0; i < N; i++)
	{
		ps[i] = 0;
		xmean[i] = xmin + (xmax - xmin)*random_Uniform(&gt.ttime);
	}

	for (int i = 0; i < nvectors; i++)
		for (int j = 0; j < N; j++)
			ps_arr[i*N + j] = 0;

	double counteval = 0;
	int iterator_sz = 0;
	int stop = 0;
	int itr = 0;
	double BestF;

	FILE* pFile;
	if (printToFile == 1)
	{
		char filename[250];
		sprintf(filename, "LMMAES%dfunc%d_%d.txt", N, (int)(FuncId), inseed);
		pFile = fopen(filename, "w");
		BestF = MyFunc(FuncId, N, &xmean[0], &gt);
		counteval += 1;
		fprintf(pFile, "%g %g\n", counteval, BestF);
	}

	time_tic(&gt);
	while (stop == 0)
	{
		int sign = 1;
		for (int i = 0; i<lambda; i++) // O(lambda*m*n)
		{
			if (sign == 1)	// if sign==1, then sample new solution, otherwise use its mirror version with sign=-1
			{
				if (sample_type == 0)// || (i == 0)) // Gaussian
					for (int k = 0; k < N; k++)	// O(n)
					{
						z[k] = random_Gauss(&gt.ttime);
						Az[k] = z[k];
					}
				else
					for (int k = 0; k < N; k++)	// O(n)
					{
						z[k] = 1.0*GetRademacherCheap();
						Az[k] = z[k];
					}

				double mcur = iterator_sz;

				for (int k = 0; k< mcur; k++)
				{
					double* ps_j = &ps_arr[k*N];			// its reference
					double c1 = 1.0 / (N * (pow(1.5, (double)k)));
					double ps_j_mult_z = 0;
					for (int p = 0; p < N; p++)				// product vector times Az
						ps_j_mult_z += ps_j[p] * Az[p];
					ps_j_mult_z = ps_j_mult_z * c1;
					for (int p = 0; p < N; p++)
						Az[p] = (1.0 - c1) * Az[p] + ps_j[p] * ps_j_mult_z;
				}
			}

			for (int k = 0; k<N; k++)	// O(n)
			{
				arx[i*N + k] = xmean[k] + sign*sigma*Az[k];
				arz[i*N + k] = sign*z[k];
			}
			if (sample_symmetry) // sample in the opposite direction, seems to work better in most cases AND decreases the CPU cost of the sampling by factor 2
				sign = -sign;

			arfitness[i] = MyFunc(FuncId, N, &arx[i*N], &gt);
			counteval = counteval + 1;
			if (counteval == 1)	BestF = arfitness[i];
			if (arfitness[i] < BestF)	BestF = arfitness[i];
		}

		myqsort(lambda, arfitness, arindex, arr_tmp);

		for (int i = 0; i<N; i++)
		{
			xold[i] = xmean[i];
			xmean[i] = 0;
			zmean[i] = 0;
		}

		for (int i = 0; i<mu; i++)
		{
			double* cur_x = &arx[arindex[i] * N];
			double* cur_z = &arz[arindex[i] * N];
			for (int j = 0; j<N; j++)
			{
				xmean[j] += weights[i] * cur_x[j];
				zmean[j] += weights[i] * cur_z[j];
			}
		}

		for (int i = 0; i < N; i++)
			ps[i] = (1.0 - cs) * ps[i] + sqrt(cs*(2 - cs)*mueff) * zmean[i];

		if (iterator_sz < nvectors)
			iterator_sz = itr + 1;
		for (int i = 0; i < nvectors; i++)
		{
			double lr = ((double)lambda/(double)N) / pow(4.0, (double)i);
			double alpha = (1.0 - lr);
			double beta = sqrt(lr*(2.0 - lr)*mueff);
			double* cur_ps = &ps_arr[i*N ];
			for (int j = 0; j < N; j++)
				cur_ps[j] = alpha * cur_ps[j] + beta * zmean[j];
		}

		double norm_ps = 0;
		for (int i = 0; i<N; i++)
			norm_ps += ps[i] * ps[i];

		sigma = sigma * exp((cs / 1.0)*(norm_ps / N - 1.0));

		if (arfitness[0] < target_f)
			stop = 1;
		if (counteval >= maxevals)
			stop = 1;
		itr = itr + 1;

		if (sigma < minsigma)
			stop = 1;
		if ((printToFile == 1) && (pFile) && ((itr % 1000 == 0) || (counteval < 10000)))
		{
			fprintf(pFile, "%g %g %g\n", counteval, BestF, time_toc(&gt));
			fflush(pFile);
		}
	}

	if ((printToFile == 1) && (pFile))
	{
		fprintf(pFile, "%g %g %g\n", counteval, BestF, time_toc(&gt));
		fclose(pFile);
	}

	output[0] = BestF;
	output[1] = counteval;
	output[2] = time_toc(&gt);

	free_gt(&gt);
	free(arr_tmp);   free(weights); free(ps);  free(xmean);
	free(zmean);     free(xold);    free(z);   free(Az);	
	free(dz);        free(ps_arr);  free(arx); free(arz);	
	free(arfitness); free(arindex);
}

void MAES(int N, enum FunctionId FuncId, int inseed, int printToFile, double* output, int fast)
{
	// set boring parameters
	double xmin = -5.0;						//	x parameters lower bound
	double xmax = 5.0;						//	x parameters upper bound
	int lambda = 4 + 3 * log((double)N);	// 	population size, e.g., 4+floor(3*log(N));
	int mu = lambda / 2;					// 	number of parents, e.g., floor(lambda/2);
	double sigma = 0.3 * (xmax - xmin);		// initial step-size
	double target_f = 1e-10;				// target fitness function value, e.g., 1e-10
	double maxevals = 50000 * N;			// maximum number of function evaluations allowed, e.g., 1e+6
	int sample_symmetry = 1;				// 1 or 0, to sample symmetrical solutions to save 50% time and sometimes evaluations
	double minsigma = 1e-20;				// stop if sigma is smaller than minsigma

	// set interesting (to be tuned) parameters
	int sample_type = 0;	 				// 0 - Gaussian, 1 - Rademacher
	// memory allocation
	ibit = 31;								// for the fast Rademacher sampling

	double* M = ALLOC(double, N * N);

	double* arx = ALLOC(double, N*lambda);
	double* arz = ALLOC(double, N*lambda);
	double* ard = ALLOC(double, N*lambda);
	double* ps = ALLOC(double, N);
	double* xmean = ALLOC(double, N);
	double* zmean = ALLOC(double, N);
	double* Mz = ALLOC(double, N);
	double* z = ALLOC(double, N);
	double* weights = ALLOC(double, mu);
	double* arfitness = ALLOC(double, lambda);
	int* arindex = ALLOC(int, lambda);
	sortedvals* arr_tmp = ALLOC(sortedvals, 2 * lambda);

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			if (i == j)	M[i*N + j] = 1.0;
			else		M[i*N + j] = 0.0;

	global_t gt;
	init_gt(&gt);
	m_z = inseed + 2345;
	m_w = inseed + 1234;

	gt.ttime.flgstored = 0;

	double sum_weights = 0;
	for (int i = 0; i<mu; i++)
	{
		weights[i] = pow(log((double)(mu + 0.5)) - log((double)(1 + i)), 1.0);
		sum_weights = sum_weights + weights[i];
	}
	
	double mueff = 0;
	for (int i = 0; i<mu; i++)
	{
		weights[i] = weights[i] / sum_weights;
		mueff = mueff + weights[i] * weights[i];
	}
	mueff = 1.0 / mueff;

	for (int i = 0; i < N; i++)
	{
		ps[i] = 0;
		xmean[i] = xmin + (xmax - xmin)*random_Uniform(&gt.ttime);
	}


	double cs = (mueff + 2.0) / (N + mueff + 5.0);
	double lcoef = 2.0;
	double c1 = lcoef / (pow(N + 1.3, 2.0) + mueff);
	double cmu = minv(1.0 - c1, lcoef * (mueff - 2.0 + 1.0 / mueff) / (pow(N + 2.0, 2.0) + mueff));
	
	double counteval = 0;
	int stop = 0;
	int itr = 0;
	double BestF;

	FILE* pFile;
	if (printToFile == 1)
	{
		char filename[250];
		sprintf(filename, "MAES%dfunc%d_%d.txt", N, (int)(FuncId), inseed);
		pFile = fopen(filename, "w");
		BestF = MyFunc(FuncId, N, &xmean[0], &gt);
		counteval += 1;
		fprintf(pFile, "%g %g\n", counteval, BestF);
	}

	time_tic(&gt);
	while (stop == 0)
	{
		int sign = 1;
		for (int i = 0; i < lambda; i++) // O(lambda*m*n)
		{
			if (sign == 1)	// if sign==1, then sample new solution, otherwise use its mirrored version with sign=-1
			{
				if (sample_type == 0)// || (i == 0)) // Gaussian
					for (int k = 0; k < N; k++)	// O(n)
						z[k] = random_Gauss(&gt.ttime);
				else
					for (int k = 0; k < N; k++)	// O(n)
						z[k] = 1.0*GetRademacherCheap();
				matrix_mult_vector(Mz, M, z, N);
			}

			for (int k = 0; k < N; k++)	// O(n)
			{
				arz[i*N + k] = sign * z[k];
				ard[i*N + k] = sign * Mz[k];
				arx[i*N + k] = xmean[k] + sigma * ard[i*N + k];
			}
			if (sample_symmetry) // sample in the opposite direction, seems to work better in most cases AND decreases the CPU cost of the sampling by 2.0
				sign = -sign;

			arfitness[i] = MyFunc(FuncId, N, &arx[i*N], &gt);
			counteval = counteval + 1;
			if (counteval == 1)	BestF = arfitness[i];
			if (arfitness[i] < BestF)	BestF = arfitness[i];
			//if (int(counteval) % 100000 == 0)
			//printf("%g\t%g\n", counteval, BestF);
		}

		myqsort(lambda, arfitness, arindex, arr_tmp);

		for (int i = 0; i < mu; i++)
		{
			double* cur_z = &arz[arindex[i] * N];
			for (int j = 0; j < N; j++)
				if (i == 0)	zmean[j] = weights[i] * cur_z[j];
				else		zmean[j] += weights[i] * cur_z[j];

			double* cur_d = &ard[arindex[i] * N];
			for (int j = 0; j < N; j++)
				xmean[j] += sigma * weights[i] * cur_d[j];
		}

		for (int i = 0; i < N; i++)
			ps[i] = (1.0 - cs) * ps[i] + sqrt(cs*(2 - cs)*mueff) * zmean[i];

		if (fast == 1) 
		{
			matrix_mult_vector(Mz, M, ps, N);
			
			double coef1 = (1.0 - 0.5*c1 - 0.5*cmu);
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
				{
					double val = 0;
					for (int k = 0; k < mu; k++)
						val += weights[k] * ard[arindex[k] * N + i];
					M[i*N + j] *= coef1;
					M[i*N + j] += 0.5 * ps[j] * (c1 * Mz[i] + cmu * val);
				}
		}
		else
		{
/*
			// don't forget to allocate Mtmp
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
				{
					double sumk = 0;
					for (int k = 0; k < mu; k++)
						sumk += weights[k] * arz[arindex[k] * N + i] * arz[arindex[k] * N + j];
					double baseval = 0;
					if (i == j)	baseval = 1;
					Mmult[i*N + j] = baseval + c1 / 2.0 * (ps[i] * ps[j] - baseval) + cmu / 2.0 * (sumk - baseval);
				}

			matrix_mult_matrix(Mtmp, M, Mmult, N);
			for (int i = 0; i < N*N; i++)
				M[i] = Mtmp[i];
*/		}

		double norm_ps = 0;
		for (int i = 0; i<N; i++)
			norm_ps += ps[i] * ps[i];

		sigma = sigma * exp( cs * (norm_ps / N - 1));

		if (arfitness[0] < target_f)
			stop = 1;
		if (counteval >= maxevals)
			stop = 1;
		itr = itr + 1;

		if (sigma < minsigma)
			stop = 1;
		if ((printToFile == 1) && (pFile) && ((itr % 1000 == 0) || (counteval < 10000)))
		{
			fprintf(pFile, "%g %g %g\n", counteval, BestF, time_toc(&gt));
			fflush(pFile);
		}
	}

	if ((printToFile == 1) && (pFile))
	{
		fprintf(pFile, "%g %g %g\n", counteval, BestF, time_toc(&gt));
		fclose(pFile);
	}

	output[0] = BestF;
	output[1] = counteval;
	output[2] = time_toc(&gt);

	free_gt(&gt);
	free(arr_tmp);   free(weights); free(ps);  free(xmean);
	free(zmean);     free(z);       free(Mz);	
	free(arx);       free(arz);     free(ard); free(arfitness);	
	free(M);         free(arindex);
}

int main(void)
{
	// the code given below will generate text files containing results for 5 runs of
	// LM-CMA-ES and fast-MA-ES on six 128D .. 8192D (respectively, 128D .. 1024D) 
	// benchmark functions 
	for (int seed = 1; seed <= 5; seed++)				// runs / seeds
		for (int ialgo = 1; ialgo <= 1; ialgo++)		// algorithms
		{
			int Nmax = 8192;
			if (ialgo == 2)	Nmax = 1024;
			//Nmax = 128;
			for (int N = 128; N <= Nmax; N *= 2)		// problem dimensions
			{
				for (int i = 1; i <= 6; i++)			// problems
				{
					// FunctionId FuncId = fid_Ellipsoid;
					// fid_Sphere = 1, fid_Ellipsoid = 2, fid_Rosenbrock = 3, fid_Discus = 4, fid_Cigar = 5, fid_DiffPowers = 6
					// fid_DiffPowers = 6, fid_rotEllipsoid = 7, fid_rotRosenbrock = 8, fid_rotDiscus = 9, fid_rotCigar = 10, fid_rotDiffPowers = 11,
					enum FunctionId FuncId = (enum FunctionId)i;
					int printToFile = 1;			// 1 or 0
					double output[3];				// output[0] = bestFit, output[1] = nevaluations, output[2] = timecostinms
					if (ialgo == 1)		LMMAES(N, FuncId, seed, printToFile, (double*)output);
					if (ialgo == 2)		MAES(N, FuncId, seed, printToFile, (double*)output, 1);
					double evaltimecost = output[2] / output[1];
					double nevals = (output[1] / (double)N);
					printf("iAlgo:%d\tDim:%d\tFuncId:%d\tbestFit:%g\tnevals/dim:%g\tevaltimecost (microsec):%g\n",
						ialgo, N, FuncId, output[0], nevals, evaltimecost*1e+6);
				}
			}
		}
	return 0;
}
