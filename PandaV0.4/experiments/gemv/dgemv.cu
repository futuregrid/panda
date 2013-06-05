#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas.h>
#include <sys/time.h>

#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))
typedef float T;
int main(int argc, char **argv)
{
    	int N = 10;
    	int M = 10;

	//struct timeval tim;
        //gettimeofday(&tim,NULL);
        //double t1 = tim.tv_sec+tim.tv_usec/1000000.0;

    	if(argc!=2){
        printf("usage:%s size \n",argv[0]);
        return 0;
    	}//if

    	int size = atoi(argv[1]);
    	N = M = size;
    	T* a = (float *)malloc (M * N * sizeof (*a));
    	if (!a)
	{
	    printf ("host memory allocation failed");
	    return EXIT_FAILURE;
	}

    	T* x = (float *)malloc (N * sizeof (*x));
    	if (!x)
	{
	    printf ("host memory allocation failed");
	    return EXIT_FAILURE;
	}

    	for (int j = 1; j <= N; j++)
	{
	    for (int i = 1; i <= M; i++)
		{
		    a[IDX2F(i,j,M)] = 1;
		}
	}

    	for (int j = 0; j < N; j++)
	{
	    x[j] = 1;
	}

	struct timeval tim;
        gettimeofday(&tim,NULL);
        double t1 = tim.tv_sec+tim.tv_usec/1000000.0;

    	cublasInit();
    	cublasStatus stat;
    	float* devPtrA;
    	stat = cublasAlloc (M*N, sizeof(*a), (void**)&devPtrA);
    	if (stat != CUBLAS_STATUS_SUCCESS)
	{
	    printf ("device memory allocation failed");
	    cublasShutdown();
	    return EXIT_FAILURE;
	}

    	stat = cublasSetVector (M*N, sizeof(*a), a, 1, devPtrA, 1);
    	if (stat != CUBLAS_STATUS_SUCCESS)
	{
	    printf ("data download failed");
	    cublasFree (devPtrA);
	    cublasShutdown();
	    return EXIT_FAILURE;
	}

    	float* devPtrx;
    	stat = cublasAlloc (N, sizeof(*x), (void**)&devPtrx);
    	if (stat != CUBLAS_STATUS_SUCCESS)
	{
	    printf ("device memory allocation failed");
	    cublasShutdown();
	    return EXIT_FAILURE;
	}//if

    	stat = cublasSetVector (N, sizeof(*x), a, 1, devPtrx, 1);
    	if (stat != CUBLAS_STATUS_SUCCESS)
	{
	    printf ("data download failed");
	    cublasFree (devPtrx);
	    cublasShutdown();
	    return EXIT_FAILURE;
	}//if

    	float* devPtry;
    	stat = cublasAlloc (N, sizeof(*devPtry), (void**)&devPtry);
    	if (stat != CUBLAS_STATUS_SUCCESS)
	{
	    printf ("device memory allocation failed");
	    cublasShutdown();
	    return EXIT_FAILURE;
	}//if

	cudaThreadSynchronize();	
	gettimeofday(&tim,NULL);
        double t3 = tim.tv_sec+tim.tv_usec/1000000.0;

	//gettimeofday(&tim,NULL);
	//double t3 = tim.tv_sec+tim.tv_usec/1000000.0;
    	cublasSgemv('n', M, N, 1, devPtrA, M, devPtrx, 1, 0, devPtry, 1);
	//gettimeofday(&tim,NULL);
	//double t4 = tim.tv_sec+tim.tv_usec/1000000.0; 	
	stat = cublasGetError();
    	if ( stat != CUBLAS_STATUS_SUCCESS )
	{
	    printf("dgemv failed\n");
	    return 0;
	}   //stat

    	T* y = (float *)malloc (M * sizeof (*y));
    	if (!y)
	{
	    printf ("host memory allocation failed");
	    return EXIT_FAILURE;
	}//if

    	stat = cublasGetVector (M, sizeof(*y), devPtry, 1, y, 1);
    	if (stat != CUBLAS_STATUS_SUCCESS)
	{
	    printf ("data download failed");
	    cublasFree (devPtry);
	    cublasShutdown();
	    return EXIT_FAILURE;
	}//if

    	//for (int i = 0; i < M; i++)
    	cublasShutdown();
	gettimeofday(&tim,NULL);
	double t2 = tim.tv_sec+tim.tv_usec/1000000.0;
        printf("Time:%f Time:%f Size:%d Mflops:%f  \n", t3-t1,t2-t1,M,2*M/1000.0*M/1000.0/(t3-t1));
    	return 0;
}
