#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>



#define NN 200


__global__ void add_cuda_good(int *x,int *y)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  for (int i = 0; i< NN ; i++) {
  y[bid*blockDim.x + tid ] += x[bid*blockDim.x + tid]; 
  y[bid*blockDim.x + tid ] *= 2;
  y[bid*blockDim.x + tid ] += bid*blockDim.x + tid;
  y[bid*blockDim.x + tid ] += 3;
  y[bid*blockDim.x + tid ] += x[bid*blockDim.x + tid]; 
  y[bid*blockDim.x + tid ] *= 2;
  y[bid*blockDim.x + tid ] += bid*blockDim.x + tid;
  y[bid*blockDim.x + tid ] += 3;
  y[bid*blockDim.x + tid ] += x[bid*blockDim.x + tid]; 
  y[bid*blockDim.x + tid ] *= 2;
  y[bid*blockDim.x + tid ] += bid*blockDim.x + tid;
  y[bid*blockDim.x + tid ] += 3;
   }
  
}


void add_cpu_bad(int *x ,int *y, int size)
{
   for (int i=0; i< size; i++){
    for (int i = 0; i< NN; i++) {
	y[i] += x[i];
	y[i] *= 2;
	y[i] += i;
	y[i] += 3;
	y[i] += x[i];
	y[i] *= 2;
	y[i] += i;
	y[i] += 3;
	y[i] += x[i];
	y[i] *= 2;
	y[i] += i;
	y[i] += 3;
     }
   }
}

void print_1D_arr(const char *text,int arr[], int size)
{
   
   if (text == NULL) printf("\n");
   else printf("--%s--\n",text);
   for (int i=0;i<size;i++)
   {
	printf(":%d:",arr[i]);
   }
   printf("\n");
}

int64_t timespecDiff(struct timespec *timeA_p, struct timespec *timeB_p)
{
  return ((timeA_p->tv_sec * 1000000000) + timeA_p->tv_nsec) -
           ((timeB_p->tv_sec * 1000000000) + timeB_p->tv_nsec);
}

int64_t timeDiffSec(struct timespec *timeA_p, struct timespec *timeB_p)
{
  return timeA_p->tv_sec - timeB_p->tv_sec ;
}

void arr_init(int *x,int *y, int N)
{
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 10;
    y[i] = 2;
  }
}


int main(int argc, char** argv)
{

  struct timespec start, end;
  int *x,*y;
  uint64_t timeElapsedGPU;
  uint64_t timeElapsedCPU;
  int N,T;

  sscanf(argv[1] ,"%d", &N);
  sscanf(argv[2], "%d", &T);

clock_gettime(CLOCK_MONOTONIC, &start);
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(int));
  cudaMallocManaged(&y, N*sizeof(int));
 
clock_gettime(CLOCK_MONOTONIC, &end);

  timeElapsedGPU = timespecDiff(&end, &start);
  printf("\n\n\n  timeElapsed for init = %d\n",timeElapsedGPU);

  arr_init(x,y,N);
//  print_1D_arr("CUDA:Input",x,10);
  int blockSize = T;
  int numBlocks = N/blockSize;
  printf(" numBlocks=%d, blockSize=%d\n", numBlocks, blockSize);
clock_gettime(CLOCK_MONOTONIC, &start);  
  add_cuda_good<<<numBlocks, blockSize>>>( x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
clock_gettime(CLOCK_MONOTONIC, &end);

  timeElapsedGPU = timespecDiff(&end, &start);
  printf("\n\n\n  timeElapsed GPU = %d\n",timeElapsedGPU);
  printf("\n  Time Diff Sec GPU = %d\n",timeDiffSec(&end,&start));
//  print_1D_arr("CUDA:Output",y,10);
//  printf("\n\n\n----Final check:%d\n", y[N-1]);

  arr_init(x,y,N);
//  print_1D_arr("CUDA:Input",x,10);

clock_gettime(CLOCK_MONOTONIC, &start);

  // Some code I am interested in measuring 
  add_cpu_bad(x,y,N);

clock_gettime(CLOCK_MONOTONIC, &end);

  timeElapsedCPU = timespecDiff(&end, &start);
  printf("\n\n\n  timeElapsed CPU= %d ratio:%f\n",timeElapsedCPU, (float)timeElapsedCPU/timeElapsedGPU);
  printf("\n  Time Diff Sec CPU = %d\n",timeDiffSec(&end,&start));

  // Free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}
