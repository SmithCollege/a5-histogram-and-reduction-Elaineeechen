#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define SIZE 10
#define BLOCK_SIZE 1024

double get_clock() {
  struct timeval tv; int ok;
  ok = gettimeofday(&tv, (void *) 0);
  if (ok<0) { printf("gettimeofday error"); }
  return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}


__global__ void sum(int* X, int* Y, int n) {
   extern __shared__ int partialSum[];

   unsigned int t = threadIdx.x;
   unsigned int start = 2*blockIdx.x*blockDim.x;
   partialSum[t] = (start + t < n) ? X[start + t] : 0;
   partialSum[blockDim.x+t] = (start + blockDim.x + t < n) ? X[start+ blockDim.x+t] : 0;

   for (unsigned int stride = blockDim.x; stride >= 1; stride >>=1) {
      __syncthreads();
      if (t < stride) {
         partialSum[t] += partialSum[t + stride];
      }
   }

   if (t == 0) {
      Y[blockIdx.x] = partialSum[0];
   }
}


__global__ void product(int* X, int* Y, int n) {
   extern __shared__ int partialProduct[];

   unsigned int t = threadIdx.x;
   unsigned int start = 2*blockIdx.x*blockDim.x;
   partialProduct[t] = (start + t < n) ? X[start + t] : 1;
   partialProduct[blockDim.x+t] = (start + blockDim.x + t < n) ? X[start+ blockDim.x+t] : 1;

   for (unsigned int stride = blockDim.x; stride >= 1; stride >>=1) {
      __syncthreads();
      if (t < stride) {
         partialProduct[t] *= partialProduct[t + stride];
      }
   }

   if (t == 0) {
      Y[blockIdx.x] = partialProduct[0];
   }
}


__global__ void min(int* X, int* Y, int n) {
   extern __shared__ int temp[];

   unsigned int t = threadIdx.x;
   unsigned int start = 2*blockIdx.x*blockDim.x;
   temp[t] = (start + t < n) ? X[start + t] : INT_MAX;
   temp[blockDim.x+t] = (start + blockDim.x + t < n) ? X[start+ blockDim.x+t] : INT_MAX;

   for (unsigned int stride = blockDim.x; stride >= 1; stride >>=1) {
      __syncthreads();
      if (t < stride) {
      	 if (temp[t + stride] < temp[t]) {
	    temp[t] = temp[t + stride];
	 }
      }
   }

   if (t == 0) {
      Y[blockIdx.x] = temp[0];
   }
}


__global__ void max(int* X, int* Y, int n) {
   extern __shared__ int temp[];

   unsigned int t = threadIdx.x;
   unsigned int start = 2*blockIdx.x*blockDim.x;
   temp[t] = (start + t < n) ? X[start + t] : INT_MIN;
   temp[blockDim.x+t] = (start + blockDim.x + t < n) ? X[start+ blockDim.x+t] : INT_MIN;

   for (unsigned int stride = blockDim.x; stride >= 1; stride >>=1) {
      __syncthreads();
      if (t < stride) {
         if (temp[t + stride] > temp[t]) {
            temp[t] = temp[t + stride];
         }
      }
   }

   if (t == 0) {
      Y[blockIdx.x] = temp[0];
   }
}


int main() {
   int *X, *Y;

   cudaMallocManaged(&X, sizeof(int) * SIZE);

   for (int i = 0; i < SIZE; i++) {
       X[i] = i+1;
   }

   int n = SIZE;
   int numBlocks = (n + (2 * BLOCK_SIZE) - 1) / (2 * BLOCK_SIZE);
   cudaMallocManaged(&Y, sizeof(int) * numBlocks);
   int sharedMemSize = 2 * BLOCK_SIZE * sizeof(int);

   double t0 = get_clock();
   while (n > 1) {
       max<<<numBlocks, BLOCK_SIZE, sharedMemSize>>>(X, Y, n);
       cudaDeviceSynchronize();

       n = numBlocks;
       X = Y;
       numBlocks = (n + (2 * BLOCK_SIZE) - 1) / (2 * BLOCK_SIZE);

       if (n > 1) {
           cudaMallocManaged(&Y, sizeof(int) * numBlocks);
       }
   }
   double t1 = get_clock();

   printf("Result: %d\n", X[0]);
   printf("Time: %f ns\n", 1000000000.0*(t1 - t0));   
   printf("%s\n", cudaGetErrorString(cudaGetLastError()));

   return 0;
}
