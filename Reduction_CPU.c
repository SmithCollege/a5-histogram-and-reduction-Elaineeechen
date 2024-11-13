#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define SIZE 10

double get_clock() {
  struct timeval tv; int ok;
  ok = gettimeofday(&tv, (void *) 0);
  if (ok<0) { printf("gettimeofday error"); }
  return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}


int sum(int* M) {
  int partialSum[SIZE];
  for (int i = 0; i < SIZE; i++) {
    partialSum[i] = M[i];
  }

  for (int stride = 1; stride < SIZE; stride *= 2) {
    for (int i = 0; i < SIZE; i += stride * 2) {
      if (i + stride < SIZE) {
	partialSum[i] += partialSum[i+stride];
      }
    }
  }

  return partialSum[0];
}


int product(int* M) {
  int partialProduct[SIZE];
  for (int i = 0; i < SIZE; i++) {
    partialProduct[i] = M[i];
  }

  for (int stride = 1; stride < SIZE; stride *= 2) {
    for (int i = 0; i < SIZE; i += stride * 2) {
      if (i + stride < SIZE) {
        partialProduct[i] *= partialProduct[i+stride];
      }
    }
  }

  return partialProduct[0];
}


int min(int* M) {
  int temp[SIZE];
  for (int i = 0; i < SIZE; i++) {
    temp[i] = M[i];
  }

  for (int stride = 1; stride < SIZE; stride *= 2) {
    for (int i = 0; i < SIZE; i += stride * 2) {
      if (i + stride < SIZE) {
	if (temp[i+stride] < temp[i]) {
	  temp[i] = temp[i+stride];
	}
      }
    }
  }

  return temp[0];
}


int max(int* M) {
  int temp[SIZE];
  for (int i = 0; i < SIZE; i++) {
    temp[i] = M[i];
  }

  for (int stride = 1; stride < SIZE; stride *= 2) {
    for (int i = 0; i < SIZE; i += stride * 2) {
      if (i + stride < SIZE) {
        if (temp[i+stride] > temp[i]) {
          temp[i] = temp[i+stride];
        }
      }
    }
  }

  return temp[0];
}


int main() {
  int* M = malloc(sizeof(int) * SIZE);
  for (int i = 0; i < SIZE; i++) {
    M[i] = 1;
  }
  
  double t0 = get_clock();
  int result = sum(M);
  double t1 = get_clock();

  printf("Result: %d\n", result);
  printf("Time: %f ns\n", 1000000000.0*(t1 - t0));

  return 0;
}
