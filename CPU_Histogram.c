#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define SIZE 10
#define RANGE 10

double get_clock() {
  struct timeval tv; int ok;
  ok = gettimeofday(&tv, (void *) 0);
  if (ok<0) { printf("gettimeofday error"); }
  return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}


void histogram(int* X, int* histo) {
  for (int i = 0; i < SIZE; i++) {
    histo[X[i]]++;
  }
}


int main() {
  int* X = malloc(sizeof(int) * SIZE);
  for (int i = 0; i < SIZE; i++) {
    X[i] = rand() % RANGE;
  }
  int* histo = malloc(sizeof(int) * RANGE);
  for (int i = 0; i < RANGE; i++) {
    histo[i] = 0;
  }

  double t0 = get_clock();
  histogram(X, histo);
  double t1 = get_clock();

  for (int i = 0; i < RANGE; i++) {
    printf("Number %d: %d\n", i, histo[i]);
  }
  printf("Time: %f ns\n", 1000000000.0*(t1 - t0));
  
  return 0;
}
