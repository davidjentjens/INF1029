#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
extern "C" {
#include "timer.h"
}

#define DATASET_SIZE 32

int main(int argc, char **argv)
{
  float *h_x, *h_y;
  float *d_x;
  cudaError_t cudaError;
  int i;
  struct timeval start, stop, overall_t1, overall_t2;

  // Mark overall start time
  gettimeofday(&overall_t1, NULL);

  // Disable buffering entirely
  setbuf(stdout, NULL);

  // Allocating arrays on host
  printf("Allocating arrays h_x e h_y on host...");
  gettimeofday(&start, NULL);

  h_x = (float*)malloc(DATASET_SIZE*sizeof(float));
  h_y = (float*)malloc(DATASET_SIZE*sizeof(float));

  // check malloc memory allocation
  if (h_x == NULL || h_y == NULL) { 
	printf("Error: malloc unable to allocate memory on host.");
        return 1;
  }

  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));

  // Allocating array on device
  printf("Allocating array d_x on device...");
  gettimeofday(&start, NULL);

  cudaError = cudaMalloc(&d_x, DATASET_SIZE*sizeof(float));

  // check cudaMalloc memory allocation
  if (cudaError != cudaSuccess) {
	printf("cudaMalloc d_x returned error %s (code %d)\n", cudaGetErrorString(cudaError), cudaError);
        return 1;
  }

  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));

  // Initialize host memory
  printf("Initializing array h_x on host...");
  gettimeofday(&start, NULL);

  for (i =0; i < DATASET_SIZE; ++i)
	h_x[i] = (float)i;

  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));

  // Copy array from host to device
  printf("Copying array from host (h_x) to device (d_x)...");
  gettimeofday(&start, NULL);

  cudaError = cudaMemcpy(d_x, h_x, DATASET_SIZE*sizeof(float), cudaMemcpyHostToDevice);

  if (cudaError != cudaSuccess) {
	printf("cudaMemcpy (h_x -> d_x) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
        return 1;
  }

  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));

  // Copy array from device to host
  printf("Copying array from device (d_x) to host (h_y)...");
  gettimeofday(&start, NULL);

  cudaError = cudaMemcpy(h_y, d_x, DATASET_SIZE*sizeof(float), cudaMemcpyDeviceToHost);

  if (cudaError != cudaSuccess)
  {
	printf("cudaMemcpy (d_x -> h_y) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
	return 1;
  }

  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));

  // Check for errors (all values should be 0.0f)
  printf("Checking for processing errors...");
  gettimeofday(&start, NULL);

  float maxError = 0.0f;
  float diffError = 0.0f;
  for (i = 0; i < DATASET_SIZE; i++) {
    maxError = (maxError > (diffError=fabs(h_x[i]-h_y[i])))? maxError : diffError;
    //printf("%d -> %f\n", i, h_y[i]);
  }

  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));
  printf("Max error: %f\n", maxError);

  // Free memory
  printf("Freeing memory...");
  gettimeofday(&start, NULL);
  cudaFree(d_x);
  free(h_x);
  free(h_y);
  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));
  
  //Mark overall stop time
  gettimeofday(&overall_t2, NULL);
  // Show elapsed time
  printf("Overall time: %f ms\n", timedifference_msec(overall_t1, overall_t2));
  // Return exit code
  return 0;
}
