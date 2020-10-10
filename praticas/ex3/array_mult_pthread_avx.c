#include <immintrin.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define COLOR_CYAN "\033[0;36m"
#define COLOR_YELLOW "\033[1;33m"
#define COLOR_RESET "\033[0m"

#define NUM_THREADS 5

#define NUMBER_A 2.0f
#define NUMBER_B 5.0f
#define EXPECTED_RESULT 10.0f

#define DELTA 0.000001

struct thread_data {
   long thread_id;
   int offset;
   int size;
   int last;
   float * arrayEvens;
   float * arrayOdds;
   float * arrayResult;
};

void printThreadData(struct thread_data * my_data, int segmentEnd, char * color){
  printf("%s-----------------\n", color);
  printf("| Thread #%ld     \n", my_data->thread_id);
  printf("| last: %d       \n", my_data->last);
  printf("| offset: %d     \n", my_data->offset); 
  printf("| segmentEnd: %d \n", segmentEnd);
  printf("| size: %d       \n", my_data->size);
  printf("-----------------%s\n", COLOR_RESET);
}

void * InitializeArraySegment(void * threadarg) {
  struct thread_data * my_data;
  my_data = (struct thread_data*) threadarg;

  printf("Thread #%ld: Initialize array segment with AVX:\n", my_data->thread_id);

  int segmentEnd = (my_data->size / NUM_THREADS) + my_data->offset;
  if(my_data->last){
    segmentEnd = my_data->size;
  }

  // printThreadData(my_data, segmentEnd, COLOR_CYAN);

  /* Inicializa os três vetores em memória */
  __m256 vectorEvens = _mm256_set1_ps(NUMBER_A);
  __m256 vectorOdds = _mm256_set1_ps(NUMBER_B);
  __m256 vectorResult = _mm256_set1_ps(0.0);

  float * arrayEvensNext = &my_data->arrayEvens[my_data->offset];
  float * arrayOddsNext = &my_data->arrayOdds[my_data->offset];
  float * arrayResultNext = &my_data->arrayResult[my_data->offset];

  for (int i = my_data->offset; i < segmentEnd; i+=8, arrayEvensNext+=8, arrayOddsNext+=8, arrayResultNext+=8){
    _mm256_store_ps(arrayEvensNext, vectorEvens);
    _mm256_store_ps(arrayOddsNext, vectorOdds);
    _mm256_store_ps(arrayResultNext, vectorResult);
  }

  pthread_exit(NULL);
}

void * MultiplyArraySegment(void * threadarg) {
  struct thread_data * my_data;
  my_data = (struct thread_data*) threadarg;

  printf("Thread #%ld: Multiply array segment with AVX\n", my_data->thread_id);

  int segmentEnd = (my_data->size / NUM_THREADS) + my_data->offset;
  if(my_data->last){
    segmentEnd = my_data->size;
  }

  /* Declara vetores AVX para uso a seguir */
  __m256 vectorEvens, vectorOdds, vectorResult;

  // printThreadData(my_data, segmentEnd, COLOR_YELLOW);

  float * arrayEvensNext = &my_data->arrayEvens[my_data->offset];
  float * arrayOddsNext = &my_data->arrayOdds[my_data->offset];
  float * arrayResultNext = &my_data->arrayResult[my_data->offset];

  for (int i = my_data->offset; i < segmentEnd; i+=8, arrayEvensNext+=8, arrayOddsNext+=8, arrayResultNext+=8){
    vectorEvens = _mm256_load_ps(arrayEvensNext);
    vectorOdds = _mm256_load_ps(arrayOddsNext);

    vectorResult = _mm256_mul_ps(vectorEvens, vectorOdds);

    _mm256_store_ps(arrayResultNext, vectorResult);
  }

  pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
  /* Verifica argumentos */
  if(argc <= 1){
    printf("The size of the arrays must be passed as a parameter.\n");
    return 1;
  }

  int size = atoi(argv[1]);

  if(size < 8*NUM_THREADS || size % 8*NUM_THREADS != 0){
    printf("The array size must be a multiple of (NUM_THREADS * 8).\n");
    return 1;
  }

  /* Inicializar variáveis */
  unsigned long int i;
  pthread_t threads[NUM_THREADS];
  pthread_attr_t attr;
  int rc;
  long t;
  struct thread_data thread_data_array[NUM_THREADS];
  void *status;

  /* Inicializa e define o atributo detached do thread */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  
  /* Aloca os tres arrays em memoria */
  float* arrayEvens = (float*)aligned_alloc(32, size * sizeof(float));
  float* arrayOdds = (float*)aligned_alloc(32, size * sizeof(float));
  float* arrayResult = (float*)aligned_alloc(32, size * sizeof(float));

  if(arrayEvens == NULL || arrayOdds == NULL || arrayResult == NULL){
    printf("Error allocating arrays.\n");
    return 1;
  }

  /* Inicializa timer */
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  
  /* Cria threads para inicializar os tres arrays em memória */
  for(t = 0; t < NUM_THREADS; t++){
    printf("In main: creating thread %ld\n", t);
    thread_data_array[t].thread_id = t;
    thread_data_array[t].size = size;
    thread_data_array[t].offset = (size / NUM_THREADS) * t;
    thread_data_array[t].arrayEvens = arrayEvens;
    thread_data_array[t].arrayOdds = arrayOdds;
    thread_data_array[t].arrayResult = arrayResult;

    if(t == NUM_THREADS-1){
      thread_data_array[t].last = 1;
    }
    else{
      thread_data_array[t].last = 0;
    }

    rc = pthread_create(&threads[t], NULL, InitializeArraySegment, (void *)&thread_data_array[t]);
    if (rc) {
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }

  /* Cria threads para executar a multiplicação dos elementos dos arrays */
  for(t = 0; t < NUM_THREADS; t++){
    printf("In main: creating thread %ld\n", t);
    thread_data_array[t].thread_id = t;
    thread_data_array[t].size = size;
    thread_data_array[t].offset = (size / NUM_THREADS) * t;
    thread_data_array[t].arrayEvens = arrayEvens;
    thread_data_array[t].arrayOdds = arrayOdds;
    thread_data_array[t].arrayResult = arrayResult;

    if(t == NUM_THREADS-1){
      thread_data_array[t].last = 1;
    }
    else{
      thread_data_array[t].last = 0;
    }
    

    rc = pthread_create(&threads[t], NULL, MultiplyArraySegment, (void *)&thread_data_array[t]);
    if (rc) {
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }

  /* Libera atributo e espera pelos outros threads */
  pthread_attr_destroy(&attr);
  for(t = 0; t < NUM_THREADS; t++) {
    rc = pthread_join(threads[t], &status);
    if (rc) {
      printf("ERROR; return code from pthread_join() is %d\n", rc);
      exit(-1);
    }
    printf("Main: completed join with thread %ld having a status of %ld\n",t,(long)status);
  }

  /* Para o timer */
  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));

  /* Verifica se ocorreu algum erro na multiplicação de algum elemento */
  for (i = 0; i < size; i++){
    // printf("%f\n", arrayResult[i]);
    if(arrayResult[i] - EXPECTED_RESULT > DELTA){
      printf("Multiplication did not occur correctly.\n");
      return 1;
    }
  }
  printf("The multiplication occurred correctly.\n");
  
  /* Libera a memória alocada pelos arrays e termina a execução */
  free(arrayEvens);
  free(arrayOdds);
  free(arrayResult);

  pthread_exit(NULL);
  
  return 0;
}

// gcc -mfma -std=c11 -pthread -Wall -o array_mult_pthread_avx array_mult_pthread_avx.c timer.c