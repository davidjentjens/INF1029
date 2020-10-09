#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define NUM_THREADS 5

#define NUMBER_A 2.0f
#define NUMBER_B 5.0f
#define EXPECTED_RESULT 10.0f

#define DELTA 0.000001

struct thread_data {
   long thread_id;
   int offset;
   double *arrayEvens;
   double *arrayOdds;
   double *arrayResult;
};

void * InitializeArraySegment(void *threadarg) {
  struct thread_data *my_data;
  my_data = (struct thread_data*) threadarg;

  printf("Thread #%ld: Initialize array segment with offset of %d\n", my_data->thread_id, my_data->offset);

  double * arrayEvensNext = my_data->arrayEvens;
  double * arrayOddsNext = my_data->arrayOdds;
  double * arrayResultNext = my_data->arrayResult;

  for (arrayResultNext; arrayResultNext < arrayResultNext+my_data->offset; arrayResultNext += sizeof(double)){
    my_data->arrayEvens[i] = NUMBER_A;
    my_data->arrayOdds[i] = NUMBER_B;
    my_data->arrayResult[i] = 0;
  }

  pthread_exit(NULL);
}

void * MultiplyArraySegment(void *threadarg) {
  struct thread_data *my_data;
  my_data = (struct thread_data*) threadarg;

  printf("Thread #%ld: Multiply array segment with offset of %d\n", my_data->thread_id, my_data->offset);

  for (unsigned long int i = 0; i < tamanho; i++){
    my_data->arrayEvens[i] = my_data->arrayEvens[i] * my_data->arrayOdds[i];
  }

  pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
  /* Verifica argumentos */
  int tamanho = atoi(argv[1]);

  if(tamanho % 8 != 0){
    printf("O tamanho deve ser múltiplo 8.\n");
    return 1;
  }

  /* Inicializar variáveis */
  unsigned long int i;
  pthread_t threads[NUM_THREADS];
  pthread_attr_t attr;
  int rc;
  long t;
  struct thread_data thread_data_array[NUM_THREADS];

  /* Inicializar e definir o atributo detached do thread */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  
  /* Aloca os tres arrays em memoria */
  float* arrayEvens = (float*)aligned_alloc(32, tamanho * sizeof(float));
  float* arrayOdds = (float*)aligned_alloc(32, tamanho * sizeof(float));
  float* arrayResult = (float*)aligned_alloc(32, tamanho * sizeof(float));

  if(arrayEvens == NULL || arrayOdds == NULL || arrayResult == NULL){
    printf("Alocação dos arrays não foi feita corretamente\n");
    return 1;
  }
  
  /* Inicializa os tres arrays em memória */
  for(t = 0; t < NUM_THREADS; t++){
    printf("In main: creating thread %ld\n", t);
    thread_data_array[t].thread_id = t;
    thread_data_array[t].offset = t * 8;
    thread_data_array[t].arrayEvens = arrayEvens;
    thread_data_array[t].arrayOdds = arrayOdds;

    rc = pthread_create(&threads[t], NULL, InitializeArraySegment, (void *)&thread_data_array[t]);
    if (rc) {
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }

  /* Executa a multiplicação dos elementos dos arrays: resultado += evens * odds */
  struct timeval start, stop;
  gettimeofday(&start, NULL);

  for (i = 0; i < tamanho; i++){
    arrayResult[i] = arrayEvens[i] * arrayOdds[i];
  }

  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));

  /* Verifica se ocorreu algum erro na multiplicação de algum elemento */
  for (i = 0; i < tamanho; i++){
    if(arrayResult[i] - EXPECTED_RESULT > DELTA){
      printf("A multiplicação não ocorreu corretamente.\n");

      return 1;
    }
  }

  printf("A multiplicação ocorreu corretamente.\n");
  
  /* Libera a memória alocada pelos arrays em memória */
  free(arrayEvens);
  free(arrayOdds);
  free(arrayResult);
  
  return 0;
}

// gcc -pthread -Wall -o array_mult_pthread array_mult_pthread.c