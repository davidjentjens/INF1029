#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define NUMBER_A 5.0f
#define NUMBER_B 2.0f
#define NUMBER_C 3.0f
#define EXPECTED_RESULT 13.0f

#define DELTA 0.000001

int main(int argc, char *argv[]) {

  /* Verifica argumentos */
  unsigned long int i;
  int tamanho = atoi(argv[1]);

  if(tamanho % 8 != 0){
    printf("O tamanho deve ser múltiplo 8.\n");
    return 1;
  }
    
  /* Aloca os tres arrays em memoria */
  float* arrayA = (float*)aligned_alloc(32, tamanho * sizeof(float));
  if(arrayA == NULL){
    printf("Alocação de arrayA não foi feita corretamente\n");
    return 1;
  }
  float* arrayB = (float*)aligned_alloc(32, tamanho * sizeof(float));
  if(arrayB == NULL){
    printf("Alocação de arrayB não foi feita corretamente\n");
    return 1;
  }
  float* arrayC = (float*)aligned_alloc(32, tamanho * sizeof(float));
  if(arrayC == NULL){
    printf("Alocação de arrayC não foi feita corretamente\n");
    return 1;
  }
  float* arrayResult = (float*)aligned_alloc(32, tamanho * sizeof(float));
  if(arrayResult == NULL){
    printf("Alocação de arrayResult não foi feita corretamente\n");
    return 1;
  }
  
  /* Inicializa os dois arrays em memória */
  for (i = 0; i < tamanho; i += 1){
    arrayA[i] = NUMBER_A;
    arrayB[i] = NUMBER_B;
    arrayC[i] = NUMBER_C;
  }

  /* Executa a multiplicação dos elementos dos arrays: resultado += evens * odds */
  struct timeval start, stop;
  gettimeofday(&start, NULL);

  for (i = 0; i < tamanho; i++){
    arrayResult[i] = arrayA[i] * arrayB[i] + arrayC[i];
  }

  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));

  /* Verifica se ocorreu algum erro na multiplicação de algum elemento */
  for (i = 0; i < tamanho; i++){
    if(arrayC[i] - EXPECTED_RESULT > DELTA){
      printf("A multiplicação não ocorreu corretamente.\n");

      return 1;
    }
  }

  printf("A multiplicação ocorreu corretamente.\n");
  
  /* Libera a memória alocada pelos arrays em memória */
  free(arrayA);
  free(arrayB);
  free(arrayC);
  free(arrayResult);
  
  return 0;
}

// gcc -o vector_mult vector_mult.c timer.c
