#include <immintrin.h>
#include <stdio.h>
#include "timer.h"

#define NUMBER_A 5.0f
#define NUMBER_B 2.0f
#define NUMBER_C 3.0f
#define EXPECTED_RESULT 13.0f

#define DELTA 0.000001

int main(int argc, char *argv[]) {

  /* Verifica argumentos */
  unsigned long int i;
  unsigned long int tamanho = atol(argv[1]);

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
  
  /* Inicializa os três arrays em memória */
  __m256 vectorA = _mm256_set1_ps(NUMBER_A);
  __m256 vectorB = _mm256_set1_ps(NUMBER_B);
  __m256 vectorC = _mm256_set1_ps(NUMBER_C);

  float * vectorANext = arrayA;
  float * vectorBNext = arrayB;
  float * vectorCNext = arrayC;

  for (i = 0; i < tamanho; i+=8, vectorANext+=8, vectorBNext+=8, vectorCNext+=8){
    _mm256_store_ps(vectorANext, vectorA);
    _mm256_store_ps(vectorBNext, vectorB);
    _mm256_store_ps(vectorCNext, vectorC);
  }
  
  /* Executa a multiplicação dos elementos dos arrays: resultado = vectorA * vectorB + vectorC */
  struct timeval start, stop;
  gettimeofday(&start, NULL);

  __m256 result;

  float * resultsNext = arrayResult;

  vectorANext = arrayA;
  vectorBNext = arrayB;
  vectorCNext = arrayC;

  for (i = 0; i < tamanho; i+=8, vectorANext+=8, vectorBNext+=8, vectorCNext+=8, resultsNext+=8){
    vectorA = _mm256_load_ps(vectorANext);
    vectorB = _mm256_load_ps(vectorBNext);
    vectorC = _mm256_load_ps(vectorCNext);

    result = _mm256_fmadd_ps(vectorA, vectorB, vectorC);

    _mm256_store_ps(resultsNext, result);
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
  free(arrayA);
  free(arrayB);
  free(arrayC);
  free(arrayResult);
  
  return 0;
}

// gcc -mfma -std=c11 -o vector_mult_avx vector_mult_avx.c timer.c
