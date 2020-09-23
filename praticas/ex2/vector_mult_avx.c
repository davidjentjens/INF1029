#include <immintrin.h>
#include <stdio.h>
#include "timer.h"

#define MINUEND 8.0f
#define SUBTRAHEND 5.0f
#define EXPECTED_RESULT 3.0

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
  float* arrayEvens = (float*)aligned_alloc(32, tamanho * sizeof(float));
  if(arrayEvens == NULL){
    printf("Alocação de arrayEvens não foi feita corretamente\n");
    return 1;
  }
  float* arrayOdds = (float*)aligned_alloc(32, tamanho * sizeof(float));
  if(arrayEvens == NULL){
    printf("Alocação de arrayOdds não foi feita corretamente\n");
    return 1;
  }
  float* arrayResult = (float*)aligned_alloc(32, tamanho * sizeof(float));
  if(arrayEvens == NULL){
    printf("Alocação de arrayResult não foi feita corretamente\n");
    return 1;
  }
  
  /* Inicializa os dois arrays em memória */
  __m256 evens = _mm256_set1_ps(MINUEND);
  __m256 odds = _mm256_set1_ps(SUBTRAHEND);

  float * evensNext = arrayEvens;
  float * oddsNext = arrayOdds;

  for (i = 0; i < tamanho; i+=8, evensNext+=8, oddsNext+=8){
    _mm256_store_ps(evensNext, evens);
    _mm256_store_ps(oddsNext, odds);
  }
  
  /* Executa a subtração dos elementos dos arrays: resultado = evens – odds */
  struct timeval start, stop;
  gettimeofday(&start, NULL);

  __m256 result = _mm256_set1_ps(0.0);

  float * resultsNext = arrayResult;

  evensNext = arrayEvens;
  oddsNext = arrayOdds;

  for (i = 0; i < tamanho; i+=8, evensNext+=8, oddsNext+=8, resultsNext+=8){
    evens = _mm256_load_ps(evensNext);
    odds = _mm256_load_ps(oddsNext);

    result = _mm256_sub_ps(evens, odds);

    _mm256_store_ps(resultsNext, result);
  }

  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));

  /* Verifica se ocorreu algum erro na subtração de algum elemento */
  for (i = 0; i < tamanho; i++){
    if(arrayResult[i] - EXPECTED_RESULT > DELTA){
      printf("A subtração não ocorreu corretamente.\n");

      return 1;
    }
  }

  printf("A subtração ocorreu corretamente.\n");
  
  /* Libera a memória alocada pelos arrays em memória */
  free(arrayEvens);
  free(arrayOdds);
  free(arrayResult);
  
  return 0;
}

// gcc -mavx -std=c11 -o vector_sub_avx vector_sub_avx.c timer.c
