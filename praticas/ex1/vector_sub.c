#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define MINUEND 8.0f
#define SUBTRAHEND 5.0f
#define EXPECTED_RESULT 3.0f

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
  float* arrayEvens = (float*)aligned_alloc(32, tamanho * sizeof(float));
  float* arrayOdds = (float*)aligned_alloc(32, tamanho * sizeof(float));
  float* arrayResult = (float*)aligned_alloc(32, tamanho * sizeof(float));
  
  /* Inicializa os dois arrays em memória */
  for (i = 0; i < tamanho; i += 1){
    arrayEvens[i] = MINUEND;
    arrayOdds[i] = SUBTRAHEND;
  }

  /* Executa a subtração dos elementos dos arrays: resultado = evens – odds */
  struct timeval start, stop;
  gettimeofday(&start, NULL);

  for (i = 0; i < tamanho; i++){
    arrayResult[i] = arrayEvens[i] - arrayOdds[i];
  }

  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));

  /* Verifica se ocorreu algum erro na subtração de algum elemento */
  for (i = 0; i < tamanho; i++){
    if(arrayResult[i] - EXPECTED_RESULT > DELTA){
      printf("A subtração não ocorreu corretamente.\n");

      free(arrayEvens);
      free(arrayOdds);
      free(arrayResult);

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

// gcc -o vector_sub vector_sub.c timer.c
