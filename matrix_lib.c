#include <stdio.h>
#include <stdlib.h>

#include "matrix_lib.h"

Matrix * create_matrix(int matrix_height, int matrix_width)
{
  Matrix * matrix = (Matrix *) malloc(sizeof(int) * (matrix_height + matrix_width) + sizeof(float) * (matrix_height*matrix_width));
  
  matrix->height = matrix_height;
  matrix->width = matrix_width;
  matrix->rows = (float *) malloc(matrix_height * matrix_width * sizeof(float));
  
  return matrix;
}

int fill_matrix(float value, Matrix *matrix)
{
  if(matrix == NULL){
    printf("\nMatriz não declarada.\n");
    return 0;
  }

  printf("\nPreenchendo matriz...\n");

  for(int i = 0; i < matrix->height; i++)
  {
    for (int j = 0; j < matrix->width; j++)
    {
      matrix->rows[i * matrix->height + j] = value;
    }
  }

  return 1;
}

int scalar_matrix_mult(float scalar_value, Matrix *matrix){
  if(matrix == NULL){
    printf("\nMatriz não declarada.\n");
    return 0;
  }

  printf("\nMultiplicando matriz por %.3f...\n", scalar_value);

  for(int i = 0; i < matrix->height; i++){
    for (int j = 0; j < matrix->width; j++){
      matrix->rows[i * matrix->height + j] *= scalar_value;
    }
  }

  return 1;
}

int matrix_matrix_mult(Matrix *matrix_a, Matrix * matrix_b, Matrix * matrix_c){
  if(matrix_a == NULL || matrix_b == NULL){
    printf("\nUma ou duas das matrizes não declaradas.\n");
    return 0;
  }

  if(matrix_a->width != matrix_b->height){
    printf("\nA matriz A deve ter o número de colunas igual ao número de linhas da matriz B.\n");
    return 0;
  }

  printf("\nMultiplicando matriz A por matriz B...\n");

  // Tamanho do somatório de multiplicações de elementos da matriz A com elementos da matriz B
  int equalSeqLen = matrix_a->width;

  for(int i = 0; i < matrix_c->height; i++){
    for (int j = 0; j < matrix_c->width; j++){
      float equalSeqTotalSum = 0;

      // Para cada elemento da matriz C, é somado A[i,seqIter] com B[seqIter,j]
      for(int seqIter = 0; seqIter < equalSeqLen; seqIter++){
        float cA = matrix_a->rows[i * matrix_a->height + seqIter];
        float cB = matrix_b->rows[seqIter * matrix_b->height + j];

        equalSeqTotalSum += cA * cB;
      }

      matrix_c->rows[i * matrix_c->height + j] = equalSeqTotalSum;
    }
  }

  return 1;
}

int matrix_print(Matrix *matrix){
  if(matrix == NULL){
    printf("\nMatriz não declarada.\n");
    return 0;
  }

  printf("\nImprimindo matriz para o console:\n");

  for(int i = 0; i < matrix->height; i++){
    for (int j = 0; j < matrix->width; j++){
      printf("%.3f, ", matrix->rows[i * matrix->height + j]);
    }
    printf("\n");
  }

  return 1;
}