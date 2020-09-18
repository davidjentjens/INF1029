#include "matrix_lib.h"

/** Aloca uma matriz com a altura e a largura informadas. */
Matrix * create_matrix(int matrix_height, int matrix_width){
  Matrix * matrix = (Matrix *) malloc(sizeof(int) * 2 + sizeof(float) * (matrix_height*matrix_width));
  
  matrix->height = matrix_height;
  matrix->width = matrix_width;
  matrix->rows = (float *) malloc(matrix_height * matrix_width * sizeof(float));
  
  return matrix;
}

/** Preenche matriz com um arquivo .dat fornecido. */
int fill_matrix_with_file(FILE * file, Matrix * matrix){
  if(matrix == NULL){
    return 0;
  }

  int count = 0;
  float * vet = (float*) aligned_alloc(32, (matrix->height * matrix->width) * sizeof(float));
  float vet_aux;

  for(int i = 0; i < matrix->height * matrix->width; i++){
		fread((void*) (&vet_aux), sizeof(vet_aux), 1, file);
		vet[count] = vet_aux;
		count++;
	}

  matrix->rows = vet;

  return 1;
}

/** Preenche matriz com um valor fornecido. */
int fill_matrix(float value, Matrix * matrix){
  if(matrix == NULL){
    printf("\nMatriz não declarada.\n");
    return 0;
  }

  printf("\nPreenchendo matriz...\n");

  for(int i = 0; i < matrix->height; i++){
    for (int j = 0; j < matrix->width; j++){
      matrix->rows[i * matrix->width + j] = value;
    }
  }

  return 1;
}

/** Multiplica matriz por um valor fornecido. */
int scalar_matrix_mult(float scalar_value, Matrix * matrix){
  if(matrix == NULL){
    printf("\nMatriz não declarada.\n");
    return 0;
  }

  printf("\nMultiplicando matriz por %.3f...\n", scalar_value);

  for(int i = 0; i < matrix->height; i++){
    for (int j = 0; j < matrix->width; j++){
      matrix->rows[i * matrix->width + j] *= scalar_value;
    }
  }

  return 1;
}

/** Multiplica matriz A por matriz B de um valor fornecido. */
int matrix_matrix_mult(Matrix * matrix_a, Matrix * matrix_b, Matrix * matrix_c){
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
        float cA = matrix_a->rows[i * matrix_a->width + seqIter];
        float cB = matrix_b->rows[seqIter * matrix_b->width + j];

        equalSeqTotalSum += cA * cB;
      }

      matrix_c->rows[i * matrix_c->width + j] = equalSeqTotalSum;
    }
  }

  return 1;
}

/** Imprime a matriz fornecida */
int matrix_print(Matrix * matrix, char * nome){
  if(matrix == NULL){
    printf("\nMatriz não declarada.\n");
    return 0;
  }

  printf("\nImprimindo matriz %s para o console:\n",nome);


  for(int i = 0; i < matrix->height; i++){
    for (int j = 0; j < matrix->width; j++){
      printf("%.3f, ", matrix->rows[i * matrix->width + j]);
    }
    printf("\n");
  }

  return 1;
}

int write_matrix_to_file(FILE * file, Matrix * matrix){
  if(matrix == NULL){
    printf("\nMatriz não declarada.\n");
    return 0;
  }

  for(int i=0; i<matrix->height*matrix->width; i++){	
 		fwrite((void*)(&matrix->rows[i]), sizeof(matrix->rows[i]), 1, file);
	}

  return 1;
} 