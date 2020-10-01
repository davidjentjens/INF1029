#include "matrix_lib.h"

/** Aloca uma matriz com a altura e a largura informadas. */
Matrix * create_matrix(int matrix_height, int matrix_width){
  Matrix * matrix = (Matrix *) aligned_alloc(32, sizeof(int) * 2 + sizeof(float) * (matrix_height*matrix_width));
  
  matrix->height = matrix_height;
  matrix->width = matrix_width;
  matrix->rows = (float *) aligned_alloc(32, matrix_height * matrix_width * sizeof(float));
  
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

/** Multiplica matriz A por matriz B de um valor fornecido de uma forma otimizada. */
int matrix_matrix_mult_otm(Matrix * matrix_a, Matrix * matrix_b, Matrix * matrix_c){

  if(matrix_a == NULL || matrix_b == NULL){
    printf("\nUma ou duas das matrizes não declaradas.\n");
    return 0;
  }

  if(matrix_a->width != matrix_b->height){
    printf("\nA matriz A deve ter o número de colunas igual ao número de linhas da matriz B.\n");
    return 0;
  }

  for(int i = 0; i < matrix_a->height; i++){

    for(int j=0; j < matrix_a->width; j++){
      float position = matrix_a->rows[i * matrix_a->width + j];

       for(int k =0; k < matrix_b->width; k++){
         matrix_c->rows[i * matrix_c->width + k] += (position * matrix_b->rows[j+matrix_b->width + k]);
       }
    }

  }
  

  return 1;
}

/** Multiplica matriz A por matriz B de um valor fornecido de uma forma otimizada, utilizando AVX. */
int matrix_matrix_mult_otm_avx(Matrix * matrix_a, Matrix * matrix_b, Matrix * matrix_c){
  if(matrix_a == NULL || matrix_b == NULL){
    printf("\nUma ou duas das matrizes não declaradas.\n");
    return 0;
  }

  if(matrix_a->width != matrix_b->height){
    printf("\nA matriz A deve ter o número de colunas igual ao número de linhas da matriz B.\n");
    return 0;
  }

  __m256 vectorA, vectorB, vectorC, result;

  float * arrayANext = matrix_a->rows;
  float * arrayBNext = matrix_b->rows;
  float * arrayCNext = matrix_c->rows;

  for(int i = 0; i < matrix_a->height*matrix_a->width; i++, arrayANext++){

    vectorA = _mm256_set1_ps(*arrayANext);
    arrayBNext = matrix_b->rows;

    int row = i / matrix_a->width;
    arrayCNext = matrix_c->rows + row * matrix_b->width;

    for(int k = 0; k < matrix_b->width; k+=8, arrayBNext+=8, arrayCNext+=8){
      vectorB = _mm256_load_ps(arrayBNext);
      vectorC = _mm256_load_ps(arrayCNext);

      result = _mm256_fmadd_ps(vectorA, vectorB, vectorC);
      
      _mm256_store_ps(arrayCNext, result);
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