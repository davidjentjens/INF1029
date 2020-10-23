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

/* Função que será executada por cada thread do algoritmo de multiplicação de matrizes com pthreads */
void * MultiplyMatrixLine(void * threadarg) {
  unsigned long int i, j, k;
  struct thread_data * my_data;
  my_data = (struct thread_data*) threadarg;

  //printf("Thread #%ld: Multiply array segment\n", my_data->thread_id);

  int segmentEnd = (my_data->size / my_data->num_threads) + my_data->offset;
  if(my_data->last){
    segmentEnd = my_data->size;
  }

  // printThreadData(my_data, segmentEnd, COLOR_YELLOW);

  Matrix * matrix_a = my_data->matrix_A;
  Matrix * matrix_b = my_data->matrix_B;
  Matrix * matrix_c = my_data->matrix_C;

  for(i = 0; i < segmentEnd; i++){
    for(j = 0; j < matrix_a->width; j++){
      float position = matrix_a->rows[i * matrix_a->width + j];

       for(k =0; k < matrix_b->width; k++){
         matrix_c->rows[i * matrix_c->width + k] += (position * matrix_b->rows[j+matrix_b->width + k]);
       }
    }
  }

  pthread_exit(NULL);
}

/** Multiplica matriz A por matriz B de um valor fornecido de uma forma otimizada, utilizando pthreads. */
int matrix_matrix_mult_otm_pthread(Matrix * matrix_a, Matrix * matrix_b, Matrix * matrix_c, int num_threads){
  /* Verifica se as matrizes estão de acordo com as restrições de multiplicação */
  if(matrix_a == NULL || matrix_b == NULL){
    printf("\nUma ou duas das matrizes não declaradas.\n");
    return 0;
  }

  if(matrix_a->width != matrix_b->height){
    printf("\nA matriz A deve ter o número de colunas igual ao número de linhas da matriz B.\n");
    return 0;
  }

  /* Inicializar variáveis */
  pthread_t threads[num_threads];
  pthread_attr_t attr;
  int rc;
  long t;
  struct thread_data thread_data_array[num_threads];
  void *status;
  int size = matrix_a->height;

  if(size % 8*num_threads != 0){
    printf("The array size must be a multiple of (num_threads * 8).\n");
    return 1;
  }

  /* Inicializa e define o atributo detached do thread */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  /* Cria threads para inicializar os tres arrays em memória */
  for(t = 0; t < num_threads; t++){
    // printf("In main: creating thread %ld\n", t);
    thread_data_array[t].thread_id = t;
    thread_data_array[t].size = size;
    thread_data_array[t].offset = (matrix_a->height / num_threads) * t;
    thread_data_array[t].num_threads = num_threads;
    thread_data_array[t].matrix_A = matrix_a;
    thread_data_array[t].matrix_B = matrix_b;
    thread_data_array[t].matrix_C = matrix_c;

    if(t == num_threads-1){
      thread_data_array[t].last = 1;
    }
    else{
      thread_data_array[t].last = 0;
    }

    rc = pthread_create(&threads[t], NULL, MultiplyMatrixLine, (void *)&thread_data_array[t]);
    if (rc) {
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }

  /* Libera atributo e espera pelos outros threads */
  pthread_attr_destroy(&attr);
  for(t = 0; t < num_threads; t++) {
    rc = pthread_join(threads[t], &status);
    if (rc) {
      printf("ERROR; return code from pthread_join() is %d\n", rc);
      exit(-1);
    }
    // printf("Main: completed join with thread %ld having a status of %ld\n",t,(long)status);
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