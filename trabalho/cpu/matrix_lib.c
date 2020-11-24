#include "matrix_lib.h"

#define AVX_STEP 8

#define COLOR_CYAN "\033[0;36m"
#define COLOR_YELLOW "\033[1;33m"
#define COLOR_RESET "\033[0m"

static int thread_num;

/** Determina o número de threads a serem utilizados nas funções que usam threads */
void set_number_threads(int num_threads){
  thread_num = num_threads;
}

/** Função para imprimir os dados de uma thread de multiplicação de matriz por escalar*/
void printThreadScalarData(struct scalar_thread_data * my_data, int segmentEnd, char * color){
  printf("%s-----------------\n", color);
  printf("| Thread #%ld     \n", my_data->thread_id);
  printf("| last: %d       \n", my_data->last);
  printf("| offset: %d     \n", my_data->offset); 
  printf("| segmentEnd: %d \n", segmentEnd);
  printf("| size: %d       \n", my_data->size);
  printf("-----------------%s\n", COLOR_RESET);
}

/** Função para imprimir os dados de uma thread de multiplicação de matrizes */
void printThreadMatrixData(struct matrix_thread_data * my_data, int segmentEnd, char * color){
  printf("%s-----------------\n", color);
  printf("| Thread #%ld     \n", my_data->thread_id);
  printf("| last: %d       \n", my_data->last);
  printf("| offset: %d     \n", my_data->offset); 
  printf("| segmentEnd: %d \n", segmentEnd);
  printf("| size: %d       \n", my_data->size);
  printf("-----------------%s\n", COLOR_RESET);
}

/** Aloca uma matriz com a altura e a largura informadas. */
Matrix * create_matrix(int matrix_height, int matrix_width){
  Matrix * matrix = (Matrix *) aligned_alloc(32, sizeof(int) * 2 + sizeof(float) * (matrix_height*matrix_width));
  
  matrix->height = matrix_height;
  matrix->width = matrix_width;
  matrix->rows = (float *) aligned_alloc(32, matrix_height * matrix_width * sizeof(float));
  
  return matrix;
}


/** ------------READ MATRIX------------ **/

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


/** ----------SCALAR MATRIX MULT---------- **/

/** Multiplica matriz por um valor fornecido da forma mais otimizada disponível. */
int scalar_matrix_mult(float scalar_value, Matrix * matrix){
  return scalar_matrix_mult_avx_pthread(scalar_value, matrix);
}

/** Multiplica matriz por um valor fornecido sem otimizações. */
int scalar_matrix_mult_normal(float scalar_value, Matrix * matrix){
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

/** Multiplica matriz por um valor fornecido com avx. */
int scalar_matrix_mult_avx(float scalar_value, Matrix * matrix){
  if(matrix == NULL){
    printf("\nMatriz não declarada.\n");
    return 0;
  }

  __m256 vector, scalar, result;

  float * arrayNext = matrix->rows;

  for(int i = 0; i < matrix->height*matrix->width; i+=AVX_STEP, arrayNext+=AVX_STEP){

    vector = _mm256_load_ps(arrayNext);
    scalar = _mm256_set1_ps(scalar_value);

    result = _mm256_mul_ps(vector, scalar);
      
    _mm256_store_ps(arrayNext, result);
  }

  return 1;
}

void * MultiplyMatrixLineByScalarAvx(void * threadarg) {
  unsigned long int i;
  struct scalar_thread_data * my_data;
  my_data = (struct scalar_thread_data*) threadarg;

  int segmentEnd = (my_data->size / my_data->num_threads) + my_data->offset;
  if(my_data->last){
    segmentEnd = my_data->size;
  }

  __m256 vector, scalar, result;

  float * arrayNext = &my_data->matrix->rows[my_data->offset];

  for(i = my_data->offset; i < segmentEnd; i+=AVX_STEP, arrayNext+=AVX_STEP){

    vector = _mm256_load_ps(arrayNext);
    scalar = _mm256_set1_ps(my_data->scalar);

    result = _mm256_mul_ps(vector, scalar);
      
    _mm256_store_ps(arrayNext, result);
  }

  pthread_exit(NULL);
}

/** Multiplica matriz por um valor fornecido com avx e pthreads. */
int scalar_matrix_mult_avx_pthread(float scalar_value, Matrix * matrix){
  if(matrix == NULL){
    printf("\nMatriz não declarada.\n");
    return 0;
  }

  pthread_t threads[thread_num];
  pthread_attr_t attr;
  int rc;
  long t;
  struct scalar_thread_data thread_data_array[thread_num];
  void *status;
  int size = matrix->height*matrix->width;

  if(size % AVX_STEP*thread_num != 0){
    printf("The array size must be a multiple of (thread_num * AVX_STEP).\n");
    return 1;
  }

  /* Inicializa e define o atributo detached do thread */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  /* Cria threads para inicializar o array em memória */
  for(t = 0; t < thread_num; t++){
    // printf("In main: creating thread %ld\n", t);
    thread_data_array[t].thread_id = t;
    thread_data_array[t].size = size;
    thread_data_array[t].offset = (size / thread_num) * t;
    thread_data_array[t].num_threads = thread_num;
    thread_data_array[t].matrix = matrix;
    thread_data_array[t].scalar = scalar_value;

    if(t == thread_num-1){
      thread_data_array[t].last = 1;
    }
    else{
      thread_data_array[t].last = 0;
    }

    rc = pthread_create(&threads[t], NULL, MultiplyMatrixLineByScalarAvx, (void *)&thread_data_array[t]);
    if (rc) {
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }

  /* Libera atributo e espera pelos outros threads */
  pthread_attr_destroy(&attr);
  for(t = 0; t < thread_num; t++) {
    rc = pthread_join(threads[t], &status);
    if (rc) {
      printf("ERROR; return code from pthread_join() is %d\n", rc);
      exit(-1);
    }
    // printf("Main: completed join with thread %ld having a status of %ld\n",t,(long)status);
  }

  return 1;
}


/** ----------MATRIX MATRIX MULT---------- **/

/** Multiplica matriz A por matriz B de um valor fornecido da forma mais otimizada disponível. */
int matrix_matrix_mult(Matrix * matrix_a, Matrix * matrix_b, Matrix * matrix_c){
  return matrix_matrix_mult_otm_avx_pthread(matrix_a, matrix_b, matrix_c);
}

/** Multiplica matriz A por matriz B de um valor fornecido sem otimizações. */
int matrix_matrix_mult_normal(Matrix * matrix_a, Matrix * matrix_b, Matrix * matrix_c){
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

    for(int k = 0; k < matrix_b->width; k+=AVX_STEP, arrayBNext+=AVX_STEP, arrayCNext+=AVX_STEP){
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
  struct matrix_thread_data * my_data;
  my_data = (struct matrix_thread_data*) threadarg;

  int segmentEnd = (my_data->size / my_data->num_threads) + my_data->offset;
  if(my_data->last){
    segmentEnd = my_data->size;
  }

  // printThreadMatrixData(my_data, segmentEnd, COLOR_YELLOW);

  Matrix * matrix_a = my_data->matrix_a;
  Matrix * matrix_b = my_data->matrix_b;
  Matrix * matrix_c = my_data->matrix_c;

  for(i = my_data->offset; i < segmentEnd; i++){
    for(j = 0; j < matrix_a->width; j++){
      float position = matrix_a->rows[i * matrix_a->width + j];

       for(k = 0; k < matrix_b->width; k++){
         matrix_c->rows[i * matrix_c->width + k] += (position * matrix_b->rows[j+matrix_b->width + k]);
       }
    }
  }

  pthread_exit(NULL);
}

/** Multiplica matriz A por matriz B de um valor fornecido de uma forma otimizada, utilizando pthreads. */
int matrix_matrix_mult_otm_pthread(Matrix * matrix_a, Matrix * matrix_b, Matrix * matrix_c){
  if(matrix_a == NULL || matrix_b == NULL){
    printf("\nUma ou duas das matrizes não declaradas.\n");
    return 0;
  }

  /* Inicializar variáveis */
  pthread_t threads[thread_num];
  pthread_attr_t attr;
  int rc;
  long t;
  struct matrix_thread_data thread_data_array[thread_num];
  void *status;
  int size = matrix_a->height;
  
  /* Verifica se as matrizes estão de acordo com as restrições de multiplicação */
  if(matrix_a->width != matrix_b->height){
    printf("\nA matriz A deve ter o número de colunas igual ao número de linhas da matriz B.\n");
    return 0;
  }

  if(size % AVX_STEP*thread_num != 0){
    printf("The array size must be a multiple of (thread_num * AVX_STEP).\n");
    return 1;
  }

  /* Inicializa e define o atributo detached do thread */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  /* Cria threads para inicializar os tres arrays em memória */
  for(t = 0; t < thread_num; t++){
    // printf("In main: creating thread %ld\n", t);
    thread_data_array[t].thread_id = t;
    thread_data_array[t].size = size;
    thread_data_array[t].offset = (size / thread_num) * t;
    thread_data_array[t].num_threads = thread_num;
    thread_data_array[t].matrix_a = matrix_a;
    thread_data_array[t].matrix_b = matrix_b;
    thread_data_array[t].matrix_c = matrix_c;

    if(t == thread_num-1){
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
  for(t = 0; t < thread_num; t++) {
    rc = pthread_join(threads[t], &status);
    if (rc) {
      printf("ERROR; return code from pthread_join() is %d\n", rc);
      exit(-1);
    }
    // printf("Main: completed join with thread %ld having a status of %ld\n",t,(long)status);
  }

  return 1;
}

/* Função que será executada por cada thread do algoritmo de multiplicação de matrizes com avx e pthreads */
void * MultiplyMatrixLineAvx(void * threadarg) {
  unsigned long int i, j;
  struct matrix_thread_data * my_data;
  my_data = (struct matrix_thread_data*) threadarg;

  int segmentEnd = (my_data->size / my_data->num_threads) + my_data->offset;
  if(my_data->last){
    segmentEnd = my_data->size;
  }

  // printThreadMatrixData(my_data, segmentEnd, COLOR_CYAN);

  /* Declara os três vetores */
  __m256 vectorA, vectorB, vectorC, result;

  Matrix * matrix_a = my_data->matrix_a;
  Matrix * matrix_b = my_data->matrix_b;
  Matrix * matrix_c = my_data->matrix_c;

  float * arrayANext = matrix_a->rows;
  float * arrayBNext = matrix_b->rows;
  float * arrayCNext = matrix_c->rows;

  for(i = my_data->offset; i < segmentEnd; i++, arrayANext++){

    vectorA = _mm256_set1_ps(*arrayANext);
    arrayBNext = matrix_b->rows;

    int row = i / matrix_a->width;
    arrayCNext = matrix_c->rows + row * matrix_b->width;

    for(j = 0; j < matrix_b->width; j+=AVX_STEP, arrayBNext+=AVX_STEP, arrayCNext+=AVX_STEP){
      vectorB = _mm256_load_ps(arrayBNext);
      vectorC = _mm256_load_ps(arrayCNext);

      result = _mm256_fmadd_ps(vectorA, vectorB, vectorC);
      
      _mm256_store_ps(arrayCNext, result);
    }

  }

  pthread_exit(NULL);
}

/** Multiplica matriz A por matriz B de um valor fornecido de uma forma otimizada, utilizando avx e pthreads. */
int matrix_matrix_mult_otm_avx_pthread(Matrix * matrix_a, Matrix * matrix_b, Matrix * matrix_c){
  if(matrix_a == NULL || matrix_b == NULL){
    printf("\nUma ou duas das matrizes não declaradas.\n");
    return 0;
  }

  /* Inicializar variáveis */
  pthread_t threads[thread_num];
  pthread_attr_t attr;
  int rc;
  long t;
  struct matrix_thread_data thread_data_array[thread_num];
  void *status;
  int size = matrix_a->height*matrix_a->width;

  /* Verifica se as matrizes estão de acordo com as restrições de multiplicação */
  if(matrix_a->width != matrix_b->height){
    printf("\nA matriz A deve ter o número de colunas igual ao número de linhas da matriz B.\n");
    return 0;
  }

  if(size % AVX_STEP*thread_num != 0){
    printf("The array size must be a multiple of (thread_num * AVX_STEP).\n");
    return 1;
  }

  /* Inicializa e define o atributo detached do thread */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  /* Cria threads para inicializar os tres arrays em memória */
  for(t = 0; t < thread_num; t++){
    // printf("In main: creating thread %ld\n", t);
    thread_data_array[t].thread_id = t;
    thread_data_array[t].size = size;
    thread_data_array[t].offset = (size / thread_num) * t;
    thread_data_array[t].num_threads = thread_num;
    thread_data_array[t].matrix_a = matrix_a;
    thread_data_array[t].matrix_b = matrix_b;
    thread_data_array[t].matrix_c = matrix_c;

    if(t == thread_num-1){
      thread_data_array[t].last = 1;
    }
    else{
      thread_data_array[t].last = 0;
    }

    rc = pthread_create(&threads[t], NULL, MultiplyMatrixLineAvx, (void *)&thread_data_array[t]);
    if (rc) {
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }

  /* Libera atributo e espera pelos outros threads */
  pthread_attr_destroy(&attr);
  for(t = 0; t < thread_num; t++) {
    rc = pthread_join(threads[t], &status);
    if (rc) {
      printf("ERROR; return code from pthread_join() is %d\n", rc);
      exit(-1);
    }
    // printf("Main: completed join with thread %ld having a status of %ld\n",t,(long)status);
  }

  return 1;
}


/** ------------WRITE MATRIX------------ **/

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