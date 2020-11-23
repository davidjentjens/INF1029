#include <cuda_runtime.h>
#include "matrix_lib.h"

#define COLOR_CYAN "\033[0;36m"
#define COLOR_YELLOW "\033[1;33m"
#define COLOR_RESET "\033[0m"

#define DEVICE_DATASET_SIZE 1024000

#define THREADS_PER_BLOCK_LIMIT 1024
#define MAX_BLOCKS_PER_GRID_LIMIT 65535

static int threads_per_block = 256;
static int max_blocks_per_grid = 4096;

/** Determina o número de threads por bloco e o número máximo de blocos por grid */
int set_grid_size(int threads_per_block_param, int max_blocks_per_grid_param){
  if(threads_per_block_param < THREADS_PER_BLOCK_LIMIT && max_blocks_per_grid_param < MAX_BLOCKS_PER_GRID_LIMIT){
    threads_per_block = threads_per_block_param;
    max_blocks_per_grid = max_blocks_per_grid_param;

    return 1;
  }
  
  return 0;
}

/** Aloca uma matriz com a altura e a largura informadas. */
Matrix * create_matrix(int matrix_height, int matrix_width){
  
  Matrix * matrix = (Matrix *) malloc(sizeof(int) * 2 + sizeof(float) * (matrix_height*matrix_width) + (DEVICE_DATASET_SIZE * sizeof(float)));
  
  matrix->height = matrix_height;
  matrix->width = matrix_width;
  matrix->h_rows = (float *) malloc(matrix_height*matrix_width * sizeof(float));

  // check malloc memory allocation
  if (matrix->h_rows == NULL) { 
    printf("Error: malloc unable to allocate memory on host.");
    return 0;
  }

  cudaError = cudaMalloc(&(matrix->d_rows), DEVICE_DATASET_SIZE*sizeof(float));

  // check cudaMalloc memory allocation
  if (cudaError != cudaSuccess) {
    printf("cudaMalloc d_x returned error %s (code %d)\n",cudaGetErrorString(cudaError), cudaError);
    return 0;
  }

  return matrix;
}


/** ------------READ MATRIX------------ **/

/** Preenche matriz com um arquivo .dat fornecido. */
int fill_matrix_with_file(FILE * file, Matrix * matrix){
  if(matrix == NULL){
    return 0;
  }

  int count = 0;
  float * vet = (float*) malloc((matrix->height * matrix->width) * sizeof(float));
  float vet_aux;

  for(int i = 0; i < matrix->height * matrix->width; i++){
		fread((void*) (&vet_aux), sizeof(vet_aux), 1, file);
		vet[count] = vet_aux;
		count++;
	}

  matrix->h_rows = vet;

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
      matrix->h_rows[i * matrix->width + j] = value;
    }
  }

  return 1;
}


/** ----------SCALAR MATRIX MULT---------- **/

// Kernel function to scalar to array
__global__ 
void mult_scalar(int n, float *matrix_rows, float scalar_value)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  
  if(index == 0){
    printf("\nblockDim.x=%d   gridDim.x%d   stride=%d\n", blockDim.x, gridDim.x, stride);
  }

  for (int i = index; i < n; i += stride) {
    matrix_rows[i] = matrix_rows[i] * scalar_value;
  }
}

/** Multiplica matriz por um valor fornecido utilizando a GPU. */
int scalar_matrix_mult(float scalar_value, Matrix * matrix){
  if(matrix == NULL){
    printf("\nMatriz não declarada.\n");
    return 0;
  }

  int matrix_size = matrix->height * matrix->width;

  int loop_limit = (matrix_size + DEVICE_DATASET_SIZE - 1) / DEVICE_DATASET_SIZE;
  int chunk_size = DEVICE_DATASET_SIZE;

  for(int count = 0; count < loop_limit; ++count){
    if(matrix_size % DEVICE_DATASET_SIZE != 0 && count == loop_limit - 1){
      chunk_size = matrix_size % DEVICE_DATASET_SIZE;
    }
    
    cudaError = cudaMemcpy(matrix->d_rows, matrix->h_rows+(count*chunk_size), chunk_size*sizeof(float), cudaMemcpyHostToDevice);

    if (cudaError != cudaSuccess) {
      printf("cudaMemcpy (h -> d) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
      return 0;
    }

    int blockSize = threads_per_block;
    int numBlocks = (chunk_size + blockSize - 1) / blockSize;
    
    if (numBlocks > max_blocks_per_grid){
      numBlocks = max_blocks_per_grid;
    }

    mult_scalar<<<numBlocks, blockSize>>>(chunk_size, matrix->d_rows, scalar_value);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    cudaError = cudaMemcpy(matrix->h_rows+(count*chunk_size), matrix->d_rows, chunk_size*sizeof(float), cudaMemcpyDeviceToHost);
  
    if (cudaError != cudaSuccess){
      printf("cudaMemcpy (d -> h) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
      return 0;
    }
  }

  return 1;
}



/** ----------MATRIX MATRIX MULT---------- **/

// Kernel function to mult to array
__global__ 
void matrix_mult(int n, Matrix * matrix_a, Matrix * matrix_b, Matrix * matrix_c)
{
  //printf("PPPPPPPPPPPPPPPP\n");

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  
  if(index == 0){
    printf("\nblockDim.x=%d   gridDim.x%d   stride=%d\n", blockDim.x, gridDim.x, stride);
  }

  float * arrayANext = matrix_a->d_rows;
  float * arrayBNext = matrix_b->d_rows;
  float * arrayCNext = matrix_c->d_rows;

  for(int i = 0; i < n; i += stride, arrayANext += stride){
    printf("PPPPPPPPPPPPPPPP\n");

    arrayBNext = matrix_b->d_rows;

    int row = i / matrix_a->width;
    arrayCNext = matrix_c->d_rows + row * matrix_b->width;

    for(int k = 0; k < matrix_b->width; k++, arrayBNext++, arrayCNext++){
      printf("%d\n", (*arrayANext) * (*arrayBNext));
      *arrayCNext = (*arrayANext) * (*arrayBNext);
    }
  } 
}

/** Multiplica matriz A por matriz B de um valor fornecido de uma forma otimizada utilizando a GPU. */
int matrix_matrix_mult(Matrix * matrix_a, Matrix * matrix_b, Matrix * matrix_c){

  if(matrix_a == NULL || matrix_b == NULL){
    printf("\nUma ou duas das matrizes não declaradas.\n");
    return 0;
  }

  if(matrix_a->width != matrix_b->height){
    printf("\nA matriz A deve ter o número de colunas igual ao número de linhas da matriz B.\n");
    return 0;
  }

  int matrix_size = matrix_a->height * matrix_a->width;

  int loop_limit = (matrix_size + DEVICE_DATASET_SIZE - 1) / DEVICE_DATASET_SIZE;
  int chunk_size = DEVICE_DATASET_SIZE;
  for(int count = 0; count < loop_limit; ++count){
    if(matrix_size % DEVICE_DATASET_SIZE != 0 && count == loop_limit - 1){
      chunk_size = matrix_size % DEVICE_DATASET_SIZE;
    }
    
    cudaError = cudaMemcpy(matrix_a->d_rows, matrix_a->h_rows+(count*chunk_size), chunk_size*sizeof(float), cudaMemcpyHostToDevice);

    if (cudaError != cudaSuccess) {
      printf("cudaMemcpy (h_a -> d_a) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
      return 0;
    }

    cudaError = cudaMemcpy(matrix_b->d_rows, matrix_b->h_rows+(count*chunk_size), chunk_size*sizeof(float), cudaMemcpyHostToDevice);

    if (cudaError != cudaSuccess) {
      printf("cudaMemcpy (h_b -> d_b) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
      return 0;
    }

    cudaError = cudaMemcpy(matrix_c->d_rows, matrix_c->h_rows+(count*chunk_size), chunk_size*sizeof(float), cudaMemcpyHostToDevice);

    if (cudaError != cudaSuccess) {
      printf("cudaMemcpy (h_c -> d_c) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
      return 0;
    }

    int blockSize = threads_per_block;
    int numBlocks = (chunk_size + blockSize - 1) / blockSize;
    if (numBlocks > max_blocks_per_grid) {
      numBlocks = max_blocks_per_grid;
    }

    matrix_mult<<<numBlocks, blockSize>>>(chunk_size, matrix_a, matrix_b, matrix_c);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    cudaError = cudaMemcpy(matrix_c->h_rows+(count*chunk_size), matrix_c->d_rows, chunk_size*sizeof(float), cudaMemcpyDeviceToHost);
  
    if (cudaError != cudaSuccess){
      printf("cudaMemcpy (d_c -> h_c) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
      return 0;
    }
  }

  return 1;
}

/** ------------WRITE MATRIX------------ **/

/** Imprime a matriz fornecida */
int matrix_print(Matrix * matrix, const char * nome){
  if(matrix == NULL){
    printf("\nMatriz não declarada.\n");
    return 0;
  }

  printf("\nImprimindo matriz %s para o console:\n",nome);


  for(int i = 0; i < matrix->height; i++){
    for (int j = 0; j < matrix->width; j++){
      printf("%.3f, ", matrix->h_rows[i * matrix->width + j]);
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
 		fwrite((void*)(&matrix->h_rows[i]), sizeof(matrix->h_rows[i]), 1, file);
	}

  return 1;
} 