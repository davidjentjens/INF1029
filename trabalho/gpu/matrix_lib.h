#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** Matriz com altura e largura definidas. */
typedef struct matrix {
   unsigned long int height;
   unsigned long int width;
   float *h_rows;
   float *d_rows;
} Matrix;

/** Determina o número de threads por bloco e o número máximo de blocos por grid */
int set_grid_size(int threads_per_block_param, int max_blocks_per_grid_param);

/** Aloca uma matriz com a altura e a largura informadas. */
Matrix * create_matrix(int matrix_height, int matrix_width);

/** ------------READ MATRIX------------ **/

/** Preenche matriz com um arquivo .dat fornecido. */
int fill_matrix_with_file(FILE * file, Matrix * matrix);

/** Preenche matriz com um valor fornecido. */
int fill_matrix(float value, Matrix * matrix);

/** ----------SCALAR MATRIX MULT---------- **/

/** Multiplica matriz por um valor fornecido utilizando a GPU. */
int scalar_matrix_mult(float scalar_value, Matrix * matrix);

/** ----------MATRIX MATRIX MULT---------- **/

/** Multiplica matriz A por matriz B de um valor fornecido de uma forma otimizada utilizando a GPU. */
int matrix_matrix_mult(Matrix * matrix_a, Matrix * matrix_b, Matrix * matrix_c);

/** ------------WRITE MATRIX------------ **/

/** Imprime a matriz fornecida */
int matrix_print(Matrix * matrix, const char * nome);

/** Escreve matriz para um arquivo .dat */
int write_matrix_to_file(FILE * file, Matrix * matrix);