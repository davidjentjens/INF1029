#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <pthread.h>

/** Matriz com altura e largura definidas. */
typedef struct matrix {
  unsigned long int height;
  unsigned long int width;
  float *rows;
} Matrix;

/* Estrutura de dados de cada thread de multiplicação de matriz por escalar*/
struct scalar_thread_data {
   long thread_id;
   int offset;
   int size;
   int last;
   int num_threads;
   float scalar;
   Matrix * matrix;
};

/* Estrutura de dados de cada thread de multiplicação de matriz por matriz */
struct matrix_thread_data {
   long thread_id;
   int offset;
   int size;
   int last;
   int num_threads;
   Matrix * matrix_a;
   Matrix * matrix_b;
   Matrix * matrix_c;
};

/** Determina o número de threads a serem utilizados nas funções que usam threads **/
void set_number_threads(int num_threads);

/** Aloca uma matriz com a altura e a largura informadas. */
Matrix * create_matrix(int matrix_height, int matrix_width);

/** ------------READ MATRIX------------ **/

/** Preenche matriz com um arquivo .dat fornecido. */
int fill_matrix_with_file(FILE * file, Matrix * matrix);

/** Preenche matriz com um valor fornecido. */
int fill_matrix(float value, Matrix * matrix);


/** ----------SCALAR MATRIX MULT---------- **/

/** Multiplica matriz por um valor fornecido da forma mais otimizada disponível. */
int scalar_matrix_mult(float scalar_value, Matrix * matrix);

/** Multiplica matriz por um valor fornecido sem otimizações. */
int scalar_matrix_mult_normal(float scalar_value, Matrix * matrix);

/** Multiplica matriz por um valor fornecido com avx. */
int scalar_matrix_mult_avx(float scalar_value, Matrix * matrix);

/** Multiplica matriz por um valor fornecido com avx e pthreads. */
int scalar_matrix_mult_avx_pthread(float scalar_value, Matrix * matrix);


/** ----------MATRIX MATRIX MULT---------- **/

/** Multiplica matriz A por matriz B de um valor fornecido da forma mais otimizada disponível. */
int matrix_matrix_mult(Matrix * matrix_a, Matrix * matrix_b, Matrix * matrix_c);

/** Multiplica matriz A por matriz B de um valor fornecido sem otimizações. */
int matrix_matrix_mult_normal(Matrix * matrix_a, Matrix * matrix_b, Matrix * matrix_c);

/** Multiplica matriz A por matriz B de um valor fornecido de uma forma otimizada. */
int matrix_matrix_mult_otm(Matrix * matrix_a, Matrix * matrix_b, Matrix * matrix_c);

/** Multiplica matriz A por matriz B de um valor fornecido de uma forma otimizada, utilizando AVX. */
int matrix_matrix_mult_otm_avx(Matrix * matrix_a, Matrix * matrix_b, Matrix * matrix_c);

/** Multiplica matriz A por matriz B de um valor fornecido de uma forma otimizada, utilizando pthreads. */
int matrix_matrix_mult_otm_pthread(Matrix * matrix_a, Matrix * matrix_b, Matrix * matrix_c);

/** Multiplica matriz A por matriz B de um valor fornecido de uma forma otimizada, utilizando avx e pthreads. */
int matrix_matrix_mult_otm_avx_pthread(Matrix * matrix_a, Matrix * matrix_b, Matrix * matrix_c);


/** ------------WRITE MATRIX------------ **/

/** Imprime a matriz fornecida */
int matrix_print(Matrix * matrix, char * nome);

/** Escreve matriz para um arquivo .dat */
int write_matrix_to_file(FILE * file, Matrix * matrix);