#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

/** Matriz com altura e largura definidas. */
typedef struct matrix {
  unsigned long int height;
  unsigned long int width;
  float *rows;
} Matrix;

/** Aloca uma matriz com a altura e a largura informadas. */
Matrix * create_matrix(int matrix_height, int matrix_width);

/** Preenche matriz com um arquivo .dat fornecido. */
int fill_matrix_with_file(FILE * file, Matrix * matrix);

/** Preenche matriz com um valor fornecido. */
int fill_matrix(float value, Matrix * matrix);

/** Multiplica matriz por um valor fornecido. */
int scalar_matrix_mult(float scalar_value, Matrix * matrix);


/** Multiplica matriz A por matriz B de um valor fornecido. */
int matrix_matrix_mult(Matrix * matrix_a, Matrix * matrix_b, Matrix * matrix_c);

/** Multiplica matriz A por matriz B de um valor fornecido de uma forma otimizada. */
int matrix_matrix_mult_otm(Matrix * matrix_a, Matrix * matrix_b, Matrix * matrix_c);

/** Multiplica matriz A por matriz B de um valor fornecido de uma forma otimizada, utilizando AVX. */
int matrix_matrix_mult_otm_avx(Matrix * matrix_a, Matrix * matrix_b, Matrix * matrix_c);


/** Imprime a matriz fornecida */
int matrix_print(Matrix * matrix, char * nome);

/** Escreve matriz para um arquivo .dat */
int write_matrix_to_file(FILE * file, Matrix * matrix);