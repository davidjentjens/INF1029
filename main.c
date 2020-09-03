#include <stdio.h>
#include <stdlib.h>

#include "matrix_lib.h"

#define MA_HEIGHT 3
#define MA_WIDTH 2

#define MB_HEIGHT 2
#define MB_WIDTH 4

int main(int argc, char *argv[]){
  
  Matrix * matrix_a = create_matrix(MA_HEIGHT, MA_WIDTH);
  Matrix * matrix_b = create_matrix(MB_HEIGHT, MB_WIDTH);

  Matrix * matrix_c = create_matrix(MA_HEIGHT, MB_WIDTH);

  fill_matrix(10, matrix_a);
  fill_matrix(2, matrix_b);

  matrix_print(matrix_a);
  matrix_print(matrix_b);

  matrix_matrix_mult(matrix_a, matrix_b, matrix_c);

  matrix_print(matrix_c);

  return 0;
}