typedef struct matrix {
  unsigned long int height;
  unsigned long int width;
  float *rows;
} Matrix;

Matrix * create_matrix(int matrix_height, int matrix_width);

int fill_matrix(float value, Matrix *matrix);

int scalar_matrix_mult(float scalar_value, struct matrix *matrix);

int matrix_matrix_mult(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC);

int matrix_print(Matrix *matrix);