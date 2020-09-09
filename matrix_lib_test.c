#include "matrix_lib.h"

//8x16 * 16x10

//8x10

int main(int argc, char **argv){
	// Declaracao e inicializacao de variaveis misc
	float scalar = atof(argv[1]);
	int error;

	// Declaracao das matrizes
	Matrix * mA;
	Matrix * mB;
	Matrix * mC;

	// Declaracao dos ponteiros dos arquivos
	FILE *file1, *file2, *result1, *result2;

	// Armazenando as dimensoes das duas matrizes
	int mA_height = atoi(argv[2]);
	int mA_width = atoi(argv[3]);
	int mB_height = atoi(argv[4]);
	int mB_width = atoi(argv[5]);

	// Armazenando as dimensoes da matriz resultante
	int mC_height = mA_height;
	int mC_width = mB_width;

	// Read & Write dos arquivos de entrada e saida
  	file1 = fopen(argv[6],"rb");
	file2 = fopen(argv[7],"rb");
	result1 = fopen(argv[8],"wb");
	result2 = fopen(argv[9],"wb");

	//Verificacao de abertura de arquivos
	if(file1 == NULL || file2 == NULL)
	{
		fprintf(stdout, "Erro ao abrir arquivo de dados\n");
		exit(1);
	}
	
	// Inicializacao das matrizes
	mA = create_matrix(mA_height, mA_width);
	mB = create_matrix(mB_height, mB_width);
	mC = create_matrix(mC_height, mC_width);

	// Leitura e preenchimento das matrizes
	error = fill_matrix_with_file(file1, mA);
	if (error == 0){
		printf("Matriz A nao inicializada\n");
	}
	
	error = fill_matrix_with_file(file2, mB);
	if (error == 0){
		printf("Matriz B nao inicializada\n");
	}
	
	// Preenchendo a matriz C com zeros
	fill_matrix(0, mC);
	
	matrix_print(mA, "A");
	matrix_print(mB, "B");

	// Multiplicando matriz A pelo valor escalar
	error = scalar_matrix_mult(scalar, mA);
	if(error == 0){
		printf("Erro função de multiplicar matriz A por escalar");
	}

	error = write_matrix_to_file(result1, mA);
	if(error == 0){
		printf("Erro ao escrever matriz no arquivo .dat");
	}
	
	
	// Multiplicando matriz A pela matriz B
	error = matrix_matrix_mult(mA, mB, mC);
	if(error == 0){
		printf("Erro função de multiplar matriz A por matriz B");
	}

	error = write_matrix_to_file(result2, mC);
	if(error == 0){
		printf("Erro ao escrever matriz no arquivo .dat");
	}

	matrix_print(mC, "C");

	// Fechando arquivos .dat
	fclose(file1);
	fclose(file2);
	fclose(result1);
	fclose(result2);

	// Liberando memoria das matrizes
	free(mA);
	free(mB);
	free(mC);
	
	return 0;
}
