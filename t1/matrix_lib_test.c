#include "matrix_lib.h"
#include "timer.h"

int main(int argc, char **argv){
	// Declaracao das variaveis de tempo
	struct timeval start_scalar_mult, stop_scalar_mult, start_matrix_mult,stop_matrix_mult,
					start_matrix_mult_otm,stop_matrix_mult_otm, overall_t1, overall_t2;
	float overall_time, matrix_scalar_time, matrix_mult_time,matrix_mult_time_otm;
	gettimeofday(&overall_t1, NULL);

	// Declaracao e inicializacao de variaveis misc
	float scalar = atof(argv[1]);
	int error;

	// Declaracao das matrizes
	Matrix * mA;
	Matrix * mB;
	Matrix * mC;
	Matrix * moC; // Para fins de teste--

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
	int moC_height = mA_height; // Para fins de teste--
	int moC_width = mB_width; // Para fins de teste--

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
	moC = create_matrix(moC_height, moC_width); // Para fins de teste--

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
	fill_matrix(0, moC);// Para fins de teste--

	// Multiplicando matriz A pelo valor escalar e cronometrando o tempo
	gettimeofday(&start_scalar_mult, NULL);
	printf("\nMultiplicando matriz A por %.3f...\n", scalar);
	error = scalar_matrix_mult(scalar, mA);
	gettimeofday(&stop_scalar_mult, NULL);
	if(error == 0){
		printf("Erro função de multiplicar matriz A por escalar");
	}

	error = write_matrix_to_file(result1, mA);
	if(error == 0){
		printf("Erro ao escrever matriz no arquivo .dat");
	}
	
	
	// Multiplicando matriz A pela matriz B e cronometrando o tempo
	gettimeofday(&start_matrix_mult, NULL);
	error = matrix_matrix_mult(mA, mB, mC);
	gettimeofday(&stop_matrix_mult, NULL);
	if(error == 0){
		printf("Erro função de multiplar matriz A por matriz B");
	}
	
	// Multiplicando matriz A pela matriz B (otimizado) e cronometrando o tempo
	gettimeofday(&start_matrix_mult_otm, NULL);
	error = matrix_matrix_mult_otm(mA, mB, moC);
	gettimeofday(&stop_matrix_mult_otm, NULL);
	if(error == 0){
		printf("Erro função de multiplar matriz A por matriz B (otimizado)");
	}


	error = write_matrix_to_file(result2, mC);
	if(error == 0){
		printf("Erro ao escrever matriz no arquivo .dat");
	}

	// Fechando arquivos .dat
	fclose(file1);
	fclose(file2);
	fclose(result1);
	fclose(result2);

	// Liberando memoria das matrizes
	free(mA);
	free(mB);
	free(mC);
	free(moC);
	
	gettimeofday(&overall_t2, NULL);

	overall_time = timedifference_msec(overall_t1, overall_t2);
	matrix_scalar_time = timedifference_msec(start_scalar_mult, stop_scalar_mult);
	matrix_mult_time = timedifference_msec(start_matrix_mult, stop_matrix_mult);
	matrix_mult_time_otm = timedifference_msec(start_matrix_mult_otm, stop_matrix_mult_otm);

	printf("Tempo total do programa: (%f ms) | (%f s) | (%f min)\n", overall_time,overall_time/1000,overall_time/60000);
	printf("Tempo total para multiplicacao escalar: (%f ms) | (%f s) | (%f min)\n", matrix_scalar_time,matrix_scalar_time/1000,matrix_scalar_time/60000);
	printf("Tempo total para multiplicacao de matrizes A por B: (%f ms) | (%f s) | (%f min)\n", matrix_mult_time,matrix_mult_time/1000,matrix_mult_time/60000);
	printf("Tempo total para multiplicacao de matrizes A por B: (otimizado): (%f ms) | (%f s) | (%f min)\n", matrix_mult_time_otm,matrix_mult_time_otm/1000,matrix_mult_time_otm/60000);

	return 0;
}
