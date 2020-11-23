#include "matrix_lib.cu"
extern "C" {
#include "timer.h"
}

int main(int argc, char **argv){
	// Declaracao das variaveis de tempo
	struct timeval start_scalar_mult, stop_scalar_mult,
			start_matrix_mult, stop_matrix_mult,
			start_matrix_read_time, stop_matrix_read_time,
			overall_t1, overall_t2;
					
	float overall_time, 
			matrix_scalar_time,
			matrix_mult_time,
			matrix_read_time;

	gettimeofday(&overall_t1, NULL);

	// Declaracao e inicializacao de variaveis misc
	float scalar = atof(argv[1]);
	int error;

	// Declaracao das matrizes
	Matrix * mA;
	Matrix * mB;

	// Para fins de teste
	Matrix * mC;

	// Declaracao dos ponteiros dos arquivos
	FILE *file1, *file2, *result1, *result2;

	// Armazenando as dimensoes das duas matrizes
	int mA_height = atoi(argv[2]);
	int mA_width = atoi(argv[3]);
	int mB_height = atoi(argv[4]);
	int mB_width = atoi(argv[5]);
	
	// Setando o tamanho de grid
	int threads_per_block_param = atoi(argv[6]);
	int max_blocks_per_grid_param = atoi(argv[7]);
	error = set_grid_size(threads_per_block_param,max_blocks_per_grid_param);
	if (error == 1)
	{
		printf("Valores aceitos com sucesso\n");
	}
	else
	{
		printf("Erro! Valores utilizados sao os default\n");
	}
	// Read & Write dos arquivos de entrada e saida
	file1 = fopen(argv[8],"rb");
	file2 = fopen(argv[9],"rb");
	result1 = fopen(argv[10],"wb");
	result2 = fopen(argv[11],"wb");

	//Verificacao de abertura de arquivos
	if(file1 == NULL || file2 == NULL)
	{
		fprintf(stdout, "Erro ao abrir arquivo de dados\n");
		exit(1);
	}
	
	// Inicializacao das matrizes
	mA = create_matrix(mA_height, mA_width);
	mB = create_matrix(mB_height, mB_width);
	mC = create_matrix(mA_height, mB_width);
	
	// Leitura e preenchimento das matrizes
	printf("\nLendo matriz A por arquivo...\n");
	gettimeofday(&start_matrix_read_time, NULL);
	error = fill_matrix_with_file(file1, mA);
	if (error == 0){
		printf("Matriz A nao inicializada\n");
	}
	
	printf("\nLendo matriz B por arquivo...\n");
	error = fill_matrix_with_file(file2, mB);
	if (error == 0){
		printf("Matriz B nao inicializada\n");
	}
	gettimeofday(&stop_matrix_read_time, NULL);
	
	// Preenchendo a matriz C com zeros
	fill_matrix(0, mC);
	// Multiplicando matriz A pelo valor escalar e cronometrando o tempo
	gettimeofday(&start_scalar_mult, NULL);
	printf("\nMultiplicando matriz A por %.3f...\n", scalar);
	error = scalar_matrix_mult(scalar, mB);
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
	printf("\nMultiplicando matriz A por matriz B...\n");
	error = matrix_matrix_mult(mA, mB, mC);
	gettimeofday(&stop_matrix_mult, NULL);
	if(error == 0){
		printf("Erro função de multiplar matriz A por matriz B");
	}

	// Comandos para debugging do resultado
	matrix_print(mC, "C");

	//error = write_matrix_to_file(result2, mC);
	if(error == 0){
		printf("Erro ao escrever matriz no arquivo .dat");
	}

	// Fechando arquivos .dat
	fclose(file1);
	fclose(file2);
	fclose(result1);
	fclose(result2);

	// Liberando memoria das matrizes
	cudaFree(mA);
	cudaFree(mB);
	cudaFree(mC);
	
	gettimeofday(&overall_t2, NULL);

	overall_time = timedifference_msec(overall_t1, overall_t2);
	matrix_scalar_time = timedifference_msec(start_scalar_mult, stop_scalar_mult);
	matrix_mult_time = timedifference_msec(start_matrix_mult, stop_matrix_mult);
	matrix_read_time = timedifference_msec(start_matrix_read_time, stop_matrix_read_time);

	printf("\n==========================================================================================================================");
	printf("\nTempo total para leitura das matrizes: (%f ms) | (%f s) | (%f min)\n", matrix_read_time, matrix_read_time/1000, matrix_read_time/60000);
	
	printf("\nTempo total para multiplicacao escalar: (%f ms) | (%f s) | (%f min)\n", matrix_scalar_time, matrix_scalar_time/1000, matrix_scalar_time/60000);

	printf("Tempo total para multiplicacao de matrizes A por B: (%f ms) | (%f s) | (%f min)\n", matrix_mult_time, matrix_mult_time/1000, matrix_mult_time/60000);

	printf("\nTempo total do programa: (%f ms) | (%f s) | (%f min)\n", overall_time, overall_time/1000, overall_time/60000);
	printf("==========================================================================================================================\n");


	return 0;
}
