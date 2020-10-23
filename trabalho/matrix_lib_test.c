#include "matrix_lib.h"
#include "timer.h"

#define NUM_THREADS 8

int main(int argc, char **argv){
	// Declaracao das variaveis de tempo
	struct timeval start_scalar_mult, stop_scalar_mult, 
			start_matrix_mult, stop_matrix_mult,
			start_matrix_mult_otm, stop_matrix_mult_otm,
			start_matrix_mult_otm_avx, stop_matrix_mult_otm_avx,
			start_matrix_mult_otm_pthread, stop_matrix_mult_otm_pthread,
			start_matrix_read_time, stop_matrix_read_time,
			overall_t1, overall_t2;
					
	float overall_time, 
			matrix_scalar_time, 
			matrix_mult_time,
			matrix_mult_time_otm, 
			matrix_mult_time_otm_avx, 
			matrix_mult_time_otm_pthread,
			matrix_read_time;

	gettimeofday(&overall_t1, NULL);

	// Declaracao e inicializacao de variaveis misc
	float scalar = atof(argv[1]);
	int error;

	// Declaracao das matrizes
	Matrix * mA;
	Matrix * mB;
	Matrix * mC;
	Matrix * m_otm_C; // Para fins de teste--
	Matrix * m_avx_C; // Para fins de teste--

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
	int moC_height = mA_height;
	int moC_width = mB_width; 
	int m_avx_C_height = mA_height; 
	int m_avx_C_width = mB_width; 
	
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
	m_otm_C = create_matrix(moC_height, moC_width);
	m_avx_C = create_matrix(m_avx_C_height, m_avx_C_width); 
	
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
	fill_matrix(0, m_otm_C);
	fill_matrix(0, m_avx_C);

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
	printf("\nMultiplicando matriz A por matriz B...\n");
	error = matrix_matrix_mult(mA, mB, mC);
	gettimeofday(&stop_matrix_mult, NULL);
	if(error == 0){
		printf("Erro função de multiplar matriz A por matriz B");
	}
	
	// Multiplicando matriz A pela matriz B (otimizado) e cronometrando o tempo
	gettimeofday(&start_matrix_mult_otm, NULL);
	printf("\nMultiplicando matriz A por matriz B (Otimizada)...\n");
	error = matrix_matrix_mult_otm(mA, mB, m_otm_C);
	gettimeofday(&stop_matrix_mult_otm, NULL);
	if(error == 0){
		printf("Erro função de multiplar matriz A por matriz B (otimizado)");
	}

	// Multiplicando matriz A pela matriz B (otimizado com AVX) e cronometrando o tempo
	gettimeofday(&start_matrix_mult_otm_avx, NULL);
	printf("\nMultiplicando matriz A por matriz B (Otimizada com AVX)...\n");
	error = matrix_matrix_mult_otm_avx(mA, mB, m_avx_C);
	gettimeofday(&stop_matrix_mult_otm_avx, NULL);
	if(error == 0){
		printf("Erro função de multiplar matriz A por matriz B (otimizado com AVX)");
	}

	// Multiplicando matriz A pela matriz B (otimizado com pthread) e cronometrando o tempo
	gettimeofday(&start_matrix_mult_otm_pthread, NULL);
	printf("\nMultiplicando matriz A por matriz B (Otimizada com pthread)...\n");
	error = matrix_matrix_mult_otm_pthread(mA, mB, m_avx_C, NUM_THREADS);
	gettimeofday(&stop_matrix_mult_otm_pthread, NULL);
	if(error == 0){
		printf("Erro função de multiplar matriz A por matriz B (otimizado com pthread)");
	}

	// Comandos para debugging do resultado
	// matrix_print(mC, "C");
	// matrix_print(m_avx_C, "C");

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
	free(m_otm_C);
	free(m_avx_C);
	
	gettimeofday(&overall_t2, NULL);

	overall_time = timedifference_msec(overall_t1, overall_t2);
	matrix_scalar_time = timedifference_msec(start_scalar_mult, stop_scalar_mult);
	matrix_mult_time = timedifference_msec(start_matrix_mult, stop_matrix_mult);
	matrix_mult_time_otm = timedifference_msec(start_matrix_mult_otm, stop_matrix_mult_otm);
	matrix_mult_time_otm_avx = timedifference_msec(start_matrix_mult_otm_avx, stop_matrix_mult_otm_avx);
	matrix_mult_time_otm_pthread = timedifference_msec(start_matrix_mult_otm_pthread, stop_matrix_mult_otm_pthread);
	matrix_read_time = timedifference_msec(start_matrix_read_time, stop_matrix_read_time);

	printf("\n==========================================================================================================================");
	printf("\nTempo total para leitura das matrizes: (%f ms) | (%f s) | (%f min)\n", matrix_read_time, matrix_read_time/1000, matrix_read_time/60000);
	printf("\nTempo total para multiplicacao escalar: (%f ms) | (%f s) | (%f min)\n\n", matrix_scalar_time, matrix_scalar_time/1000, matrix_scalar_time/60000);
	
	printf("Tempo total para multiplicacao de matrizes A por B: (normal --------------): (%f ms) | (%f s) | (%f min)\n", matrix_mult_time, matrix_mult_time/1000, matrix_mult_time/60000);
	printf("Tempo total para multiplicacao de matrizes A por B: (otimizado -----------): (%f ms) | (%f s) | (%f min)\n", matrix_mult_time_otm, matrix_mult_time_otm/1000, matrix_mult_time_otm/60000);
	printf("Tempo total para multiplicacao de matrizes A por B: (otimizado com avx----): (%f ms) | (%f s) | (%f min)\n", matrix_mult_time_otm_avx, matrix_mult_time_otm_avx/1000, matrix_mult_time_otm_avx/60000);
	printf("Tempo total para multiplicacao de matrizes A por B: (otimizado com pthread): (%f ms) | (%f s) | (%f min)\n", matrix_mult_time_otm_pthread, matrix_mult_time_otm_pthread/1000, matrix_mult_time_otm_pthread/60000);

	printf("\nTempo total do programa: (%f ms) | (%f s) | (%f min)\n", overall_time, overall_time/1000, overall_time/60000);
	printf("==========================================================================================================================\n");

	pthread_exit(NULL);

	return 0;
}
