#include "matrix_lib.h"
#include "timer.h"

int main(int argc, char **argv){
	// Declaracao das variaveis de tempo
	struct timeval start_scalar_mult, stop_scalar_mult,
			start_scalar_mult_avx, stop_scalar_mult_avx,
			start_scalar_mult_avx_pthread, stop_scalar_mult_avx_pthread,
			start_matrix_mult, stop_matrix_mult,
			start_matrix_mult_otm, stop_matrix_mult_otm,
			start_matrix_mult_otm_avx, stop_matrix_mult_otm_avx,
			start_matrix_mult_otm_pthread, stop_matrix_mult_otm_pthread,
			start_matrix_mult_otm_avx_pthread, stop_matrix_mult_otm_avx_pthread,
			start_matrix_read_time, stop_matrix_read_time,
			overall_t1, overall_t2;
					
	float overall_time, 
			matrix_scalar_time,
			matrix_scalar_time_avx,
			matrix_scalar_time_avx_pthread,
			matrix_mult_time,
			matrix_mult_time_otm, 
			matrix_mult_time_otm_avx, 
			matrix_mult_time_otm_pthread,
			matrix_mult_time_otm_avx_pthread,
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
	Matrix * m_otm_C; 
	Matrix * m_avx_C;
	Matrix * m_pthread_C;
	Matrix * m_avx_pthread_C;

	// Declaracao dos ponteiros dos arquivos
	FILE *file1, *file2, *result1, *result2;

	// Armazenando as dimensoes das duas matrizes
	int mA_height = atoi(argv[2]);
	int mA_width = atoi(argv[3]);
	int mB_height = atoi(argv[4]);
	int mB_width = atoi(argv[5]);
	
	// Setando o número de threads
	set_number_threads(atoi(argv[6]));

	// Read & Write dos arquivos de entrada e saida
	file1 = fopen(argv[7],"rb");
	file2 = fopen(argv[8],"rb");
	result1 = fopen(argv[9],"wb");
	result2 = fopen(argv[10],"wb");

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
	m_otm_C = create_matrix(mA_height, mB_width);
	m_avx_C = create_matrix(mA_height, mB_width);
	m_pthread_C = create_matrix(mA_height, mB_width);
	m_avx_pthread_C = create_matrix(mA_height, mB_width);
	
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
	fill_matrix(0, m_pthread_C);
	fill_matrix(0, m_avx_pthread_C);

	// Multiplicando matriz A pelo valor escalar e cronometrando o tempo
	gettimeofday(&start_scalar_mult, NULL);
	printf("\nMultiplicando matriz A por %.3f...\n", scalar);
	error = scalar_matrix_mult_normal(scalar, mA);
	gettimeofday(&stop_scalar_mult, NULL);
	if(error == 0){
		printf("Erro função de multiplicar matriz A por escalar");
	}

	// Multiplicando matriz A pelo valor escalar utilizando avx e cronometrando o tempo
	gettimeofday(&start_scalar_mult_avx, NULL);
	printf("\nMultiplicando matriz A por %.3f utilizando avx...\n", scalar);
	error = scalar_matrix_mult_avx(scalar, mA);
	gettimeofday(&stop_scalar_mult_avx, NULL);
	if(error == 0){
		printf("Erro função de multiplicar matriz A por escalar utilizando avx");
	}

	// Multiplicando matriz A pelo valor escalar utilizando avx e pthread e cronometrando o tempo
	gettimeofday(&start_scalar_mult_avx_pthread, NULL);
	printf("\nMultiplicando matriz A por %.3f utilizando avx e pthread...\n", scalar);
	error = scalar_matrix_mult_avx_pthread(scalar, mA);
	gettimeofday(&stop_scalar_mult_avx_pthread, NULL);
	if(error == 0){
		printf("Erro função de multiplicar matriz A por escalar utilizando avx e pthread");
	}
	
	error = write_matrix_to_file(result1, mA);
	if(error == 0){
		printf("Erro ao escrever matriz no arquivo .dat");
	}
	
	// Multiplicando matriz A pela matriz B e cronometrando o tempo
	gettimeofday(&start_matrix_mult, NULL);
	printf("\nMultiplicando matriz A por matriz B...\n");
	error = matrix_matrix_mult_normal(mA, mB, mC);
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
	error = matrix_matrix_mult_otm_pthread(mA, mB, m_pthread_C);
	gettimeofday(&stop_matrix_mult_otm_pthread, NULL);
	if(error == 0){
		printf("Erro função de multiplar matriz A por matriz B (otimizado com pthread)");
	}

	// Multiplicando matriz A pela matriz B (otimizado com avx e pthread) e cronometrando o tempo
	gettimeofday(&start_matrix_mult_otm_avx_pthread, NULL);
	printf("\nMultiplicando matriz A por matriz B (Otimizada com avx e pthread)...\n");
	error = matrix_matrix_mult_otm_avx_pthread(mA, mB, m_avx_pthread_C);
	gettimeofday(&stop_matrix_mult_otm_avx_pthread, NULL);
	if(error == 0){
		printf("Erro função de multiplar matriz A por matriz B (otimizado com avx e pthread)");
	}

	// Comandos para debugging do resultado
	// matrix_print(mC, "C");
	// matrix_print(m_otm_C, "C");
	// matrix_print(m_avx_C, "C calculada com avx");
	// matrix_print(m_pthread_C, "C calculada com pthreads");
	// matrix_print(m_avx_pthread_C, "C calculada com avx e pthreads");

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
	free(m_pthread_C);
	free(m_avx_pthread_C);
	
	gettimeofday(&overall_t2, NULL);

	overall_time = timedifference_msec(overall_t1, overall_t2);
	matrix_scalar_time = timedifference_msec(start_scalar_mult, stop_scalar_mult);
	matrix_scalar_time_avx = timedifference_msec(start_scalar_mult_avx, stop_scalar_mult_avx);
	matrix_scalar_time_avx_pthread = timedifference_msec(start_scalar_mult_avx_pthread, stop_scalar_mult_avx_pthread);
	matrix_mult_time = timedifference_msec(start_matrix_mult, stop_matrix_mult);
	matrix_mult_time_otm = timedifference_msec(start_matrix_mult_otm, stop_matrix_mult_otm);
	matrix_mult_time_otm_avx = timedifference_msec(start_matrix_mult_otm_avx, stop_matrix_mult_otm_avx);
	matrix_mult_time_otm_pthread = timedifference_msec(start_matrix_mult_otm_pthread, stop_matrix_mult_otm_pthread);
	matrix_mult_time_otm_avx_pthread = timedifference_msec(start_matrix_mult_otm_avx_pthread, stop_matrix_mult_otm_avx_pthread);
	matrix_read_time = timedifference_msec(start_matrix_read_time, stop_matrix_read_time);

	printf("\n==========================================================================================================================");
	printf("\nTempo total para leitura das matrizes: (%f ms) | (%f s) | (%f min)\n", matrix_read_time, matrix_read_time/1000, matrix_read_time/60000);
	
	printf("\nTempo total para multiplicacao escalar: (normal ----------) (%f ms) | (%f s) | (%f min)\n", matrix_scalar_time, matrix_scalar_time/1000, matrix_scalar_time/60000);
	printf("Tempo total para multiplicacao escalar: (com avx ---------) (%f ms) | (%f s) | (%f min)\n", matrix_scalar_time_avx, matrix_scalar_time_avx/1000, matrix_scalar_time_avx/60000);
	printf("Tempo total para multiplicacao escalar: (com avx e pthread) (%f ms) | (%f s) | (%f min)\n\n", matrix_scalar_time_avx_pthread, matrix_scalar_time_avx_pthread/1000, matrix_scalar_time_avx_pthread/60000);

	printf("Tempo total para multiplicacao de matrizes A por B: (normal --------------------): (%f ms) | (%f s) | (%f min)\n", matrix_mult_time, matrix_mult_time/1000, matrix_mult_time/60000);
	printf("Tempo total para multiplicacao de matrizes A por B: (otimizado -----------------): (%f ms) | (%f s) | (%f min)\n", matrix_mult_time_otm, matrix_mult_time_otm/1000, matrix_mult_time_otm/60000);
	printf("Tempo total para multiplicacao de matrizes A por B: (otimizado com avx----------): (%f ms) | (%f s) | (%f min)\n", matrix_mult_time_otm_avx, matrix_mult_time_otm_avx/1000, matrix_mult_time_otm_avx/60000);
	printf("Tempo total para multiplicacao de matrizes A por B: (otimizado com pthread------): (%f ms) | (%f s) | (%f min)\n", matrix_mult_time_otm_pthread, matrix_mult_time_otm_pthread/1000, matrix_mult_time_otm_pthread/60000);
	printf("Tempo total para multiplicacao de matrizes A por B: (otimizado com avx e pthread): (%f ms) | (%f s) | (%f min)\n", matrix_mult_time_otm_avx_pthread, matrix_mult_time_otm_avx_pthread/1000, matrix_mult_time_otm_avx_pthread/60000);

	printf("\nTempo total do programa: (%f ms) | (%f s) | (%f min)\n", overall_time, overall_time/1000, overall_time/60000);
	printf("==========================================================================================================================\n");

	pthread_exit(NULL);

	return 0;
}
