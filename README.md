# INF1029
Repositório para trabalhos da disciplina INF1029 - Introdução a Arquitetura de Computadores



Tempo de execução usando matriz 1024x1024

>> gcc -mfma -std=c11 -pthread -Wall -o matrix_lib_test matrix_lib_test.c matrix_lib.c timer.c
>> ./matrix_lib_test 5.0 1024 1024 1024 1024 floats_256_2.0f.dat floats_256_5.0f.dat result1.dat result2.dat
>> 

Tempo total do programa: (9146.563477 ms) | (9.146564 s) | (0.152443 min)
Tempo total para multiplicação escalar: (2.564000 ms) | (0.002564 s) | (0.000043 min)
Tempo total para multiplicação de matrizes A por B: (8871.642578 ms) | (8.871642 s) | (0.147861 min)
