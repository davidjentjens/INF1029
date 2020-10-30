#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUM_LEITORES	5
#define NUM_ESCRITORES	5
#define NUM_RODADAS	5

pthread_mutex_t mutex_acesso_bd;
pthread_mutex_t mutex_leitores;
pthread_mutex_t mutex_fila;

unsigned int leitores = 0;

void le_bd(long tid) {
   printf("Leitor %ld lendo do banco de dados...\n",tid);
   sleep(1);
}

void escreve_bd(long tid) {
   printf("Escritor %ld escrevendo no banco de dados...\n",tid);
   sleep(2);
}

void consome_dado(long tid) {
   printf("Leitor %ld consumindo dado lido do banco de dados...\n",tid);
   sleep(1);
}

void produz_dado(long tid) {
   printf("Escritor %ld produzindo dado para escrever no banco de dados...\n",tid);
   sleep(2);
}

void *leitor(void *t)
{
   int i;
   long tid;
   tid = (long)t;

   for (i=0; i<NUM_RODADAS; ++i) {
      pthread_mutex_lock(&mutex_fila);

      pthread_mutex_lock(&mutex_leitores);
   
      if(leitores == 0){
         // O primeiro dos leitores tem que bloquear o bd, 
         // para garantir que ninguém vai escrever enquanto tem gente lendo
         printf("Leitor %ld vai tentar bloquear o banco de dados...\n",tid);
         pthread_mutex_lock(&mutex_acesso_bd);
         printf("Leitor %ld bloqueou o banco de dados...\n",tid);
      }
      leitores++;
      
      pthread_mutex_unlock(&mutex_fila);
      pthread_mutex_unlock(&mutex_leitores);

      le_bd(tid);

      // Precisa bloquear outros leitores, pois mais de um pode tentar modificar
      // a quantidade de leitores
      pthread_mutex_lock(&mutex_leitores);
      leitores--;

      if(leitores == 0){
         printf("Leitor %ld vai desbloquear o banco de dados...\n",tid);
         pthread_mutex_unlock(&mutex_acesso_bd);
      }

      pthread_mutex_unlock(&mutex_leitores);      
      consome_dado(tid);
   }

   pthread_exit((void*) t);
}

void *escritor(void *t)
{
   int i;
   long tid;
   tid = (long)t;

   for (i=0; i<NUM_RODADAS; ++i) {
      produz_dado(tid);

      printf("Escritor %ld vai tentar bloquear fila\n", tid);
      pthread_mutex_lock(&mutex_fila);
      printf("Escritor %ld bloqueou a fila\n", tid);
      
      printf("Escritor %ld vai tentar bloquear o banco de dados...\n",tid);
      pthread_mutex_lock(&mutex_acesso_bd);
      printf("Escritor %ld bloqueou o banco de dados...\n",tid);

      printf("Escritor %ld vai desbloquear fila\n", tid);
      pthread_mutex_unlock(&mutex_fila);

      escreve_bd(tid);

      printf("Escritor %ld vai desbloquear o banco de dados...\n",tid);
      pthread_mutex_unlock (&mutex_acesso_bd);
   }

   pthread_exit((void*) t);
}

int main (int argc, char *argv[])
{
   pthread_t thread_leitor[NUM_LEITORES];
   pthread_t thread_escritor[NUM_ESCRITORES];
   pthread_attr_t attr;
   int rc;
   long t;
   void *status;

   /* Initialize mutex semaphore */
   pthread_mutex_init(&mutex_acesso_bd, NULL);
   pthread_mutex_init(&mutex_leitores, NULL);
   pthread_mutex_init(&mutex_fila, NULL);

   /* Initialize and set thread detached attribute */
   pthread_attr_init(&attr);
   pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

   for(t=0; t<NUM_LEITORES; t++) {
      printf("Main: criando o Leitor %ld\n", t);
      rc = pthread_create(&thread_leitor[t], &attr, leitor, (void *)t);  
      if (rc) {
         printf("ERROR; return code from pthread_create() is %d\n", rc);
         exit(-1);
         }
      sleep(1);
   }

   for(t=0; t<NUM_ESCRITORES; t++) {
      printf("Main: criando o Escritor %ld\n", t);
      rc = pthread_create(&thread_escritor[t], &attr, escritor, (void *)t);  
      if (rc) {
         printf("ERROR; return code from pthread_create() is %d\n", rc);
         exit(-1);
         }
      sleep(1);
   }

    /* Free attribute and wait for the other threads */
   pthread_attr_destroy(&attr);

   for(t=0; t<NUM_LEITORES; t++) {
   rc = pthread_join(thread_leitor[t], &status);
   if (rc) {
      printf("ERROR; return code from pthread_join() is %d\n", rc);
         exit(-1);
      }
      printf("Main: completou o join do Leitor %ld com status de %ld\n",t,(long)status);
   }

   for(t=0; t<NUM_ESCRITORES; t++) {
      rc = pthread_join(thread_escritor[t], &status);
      if (rc) {
         printf("ERROR; return code from pthread_join() is %d\n", rc);
         exit(-1);
         }
      printf("Main: completou o join do Escritor %ld com status de %ld\n",t,(long)status);
   }
 
   printf("Main: Programa completado. Saindo.\n");
   pthread_mutex_destroy(&mutex_acesso_bd);
   pthread_exit(NULL);
}
