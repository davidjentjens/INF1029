#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_THREADS  3
#define TCOUNT 10
#define COUNT_LIMIT 12

int count = 0;
long thread_ids[3] = {0,1,2};
pthread_mutex_t count_mutex;
pthread_cond_t count_threshold_cond_var;

void * inc_count(void *t) {
   int i;
   long my_id = (long)t;

   for (i=0; i<TCOUNT; i++) {
     pthread_mutex_lock(&count_mutex);
     count++;

     /* 
      * Check the value of count and signal waiting thread when condition is
      * reached.  Note that this occurs while mutex is locked. 
     */
     if (count == COUNT_LIMIT) {
       pthread_cond_signal(&count_threshold_cond_var);
       printf("inc_count(): thread %ld, count = %d  Threshold reached.\n", 
              my_id, count);
     }
     printf("inc_count(): thread %ld, count = %d, unlocking mutex\n", 
	    my_id, count);
     pthread_mutex_unlock(&count_mutex);

     /* Hold on for 1 second so threads can alternate on mutex lock */
     sleep(1);
   }
   pthread_exit(NULL);
}

void *watch_count(void *t) {
   long my_id = (long)t;

   printf("Starting watch_count(): thread %ld\n", my_id);

   /*
    * Lock mutex and wait for signal.  Note that the pthread_cond_wait 
    * routine will automatically and atomically unlock mutex while it waits. 
    * Also, note that if COUNT_LIMIT is reached before this routine is run by
    * the waiting thread, the loop will be skipped to prevent pthread_cond_wait
    * from never returning. 
   */
   pthread_mutex_lock(&count_mutex);
   while (count<COUNT_LIMIT) {
     pthread_cond_wait(&count_threshold_cond_var, &count_mutex);
     printf("watch_count(): thread %ld Condition signal received.\n", my_id);
   }
   count += 125;
   printf("watch_count(): thread %ld count now = %d.\n", my_id, count);
   pthread_mutex_unlock(&count_mutex);
   pthread_exit(NULL);
}

int main (int argc, char *argv[]) {
   int i, rc;
   pthread_t threads[3];
   pthread_attr_t attr;

   /* Initialize mutex and condition variable objects */
   pthread_mutex_init(&count_mutex, NULL);
   pthread_cond_init (&count_threshold_cond_var, NULL);

   /* For portability, explicitly create threads in a joinable state */
   pthread_attr_init(&attr);
   pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
   if (rc = pthread_create(&threads[0], &attr, watch_count, (void *)thread_ids[0])) {
	printf("ERROR; return code from pthread_create() is %d\n", rc);
        exit(-1);
   }
   if (rc = pthread_create(&threads[1], &attr, inc_count, (void *)thread_ids[1])) {
	printf("ERROR; return code from pthread_create() is %d\n", rc);
        exit(-1);
   }
   if (rc = pthread_create(&threads[2], &attr, inc_count, (void *)thread_ids[2])) {
	printf("ERROR; return code from pthread_create() is %d\n", rc);
        exit(-1);
   }

   /* Wait for all threads to complete */
   for (i=0; i<NUM_THREADS; i++) {
     if (rc = pthread_join(threads[i], NULL)) {
	printf("ERROR; return code from pthread_join() is %d\n", rc);
        exit(-1);
     }
   }
   printf ("Main(): Waited on %d  threads. Done.\n", NUM_THREADS);

   /* Clean up and exit */
   pthread_attr_destroy(&attr);
   pthread_mutex_destroy(&count_mutex);
   pthread_cond_destroy(&count_threshold_cond_var);
   pthread_exit(NULL);
}
