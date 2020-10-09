#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_THREADS 5

struct thread_data {
   long thread_id;
   int  offset;
   char *message;
};

void *PrintHello(void *threadarg) {
    struct thread_data *my_data;
    my_data = (struct thread_data*) threadarg;

    printf("Thread #%ld: %s\n", my_data->thread_id, my_data->message+my_data->offset);

    pthread_exit(NULL);
}

int main (int argc, char *argv[]) {
    pthread_t threads[NUM_THREADS];
    int rc;
    long t;
    struct thread_data thread_data_array[NUM_THREADS];
    char message[]="ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    for(t=0; t<NUM_THREADS; t++){
       printf("In main: creating thread %ld\n", t);
       thread_data_array[t].thread_id = t;
       thread_data_array[t].offset = t * 4;
       thread_data_array[t].message = message; 

       rc = pthread_create(&threads[t], NULL, PrintHello, (void *)&thread_data_array[t]);
       if (rc) {
          printf("ERROR; return code from pthread_create() is %d\n", rc);
          exit(-1);
       }
    }

    /* Last thing that main() should do */
    pthread_exit(NULL);
}

// gcc -pthread -Wall -o args_pthread args_pthread.c 