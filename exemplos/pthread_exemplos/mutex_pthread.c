#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

/* This global structure will be accesse by the threads.
 * The arrays (a and b) will be split in slices for each thread.
 * The sum field is the critical region and will be protected by 
 * a mutex variable (semaphore).
 */  
 typedef struct 
  {
    double      *a;
    double      *b;
    double      sum; 
    int    slicelen; 
  } DOTDATA;

/* Define globally accessible variables and a mutex */

#define NUMTHRDS 4
#define SLICELEN 100

DOTDATA dotstr; 
pthread_t threads[NUMTHRDS];
pthread_mutex_t mutexsum;

/* This function will be executed by each thread. It calculates
 * the product of each element of a slice of the arrays a and b
 * and calculates the sum (mysum) of the result of each product.
 * Then, the critical region is accessed in order to update the
 * global sum.
 */

 void *dotprod(void *arg) {
    int i, start, end, len ;
    long offset;
    double mysum, *x, *y;

    /* Calculate the boundary of the slice of this thread */
    offset = (long)arg;
     
    len = dotstr.slicelen;
    start = offset*len;
    end   = start + len;
    x = dotstr.a;
    y = dotstr.b;

    mysum = 0;
    for (i=start; i<end ; i++) 
       mysum += (x[i] * y[i]);

/* Lock a mutex prior to updating the value in the shared
 * structure, and unlock it upon updating.
 */
    pthread_mutex_lock (&mutexsum);
    dotstr.sum += mysum;
    pthread_mutex_unlock (&mutexsum);

    pthread_exit((void*) 0);
}

int main (int argc, char *argv[])
{
    int rc;
    long i;
    double *a, *b;
    void *status;
    pthread_attr_t attr;  

    /* Assign storage and initialize values */
    a = (double*) malloc (NUMTHRDS*SLICELEN*sizeof(double));
    b = (double*) malloc (NUMTHRDS*SLICELEN*sizeof(double));
   
    for (i=0; i<SLICELEN*NUMTHRDS; i++) {
      a[i]=1.0;
      b[i]=a[i];
    }

    dotstr.slicelen = SLICELEN; 
    dotstr.a = a; 
    dotstr.b = b; 
    dotstr.sum=0;

    pthread_mutex_init(&mutexsum, NULL);
         
    /* Create threads to perform the dotproduct  */
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    for(i=0; i<NUMTHRDS; i++) {
     /* Each thread works on a different set of data. The offset is specified 
      * by 'i'. The size of the data for each thread is indicated by SLICELEN.
      */
     if (rc = pthread_create(&threads[i], &attr, dotprod, (void *)i)) {
          printf("ERROR; return code from pthread_create() is %d\n", rc);
          exit(-1);
     }
    }

    pthread_attr_destroy(&attr);

    /* Wait on the other threads */
    for(i=0; i<NUMTHRDS; i++) {
       if (rc = pthread_join(threads[i], &status)) {
          printf("ERROR; return code from pthread_join() is %d\n", rc);
          exit(-1);
       }
    }

    /* After joining, print out the results and cleanup */
    printf ("Sum =  %f \n", dotstr.sum);
    free (a);
    free (b);
    pthread_mutex_destroy(&mutexsum);
    pthread_exit(NULL);
}
