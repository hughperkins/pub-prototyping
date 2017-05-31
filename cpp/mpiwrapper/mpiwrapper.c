#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mpi.h"

#include "mpiwrapper.h"

void MPIWrapper_Initialize() {
   MPI_Init(0,0);
}

void MPIWrapper_Finalize(){
   MPI_Finalize();
}

int MPIWrapper_getRank() {
   int rank;
   MPI_Comm_rank( MPI_COMM_WORLD, &rank );
   return rank;
}

int MPIWrapper_getSize() {
   int size;
   MPI_Comm_size( MPI_COMM_WORLD, &size );
   return size;
}

int MPIWrapper_MPI_SUM() {
   return MPI_SUM;
}

int MPIWrapper_MPI_MAX() {
   return MPI_MAX;
}

void MPIWrapper_Allreduce_double_( double *inarray, int arraylength, int op ) {
   int i = 0;
   double *outarray = (double *)malloc(sizeof( double) * arraylength );
   MPI_Allreduce(inarray, outarray, arraylength, MPI_DOUBLE, op, MPI_COMM_WORLD );
   for( i = 0; i < arraylength; i++ ) {
      inarray[i] = outarray[i];
   }
   free( outarray );
}

void MPIWrapper_Allreduce_int_( int *inarray, int arraylength, int op ) {
   int i = 0;
   int *outarray = (int *)malloc(sizeof( int) * arraylength );
   MPI_Allreduce(inarray, outarray, arraylength, MPI_INT, op, MPI_COMM_WORLD );
   for( i = 0; i < arraylength; i++ ) {
      inarray[i] = outarray[i];
   }
   free( outarray );
}

void MPIWrapper_Reduce_double_( double *inarray, int arraylength, int op ) {
   int i = 0;
   double *outarray = (double *)malloc(sizeof( double) * arraylength );
   MPI_Reduce(inarray, outarray, arraylength, MPI_DOUBLE, op, 0, MPI_COMM_WORLD );
   for( i = 0; i < arraylength; i++ ) {
      inarray[i] = outarray[i];
   }
   free( outarray );
}

void MPIWrapper_Reduce_int_( int *inarray, int arraylength, int op ) {
   int i = 0;
   int *outarray = (int *)malloc(sizeof( int) * arraylength );
   MPI_Reduce(inarray, outarray, arraylength, MPI_INT, op, 0, MPI_COMM_WORLD );
   for( i = 0; i < arraylength; i++ ) {
      inarray[i] = outarray[i];
   }
   free( outarray );
}

void MPIWrapper_Bcast_double_( double *inarray, int arraylength ) {
   int i = 0;
   MPI_Bcast(inarray, arraylength, MPI_DOUBLE, 0, MPI_COMM_WORLD );
}

void MPIWrapper_Bcast_int_( int *inarray, int arraylength ) {
   int i = 0;
   MPI_Bcast(inarray, arraylength, MPI_INT, 0, MPI_COMM_WORLD );
}

