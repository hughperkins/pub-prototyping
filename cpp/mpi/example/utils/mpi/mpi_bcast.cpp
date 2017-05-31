#include "mpi.h"

#include "utils/mpi/mpi_bcast.h"

void Bcast_send_array( const double *array, int len, MPI_Comm comm ) {
   int rank;
   MPI_Comm_rank(comm, &rank); // note to self: maybe cache this value
   MPI_Bcast( (void *)array, len, MPI_DOUBLE, rank, comm );      
}

void Bcast_recv_array( double *array, int len, int target, MPI_Comm comm ) {
   MPI_Bcast( array, len, MPI_DOUBLE, target, comm );      
}

void Bcast_send_array( const int *array, int len, MPI_Comm comm ) {
   int rank;
   MPI_Comm_rank(comm, &rank); // note to self: maybe cache this value
   MPI_Bcast( (void *)array, len, MPI_INT, rank, comm );      
}

void Bcast_recv_array( int *array, int len, int target, MPI_Comm comm ) {
   MPI_Bcast( array, len, MPI_INT, target, comm );      
}

void Bcast_send_longarray( const long int *array, int len, MPI_Comm comm ) {
   int rank;
   MPI_Comm_rank(comm, &rank); // note to self: maybe cache this value
   MPI_Bcast( (void *)array, len, MPI_LONG, rank, comm );      
}

void Bcast_recv_longarray( long int *array, int len, int target, MPI_Comm comm ) {
   MPI_Bcast( array, len, MPI_LONG, target, comm );      
}

void Bcast_send_array( const char *array, int len, MPI_Comm comm ) {
   int rank;
   MPI_Comm_rank(comm, &rank); // note to self: maybe cache this value
   MPI_Bcast( (void *)array, len, MPI_CHAR, rank, comm );      
}

void Bcast_recv_array( char *array, int len, int target, MPI_Comm comm ) {
   MPI_Bcast( array, len, MPI_CHAR, target, comm );      
}

template<> void Bcast_send( const int &value, MPI_Comm comm ) {
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank); // note to self: maybe cache this value
   int valuebuffer = value;
   MPI_Bcast( &valuebuffer, 1, MPI_INT, rank, comm );   
}

template<> void Bcast_recv( int &receivedInt, int source, MPI_Comm comm ) {
   MPI_Bcast( &receivedInt, 1, MPI_INT, source, comm );   
}

