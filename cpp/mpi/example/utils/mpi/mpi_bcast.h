#pragma once

#include <stdexcept>

#include "mpi.h"

// note to self: this is sub-optimal, because it sends two messages, but we can think about that later
template<typename T>
void Bcast_send( const T &object, MPI_Comm comm = MPI_COMM_WORLD ) {
    throw std::runtime_error("not implemented");
}

// assumes use with Bcast_send above   EmSelectFeatures(Params *params, Corpus *corpus );
template<typename T> void Bcast_recv( T &receivedObject, int source = 0, MPI_Comm comm = MPI_COMM_WORLD ) {
    throw std::runtime_error("not implemented");
}

template<> void Bcast_send( const int &value, MPI_Comm comm );
template<> void Bcast_recv( int &receivedInt, int source, MPI_Comm comm );

void Bcast_send_array( const double *array, int len, MPI_Comm comm = MPI_COMM_WORLD );
void Bcast_recv_array( double *array, int len, int target = 0, MPI_Comm comm = MPI_COMM_WORLD );

void Bcast_send_array( const int *array, int len, MPI_Comm comm = MPI_COMM_WORLD );
void Bcast_recv_array( int *array, int len, int target = 0, MPI_Comm comm = MPI_COMM_WORLD );

void Bcast_send_longarray( const long *array, int len, MPI_Comm comm = MPI_COMM_WORLD );
void Bcast_recv_longarray( long *array, int len, int target = 0, MPI_Comm comm = MPI_COMM_WORLD );

void Bcast_send_array( const char *array, int len, MPI_Comm comm = MPI_COMM_WORLD );
void Bcast_recv_array( char *array, int len, int target = 0, MPI_Comm comm = MPI_COMM_WORLD );

