// (C) Copyright 2012, Jun Zhu (junzhu [at] cs [dot] cmu [dot] edu), Hugh Perkins (http://hughperkins.com)

// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA

#include <iostream>
#include <sstream>
using namespace std;
#include "mpi.h"

#include "mpi_helper.h"

template<> void Isend( const int &value, int target, int tag, MPI_Comm comm ) {
    MPI_Request request;
    int valuebuffer = value;
    MPI_Isend( (void *)&valuebuffer, 1, MPI_INT, target, tag, comm, &request );   
}

template<> void Recv( int &receivedInt, int source, int tag, MPI_Comm comm ) {
   MPI_Status status;
   MPI_Recv((void *)&receivedInt, 1, MPI_INT, source, tag, comm, &status );
}

void Isend_array( const double *array, int len, int target, int tag, MPI_Comm comm ) {
    MPI_Request request;
   MPI_Isend((void *)array, len, MPI_DOUBLE, target, tag, comm, &request );
}

void Recv_array( double *array, int len, int source, int tag, MPI_Comm comm ) {
   MPI_Status status;
   MPI_Recv((void *)array, len, MPI_DOUBLE, source, tag, comm, &status );
}

void Isend_array( const int *array, int len, int target, int tag, MPI_Comm comm ) {
   MPI_Request request;
   MPI_Isend((void *)array, len, MPI_INT, target, tag, comm, &request );
}

void Recv_array( int *array, int len, int source, int tag, MPI_Comm comm ) {
   MPI_Status status;
   MPI_Recv((void *)array, len, MPI_INT, source, tag, comm, &status );
}

void mpi_print( string message ) {
   int rank, numprocs;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
   MPI_Comm_size(MPI_COMM_WORLD, &numprocs); 
   MPI_Barrier(MPI_COMM_WORLD);
   for( int i = 0; i < numprocs; i++ ) {
      if( rank == i ) {
         cout << i << ": " << message << endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
   }
}

void mpi_get_rank_procs( int *rank, int *numprocs ) {
   MPI_Comm_rank(MPI_COMM_WORLD, rank);
   MPI_Comm_rank(MPI_COMM_WORLD, numprocs);
}

