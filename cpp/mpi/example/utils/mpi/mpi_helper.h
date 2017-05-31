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


#pragma once

#include <vector>
#include <stdexcept>

#include "mpi.h"

#include "utils/mpi/mpi_reduce.h"
#include "utils/mpi/mpi_bcast.h"

template<typename T> void Isend( const T &object, int target, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD ) {
    throw std::runtime_error("not implemented");
}

template<typename T> void Recv( T &receivedObject, int source, int tag = MPI_ANY_TAG, MPI_Comm comm = MPI_COMM_WORLD ) {
    throw std::runtime_error("not implemented");
}

void Isend_array( const double *array, int len, int target = 0, int tag = 0, MPI_Comm comm =  MPI_COMM_WORLD );
void Recv_array( double *array, int len, int source, int tag = MPI_ANY_TAG, MPI_Comm comm = MPI_COMM_WORLD );

void Isend_array( const int *array, int len, int target = 0, int tag = 0, MPI_Comm comm =  MPI_COMM_WORLD );
void Recv_array( int *array, int len, int source, int tag = MPI_ANY_TAG, MPI_Comm comm = MPI_COMM_WORLD );

template<> void Isend( const int &value, int target, int tag, MPI_Comm comm );
template<> void Recv( int &receivedInt, int source, int tag, MPI_Comm comm );

template<> void Isend( const int &value, int target, int tag, MPI_Comm comm );
template<> void Recv( int &receivedInt, int source, int tag, MPI_Comm comm );

void mpi_print( string message );

void mpi_get_rank_procs( int *prank, int *pnumprocs );
