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

template<typename T> void Reduce_send( const T &object, int target = 0, MPI_Op op = MPI_SUM, MPI_Comm comm = MPI_COMM_WORLD ) {
   throw std::runtime_error("unimplemented");
}
template<typename T> void Reduce_recv( T &object, MPI_Op op = MPI_SUM, MPI_Comm comm = MPI_COMM_WORLD ) {
   throw std::runtime_error("unimplemented");
}
template<> void Reduce_send( const vector<double> &doublevector, int target, MPI_Op op, MPI_Comm comm );
template<> void Reduce_recv( vector<double> &doublevector, MPI_Op op, MPI_Comm comm );
template<> void Reduce_send( const int &value, int target, MPI_Op op, MPI_Comm comm );
template<> void Reduce_recv( int &value, MPI_Op op, MPI_Comm comm );

template<typename T> void Reduce( T *object, MPI_Op op = MPI_SUM, int root = 0, MPI_Comm comm = MPI_COMM_WORLD ) {
   throw std::runtime_error("unimplemented");
}
template<> void Reduce( int *value, MPI_Op op, int root, MPI_Comm comm );

template<typename T> void Reduce_array( const T *sendarray, T *receivearray, int len, MPI_Op op = MPI_SUM, int root = 0, MPI_Comm comm = MPI_COMM_WORLD );
template<> void Reduce_array( const int *sendarray, int *receivearray, int len, MPI_Op op, int root, MPI_Comm comm );
template<> void Reduce_array( const double *sendarray, double *receivearray, int len, MPI_Op op, int root, MPI_Comm comm );

template<typename T> void Allreduce( T &object, MPI_Op op = MPI_SUM, MPI_Comm comm = MPI_COMM_WORLD ) {
   throw std::runtime_error("unimplemented");
}
template<> void Allreduce( int &value, MPI_Op op, MPI_Comm comm );

void Allreduce_array( double *sendarray, double *reducedarray, int len, MPI_Op op = MPI_SUM );
void Allreduce_array( int *sendarray, int *reducedarray, int len, MPI_Op op = MPI_SUM );

