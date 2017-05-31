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

#include "utils/mpi/mpi_helper.h"

template<> void Reduce_send( const vector<double> &doublevector, int target, MPI_Op op, MPI_Comm comm ) {
   // change into an array I guess :-(
   int len = doublevector.size();
   double *doublearray = new double[len];
   for( int i = 0; i < len; i++ ) {
      doublearray[i] = doublevector[i];
   }
   MPI_Reduce( doublearray, 0, len, MPI_DOUBLE, op, target, comm );
   delete[] doublearray;
}

template<> void Reduce_recv( vector<double> &doublevector, MPI_Op op, MPI_Comm comm ) {
   // change into an array I guess :-(
   int len = doublevector.size();
   double *doublearray = new double[len];
   for( int i = 0; i < len; i++ ) {
      doublearray[i] = doublevector[i];
   }
   double *receivearray = new double[len];
   int rank;
   MPI_Comm_rank(comm, &rank );
   MPI_Reduce( doublearray, receivearray, len, MPI_DOUBLE, op, rank, comm );
   for( int i = 0; i < len; i++ ) {
      doublevector[i] = receivearray[i];
   }
   delete[] doublearray;
   delete[] receivearray;
}

template<> void Reduce_send( const int &value, int target, MPI_Op op, MPI_Comm comm ) {
   MPI_Reduce( (void *)&value, 0, 1, MPI_INT, op, target, comm );
}

template<> void Reduce_recv( int &value, MPI_Op op, MPI_Comm comm ) {
   int receivevalue;
   int rank;
   MPI_Comm_rank(comm, &rank );
   MPI_Reduce( (void *)&value, &receivevalue, 1, MPI_INT, op, rank, comm );
   value = receivevalue;
}

template<> void Reduce( int *pvalue, MPI_Op op, int root, MPI_Comm comm ) {
   int receivevalue;
   //int rank;
   //MPI_Comm_rank(comm, &rank );
   MPI_Reduce( (void *)pvalue, &receivevalue, 1, MPI_INT, op, root, comm );
   *pvalue = receivevalue;
}

template<> void Reduce_array( const double *sendarray, double *receivearray, int len, MPI_Op op, int root, MPI_Comm comm ) {
   MPI_Reduce( (void *)sendarray, receivearray, len, MPI_DOUBLE, op, root, comm );
}

template<> void Reduce_array( const int *sendarray, int *receivearray, int len, MPI_Op op, int root, MPI_Comm comm ) {
   MPI_Reduce( (void *)sendarray, receivearray, len, MPI_INT, op, root, comm );
}

template<> void Allreduce( int &value, MPI_Op op, MPI_Comm comm ) {
   int receivevalue;
   MPI_Allreduce( (void *)&value, &receivevalue, 1, MPI_INT, op, comm );
   value = receivevalue;
}

void Allreduce_array( double *sendarray, double *receivearray, int len, MPI_Op op ) {
   MPI_Allreduce(sendarray, receivearray, len, MPI_DOUBLE, op, MPI_COMM_WORLD );
}

void Allreduce_array( int *sendarray, int *receivearray, int len, MPI_Op op ) {
   MPI_Allreduce(sendarray, receivearray, len, MPI_INT, op, MPI_COMM_WORLD );
}

