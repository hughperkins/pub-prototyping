#include <iostream>
using namespace std;

#include "mpi.h"
#include "utils/mpi/mpi_helper.h"
#include "utils/stringhelper.h"

int main(int argc, char *argv[] ) {
    MPI_Init(&argc, &argv );

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    
    if( rank == 0 ) {
      cout << "Hey!" << endl;
      cout << "Node " << rank << " of " << size << endl;
    }
    
    double somearray[1];
    somearray[0] =  rank + 57;
    mpi_print( "before bcast " + toString( rank ) + " somearray[0] " + toString( somearray[0] ) );
    MPI_Bcast(somearray, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD );
    mpi_print( "after bcast " + toString( rank ) + " somearray[0] " + toString( somearray[0] ) );
    
    somearray[0] = rank;
    double reducedarray[1];
    mpi_print( "before allreduce " + toString( rank ) + " somearray[0] " + toString( somearray[0] ) + " reducedarray[0] " + toString( reducedarray[0] ) );
    MPI_Allreduce(somearray, reducedarray, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
    mpi_print( "after allreduce " + toString( rank ) + " somearray[0] " + toString( somearray[0] ) + " reducedarray[0] " + toString( reducedarray[0] ) );
    
    for( int i = 0; i < size; i++ ) {
      MPI_Barrier(MPI_COMM_WORLD);
      if( rank == i ) {
         cout << "hey! from node " << rank << endl;
      }
    }
    
    int N = 3;
   double base_1d[N*N];
   double base_1d_reduced[N*N];
   double *contained_2d[N];
   double *contained_2d_reduced[N];
   for( int i = 0; i < N; i++ ) {
      contained_2d[i] = &(base_1d[i*N]);
      contained_2d_reduced[i] = &(base_1d_reduced[i*N]);
   }
   
   for( int i = 0; i < N; i++ ) {
      for( int j = 0; j < N; j++ ) {
         contained_2d[i][j] = i * j * rank;
      }
   }
   
   if( rank == 0 ) {
      cout << "before reduce" << endl;
      for( int i = 0; i < N; i++ ) {
         for( int j = 0; j < N; j++ ) {
            cout << contained_2d_reduced[i][j] << " ";
         }
         cout << endl;
      }
   }
    MPI_Allreduce(base_1d, base_1d_reduced,N * N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
   if( rank == 0 ) {
      cout << "after reduce" << endl;
      for( int i = 0; i < N; i++ ) {
         for( int j = 0; j < N; j++ ) {
            cout << contained_2d_reduced[i][j] << " ";
         }
         cout << endl;
      }
   }
    
    MPI_Finalize();
    return 0;
}

