#include <iostream>
using namespace std;

#include "mpi.h"
#include "utils/mpi/mpi_helper.h"
#include "utils/stringhelper.h"

int main( int argc, char * argv[] ) {
   MPI_Init(&argc, &argv);
   
   cout << "hello" <<endl;
   mpi_print("hello");
   
   int rank, size;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   mpi_print("rank: " + toString(rank) + " / " + toString(size) );
   if( rank == 0 ){
      cout << "hello from master" << endl;
   }
   
   double somearray[3];
   somearray[0] = rank + 57;
   mpi_print(toString(somearray[0]) );
   MPI_Bcast( somearray, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   mpi_print(toString(somearray[0]) );
   
   somearray[0] = rank;
   somearray[1] = rank * 2;
   double receivearray[3];
   mpi_print(toString(somearray[0]) + " " + toString( somearray[1] ) );
   MPI_Allreduce( somearray, receivearray, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);   
   mpi_print(toString(receivearray[0]) + " " + toString(receivearray[1]) );
   
   for( int i = 0; i < size; i++ ) {
      MPI_Barrier(MPI_COMM_WORLD);
      if( i == rank ) {
         cout << "hello from " << rank << endl;
      }
   }
   MPI_Barrier(MPI_COMM_WORLD);
   
   int N = 3;
   double base_1d[N*N];
   double *contained_2d[N];
   for( int i = 0; i < N; i++ ) {
      contained_2d[i] = &(base_1d[i*N]);
   }
   double base_1d_reduced[N*N];
   double *contained_2d_reduced[N];
   for( int i = 0; i < N; i++ ) {
      contained_2d_reduced[i] = &(base_1d_reduced[i*N]);
   }
   for( int i = 0; i < N; i++ ) {
      for( int j = 0; j < N; j++ ) {
         contained_2d[i][j] = i * j;
      }
   }
   if( rank == 0 ) {
      for( int i = 0; i < N; i++ ) {
         for( int j = 0; j < N; j++ ) {
            cout << contained_2d[i][j] << " ";
         }
         cout << endl;
      }
   }
   MPI_Reduce( base_1d, base_1d_reduced, N * N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);   
   if( rank == 0 ) {
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

