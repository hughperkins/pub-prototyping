#include <iostream>
using namespace std;

#include "utils/mpi/mpi_helper.h"

int main(int argc, char *argv[] ) {
   //MPI_Init(&argc, &argv);
   MPI_Init(0,0);

int rank, numprocs;
MPI_Comm_rank(MPI_COMM_WORLD, &rank );
MPI_Comm_size(MPI_COMM_WORLD, &numprocs );

   cout << "argc " << argc << endl;
for( int j = 0; j < numprocs; j++ ) {
if( rank == j ) {
for( int i = 0; i < argc; i++ ) {
   cout << i << " " << argv[i] << endl;
}
}
MPI_Barrier(MPI_COMM_WORLD );
}

   double somearray[2];
   somearray[0] = 123;
   somearray[1] = 567;
   double receivedarray[2];
   MPI_Allreduce( somearray, receivedarray, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
   cout << receivedarray[0] << " " << receivedarray[1] << endl;

   MPI_Finalize();
}

