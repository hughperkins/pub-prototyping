#include <iostream>
using namespace std;

#include "mpi.h"

int main(int argc, char *argv[] ) {
    MPI_Init(&argc, &argv );
    long long source = 1234567890000l;
    long long destination = 0;
    MPI_Allreduce( &source, &destination, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD );
    cout << "reduced: " << destination << endl;
    MPI_Finalize();
    return 0;
}

