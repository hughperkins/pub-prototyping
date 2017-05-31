#include <iostream>
using namespace std;

#include "utils/mpi/mpi_helper.h"

int main(int argc, char *argv[] ) {
   //MPI_Init(&argc, &argv);
   MPI_Init(0,0);

   cout << "argc " << argc << endl;
   cout << argv[0] << endl;

   double somearray[2];
   somearray[0] = 123;
   somearray[1] = 567;
   Isend_array( somearray, 2, 0 );
somearray[0] = 2;
   Isend_array( somearray, 2, 0 );
somearray[0] = 3;
   Isend_array( somearray, 2, 0 );

   double receivedarray[2];

   receivedarray[0] = receivedarray[1] = 0;
   Recv_array(receivedarray, 2, 0 );
   cout << receivedarray[0] << " " << receivedarray[1] << endl;
   receivedarray[0] = receivedarray[1] = 0;
   Recv_array(receivedarray, 2, 0 );
   cout << receivedarray[0] << " " << receivedarray[1] << endl;
   receivedarray[0] = receivedarray[1] = 0;
   Recv_array(receivedarray, 2, 0 );
   cout << receivedarray[0] << " " << receivedarray[1] << endl;

   MPI_Finalize();
}

