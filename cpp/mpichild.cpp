#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <fcntl.h>
using namespace std;

#include "mpi.h"

#include "utils/mpi/mpi_helper.h"

void dumpfds() {
   cout << "rdonly " << O_RDONLY << " wronly " << O_WRONLY << " rdwr " << O_RDWR << endl;
   for( int i = 0; i < 30; i++ ) {
      int fd = fcntl(i, F_GETFD );
      int fl = fcntl(i, F_GETFL );
      int own = fcntl(i, F_GETOWN );
      int lease = fcntl(i, F_GETLEASE);
      cout << "fd " << i << " fl " << fl << " own " << own << " fd " << fd << " lease " << lease <<
        " sig " << fcntl(i, F_GETSIG ) << endl;
   }
}
int main(int argc, char *argv[] ) {
   cout << "child dump fds" << endl;
   dumpfds();
   string somestring;
   cin >> somestring;
   cout << "somestring: " << somestring << endl;
   int alreadyinitialized = false;
   MPI_Initialized(&alreadyinitialized);
   cout << "child already initizialized: " << alreadyinitialized << endl;
   if( !alreadyinitialized ) {
   cout << "child calling init" << endl;
   MPI_Init( &argc, &argv );
   }

   int rank, numprocs;
   MPI_Comm_rank( MPI_COMM_WORLD, &rank );
   MPI_Comm_size( MPI_COMM_WORLD, &numprocs );

   double value[1];
   value[0] = rank;
   double reducedarray[1];
   Allreduce_array( value, reducedarray, 1 );
   if( rank == 0 ) {
      cout << "Output: " << reducedarray[0] << endl;
   }

   if( !alreadyinitialized ) {
     cout << "child calling finalize" << endl;
     MPI_Finalize();
   }
   cout << "child ending" << endl;
   return 0;
}


