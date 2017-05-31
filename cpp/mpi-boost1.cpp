#include <iostream>
#include <cstring>
#include <string>
using namespace std;

//#include <mpich2/mpi.h>
#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
namespace mpi = boost::mpi;

int main(int argc, char *argv[]) {
   mpi::environment env(argc, argv);
   mpi::communicator world;
   if( world.rank() == 0 ) {
      for( int i = 1; i < world.size(); i++ ) {
         world.isend(i, 0, string("hello"));
      }
      for( int i = 1; i < world.size(); i++ ) {
         string receivemessage;
         world.recv(i, 0, receivemessage );
         cout << receivemessage << endl;
      }
   } else {
      string receivemessage;
      world.recv(0, 0, receivemessage );
      cout << receivemessage << endl;
      world.isend(0,0,string("reply from child"));
   }
   return 0;
}

