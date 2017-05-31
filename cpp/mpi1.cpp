#include <iostream>
#include <cstring>
#include <string>
using namespace std;

#include <mpich2/mpi.h>

template<typename T>
struct Vector<T> {
   T[] values;
   int size;
};

void send(int node, string message ) {
   MPI_Send((void *)message.c_str(), message.length(), MPI_CHAR, node, 0, MPI_COMM_WORLD );
}

void sendDoubleVector(int node, vector<double> doublevector ) {
   double[] 
   MPI_Send((void *)doubles, size, MPI_DOUBLE, node, 0, MPI_COMM_WORLD );
}

double *receiveDoubleArray( int node, int &psize ) {
   MPI_Status stat;
   MPI_Recv(receiveBuffer,16384, MPI_CHAR, node, 0, MPI_COMM_WORLD, &stat );
   return string(receiveBuffer);
}

string receive( int node ) {
   MPI_Status stat;
   char receiveBuffer[16384];
   MPI_Recv(receiveBuffer,16384, MPI_CHAR, node, 0, MPI_COMM_WORLD, &stat );
   return string(receiveBuffer);
}

int main(int argc, char *argv[]) {
   MPI_Init(&argc, &argv );
   int numprocs, rank;
   MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   if( rank == 0 ) {
      cout << (rank+1) << "/" << numprocs << endl;
      for( int i = 1; i < numprocs; i++ ) {
         send( i, "hey!");
      }
      for( int i = 1; i < numprocs; i++ ) {
         cout << receive(i) << endl;
      }
   } else {
       receive(0);
       //cout << receive(0) << endl;
       send(0, "hi from child" );
       double *matrix = new double[2*2];
       for( int x = 0; x < 2; x++ ) {
		    for( int y = 0; y < 2; y++ ) {
             matrix[y* 2 + x] = x * y;
		    }
       }
       
   }
   MPI_Finalize();
   return 0;
}

