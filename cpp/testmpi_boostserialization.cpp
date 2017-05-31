#include <iostream>
#include <sstream>
using namespace std;

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/version.hpp>

#include "mpi.h"

#include "memoryanalysis.cpp"
#include "MyException.h"

class SomeRequest {
public:
   string name;
   int rows;
   int cols;
   vector<double> matrix;

   friend class boost::serialization::access;
   template<class Archive>
   void serialize(Archive & ar, const unsigned int version)
   {
      ar & name;
      ar & rows;
      ar & matrix;
   }

   SomeRequest(){
      rows = 0;
      cols = 0;
   }
   ~SomeRequest(){
   }
};

class InputMemoryStream : public basic_streambuf<char>
{
public:
  InputMemoryStream(char* p, size_t n) {
    setg(p, p, p + n);
  }
};

template<typename T>
void Isend( const T &object, int target, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD ) {
    std::stringstream mystringstream(std::stringstream::out|std::stringstream::binary);
    boost::archive::binary_oarchive oarchive( mystringstream );
    oarchive << object;
    string serializedstring = mystringstream.str();
    MPI_Request request;
    MPI_Isend( (void *)serializedstring.c_str(), serializedstring.length(), MPI_CHAR, target, tag, comm, &request );   
}

// note to self: this is sub-optimal, because it sends two messages, but we can think about that later
template<typename T>
void Bcast_send( const T &object, MPI_Comm comm = MPI_COMM_WORLD ) {
   int rank;
   MPI_Comm_rank(comm, &rank); // note to self: maybe cache this value
   std::stringstream mystringstream(std::stringstream::out|std::stringstream::binary);
   boost::archive::binary_oarchive oarchive( mystringstream );
   oarchive << object;
   string serializedstring = mystringstream.str();
   int len = serializedstring.length();
   MPI_Bcast( &len, 1, MPI_INT, rank, comm );   
   MPI_Bcast( (void *)serializedstring.c_str(), len, MPI_CHAR, rank, comm );   
}

// assumes use with Bcast_send above
template<typename T>
void Bcast_recv( T &receivedObject, int source, MPI_Comm comm = MPI_COMM_WORLD ) {
   int len = 0;
   MPI_Bcast( &len, 1, MPI_INT, source, comm );   
   cout << "message waiting, length " << len << endl;
   char *buffer = new char[len];
   MPI_Bcast( (void *)buffer, len, MPI_CHAR, source, comm );   
   InputMemoryStream inputMemoryStream(buffer,len);
   boost::archive::binary_iarchive iarchive( inputMemoryStream );
   iarchive >> receivedObject;
   delete[] buffer;
}

void Bcast_send_array( const double *array, int len, MPI_Comm comm = MPI_COMM_WORLD ) {
   int rank;
   MPI_Comm_rank(comm, &rank); // note to self: maybe cache this value
   MPI_Bcast( (void *)array, len, MPI_DOUBLE, rank, comm );      
}

void Bcast_recv_array( double *array, int len, int target = 0, MPI_Comm comm = MPI_COMM_WORLD ) {
   MPI_Bcast( array, len, MPI_DOUBLE, target, comm );      
}

template<typename T>
void Recv( T &receivedObject, int source, int tag = MPI_ANY_TAG, MPI_Comm comm = MPI_COMM_WORLD ) {
   MPI_Status status;
   int len = 0;
   MPI_Probe(source, tag, comm, &status);
   MPI_Get_count(&status, MPI_CHAR, &len);
   if(len == MPI_UNDEFINED) {
      cout << "undefined length received" << endl;
      exit(1);
   }
   char *buffer = new char[len];
   MPI_Recv(buffer, len, MPI_CHAR, source, tag, comm, &status );
   InputMemoryStream inputMemoryStream(buffer,len);
   boost::archive::binary_iarchive iarchive( inputMemoryStream );
   iarchive >> receivedObject;
   delete[] buffer;
}

template<>
void Isend( const int &value, int target, int tag, MPI_Comm comm ) {
    MPI_Request request;
    int valuebuffer = value;
    MPI_Isend( (void *)&valuebuffer, 1, MPI_INT, target, tag, comm, &request );   
}

template<>
void Recv( int &receivedInt, int source, int tag, MPI_Comm comm ) {
   MPI_Status status;
   MPI_Recv((void *)&receivedInt, 1, MPI_INT, source, tag, comm, &status );
}

template<typename T>
void Reduce_send( const T &object, int target = 0, MPI_Op op = MPI_SUM, MPI_Comm comm = MPI_COMM_WORLD ) {
   throw MyException("unimplemented");
}

template<typename T>
void Reduce_recv( T &object, MPI_Op op = MPI_SUM, MPI_Comm comm = MPI_COMM_WORLD ) {
   throw MyException("unimplemented");
}

template<>
void Reduce_send( const vector<double> &doublevector, int target, MPI_Op op, MPI_Comm comm ) {
   // change into an array I guess :-(
   int len = doublevector.size();
   double *doublearray = new double[len];
   for( int i = 0; i < len; i++ ) {
      doublearray[i] = doublevector[i];
   }
   MPI_Reduce( doublearray, 0, len, MPI_DOUBLE, op, target, comm );
   delete[] doublearray;
}

template<>
void Reduce_recv( vector<double> &doublevector, MPI_Op op, MPI_Comm comm ) {
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

void Reduce_send_array( const double *doublearray, int len, int target = 0, MPI_Op op = MPI_SUM, MPI_Comm comm = MPI_COMM_WORLD ) {
   MPI_Reduce( (void *)doublearray, 0, len, MPI_DOUBLE, op, target, comm );
}

void Reduce_recv_array( double *doublearray, int len, MPI_Op op = MPI_SUM, MPI_Comm comm = MPI_COMM_WORLD ) {
   int rank;
   MPI_Comm_rank(comm, &rank );
   double *receive = new double[len];
   MPI_Reduce( (void *)doublearray, (void *)receive, len, MPI_DOUBLE, op, rank, comm );
   for( int i = 0; i < len; i++ ) {
      doublearray[i] = receive[i];
   }
   delete []receive;
}

template<typename T>
ostream &operator<<( ostream &os, const vector<T> &vec ) {
   cout << "vector[ ";
   for( int i = 0; i < vec.size(); i++ ) {
      cout << vec[i] << " ";
   }
   cout << "]";
}

int main(int argc, char *argv[] ) {
   MemoryChecker memoryChecker;

   int rank, numprocs;
   MPI_Init(&argc, &argv );
   MPI_Comm_rank(MPI_COMM_WORLD, &rank );
   MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
   if( rank == 0 ) {
       SomeRequest someRequest;
       someRequest.matrix.resize(100);
       someRequest.matrix[15] = '\n';
       someRequest.matrix[17] = '\r';
       someRequest.matrix[20] = '\r';
       someRequest.matrix[21] = '\n';
       someRequest.matrix[22] = '\r';
       someRequest.matrix[39] = 123;
       //for( int i = 1; i < numprocs; i++ ) {
       //   Isend( someRequest, i );
       //}
       cout << "broadcasting... " << endl;
       Bcast_send(someRequest );
       cout << "after broadcast" << endl;
       for( int i = 1; i < numprocs; i++ ) {
          int value;
          Recv(value, i );
          cout << "received value " << value << endl;
       }
       vector<double> myvec;
       myvec.push_back(1);
       myvec.push_back(2);
       myvec.push_back(3);
       myvec.push_back(rank);
       Reduce_recv(myvec);
       cout << myvec << endl;
       double myarray[] = {5,9,2};
       Reduce_recv_array( myarray, 3 );
       cout << myarray[0] << " " << myarray[1] << " " << myarray[2] << endl;

       myarray[0] = 15;
       myarray[1] = 19;
       myarray[2] = 8;
       Bcast_send_array( myarray, 3 );
   } else {
       SomeRequest someRequest;
       //Recv( someRequest, 0 );       
       Bcast_recv( someRequest, 0 );       
       cout << "child received v0.2 " << someRequest.matrix[39] << " " << (int)someRequest.matrix[15] << endl;
       for( int i =15; i <= 22; i++ ) {
           cout << "matrix[" << i << "] = " << (int)someRequest.matrix[i] << endl;
       }
       Isend( 57, 0 );
       vector<double> myvec;
       myvec.push_back(1);
       myvec.push_back(2);
       myvec.push_back(3);
       myvec.push_back(rank);
       Reduce_send(myvec, 0 );
       double myarray[] = {3,5,8};
       Reduce_send_array( myarray, 3, 0 );

       Bcast_recv_array(myarray, 3, 0 );
       cout << myarray[0] << " " << myarray[1] << " " << myarray[2] << endl;
   }
   MPI_Finalize();
   return 0;
}


