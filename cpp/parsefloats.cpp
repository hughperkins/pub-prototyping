#include <iostream>
#include <stdexcept>
#include <fstream>
#include <cstdlib>
#include <cstring>
using namespace std;

#include "utils/memoryanalysis.cpp"
#include "utils/timerelapsed.h"

double *w;
    
inline int eatInt( char const**pptr, char delimiter ) {
   const char *pthischar = *pptr;
   while( *pthischar != delimiter && *pthischar != 0 ) {
      pthischar++;
   }
   if( pthischar == *pptr ) {
      throw std::runtime_error( "couldn't read integer" );
   }
   int value = atoi(*pptr);
   *pptr = pthischar;
   return value;
}

inline double eatDouble( const char **pptr, char delimiter ) {
   const char *pthischar = *pptr;
//    int n = 0;
   while( *pthischar != delimiter && *pthischar != 0 ) {
//    cout << "thischar " << *pthischar << " delim " << delimiter << " n " << n << endl;
      pthischar++;
//    if( n > 10 ) {
//    exit(-1);
//    }
//    n++;
   }
   double value = atof(*pptr);
//    cout << "value: " << value << " newptr " << pthischar << endl;
   *pptr = pthischar;
   return value;
}

inline void eatWhitespace( const char **pptr ) {
   const char *thischar = *pptr;
   while( *thischar == ' ' && *thischar != 0 ) {
      thischar++;
   }
   *pptr = thischar;
}

inline void eat( const char **pptr, char character ) {
   if( **pptr != character ) {
      throw std::runtime_error( "unexpected character [" + std::string(*pptr) + "]" );
   }
   (*pptr)++;
}

inline void eatIgnoreEnd( const char **pptr, char character ) {
    if( **pptr == 0 ) {
        return;
    }
   if( **pptr != character ) {
      throw std::runtime_error( "unexpected character [" + std::string(*pptr) + "]" );
   }
   (*pptr)++;
}

string getFileContents( string filename ) {
    char * buffer = 0;
    long length;
    FILE * f = fopen (filename.c_str(), "rb");

    string returnstring = "";
    if (f)
    {
      fseek (f, 0, SEEK_END);
      length = ftell (f);
      fseek (f, 0, SEEK_SET);
      buffer = new char[length+1];
      if (buffer)
      {
        int result = fread (buffer, 1, length, f);
      }
      fclose (f);
        buffer[length] = 0;
      returnstring = buffer;
      delete[] buffer;
    }
    return returnstring;
}

void readModelFstream( string filename, int K, int numVectors ) {
   ifstream myifstream(filename.c_str() );
   for( int k = 0; k < K; k++ ) {
      for( int m = 0 ; m < numVectors; m++ ) {
         myifstream >> w[m*K + k];
      }
   }
   myifstream.close();
}

void readModelFileEat( string filename, int K, int numVectors ) {
    TimerElapsed timer("readModelFileEat");
    string filecontents = getFileContents( filename );
    timer.timeCheck("read contents");
    const char *contents = filecontents.c_str();
    timer.timeCheck("as c_str()");
    const char *ptr = contents;
    float thisvalue = 0;
   for( int k = 0; k < K; k++ ) {
//    cout << "k " << k << endl;
      for( int m = 0 ; m < numVectors; m++ ) {
        if( m == numVectors - 1 ) {
         thisvalue = eatDouble(&ptr,'\n');
        eatIgnoreEnd(&ptr, '\n');
        } else {
         thisvalue = eatDouble(&ptr,' ');
        eatIgnoreEnd(&ptr, ' ');
        }
        w[m*K+k] = thisvalue;
      }
   }
    timer.timeCheck("parsed");

}

void dumpW(){  
    for( int i = 0; i < 10; i++ ) {
        cout << w[i] << " ";
    }
    cout << " ...";
    for( int i = 100000; i < 100010; i++ ) {
        cout << w[i] << " ";
    }
    cout << endl;
}

void clear( int K, double *w ) {
    for( int k = 0; k < K; k++ ) {
        w[k] = 0;
    }
}

void writeWstreams( int K, double *w, string filename ) {
    ofstream myofstream(filename.c_str());
    for( int k =0; k < K; k++ ) {
        myofstream << w[k] << endl;
    }
    myofstream.close();
}

void writeWfile( int K, double *w, string filename ) {
    FILE *file = fopen(filename.c_str(), "w");
    for( int k = 0; k < K; k++ ) {
        fprintf( file, "%lf\n", w[k] );
    }
    fclose(file);
}

int main( int argc, char *argv[] ) {
    MemoryChecker memoryChecker;

    TimerElapsed timer;

    const int K = 1355191;

//    readModelFstream("/tmp/samples/1.txt", 1355,1);
//    timer.timeCheck("fstream");
//    dumpW();

    w = new double[K];

//    readModelFstream("/tmp/samples/1.txt", K,1);
//    timer.timeCheck("fstream");
//    dumpW();
//    clear(K,w);

//    readModelFileEat("/tmp/samples/1.txt",1355,1);
//    timer.timeCheck("fileeat");
//    dumpW();

//    readModelFileEat("/tmp/samples/1.txt",K,1);
//    timer.timeCheck("fileeat");
//    dumpW();

    for( int k = 0; k < K; k++ ) {
        w[k] = (1.567+k);
    }
    timer.timeCheck("populated w");
    writeWstreams( K, w, "/tmp/foo1.txt" );
    timer.timeCheck("wrote w streams");

    writeWfile( K, w, "/tmp/foo2.txt" );
    timer.timeCheck("wrote w file");

    return 0;
}


