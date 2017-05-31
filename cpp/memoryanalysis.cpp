#include <cstdlib>
using namespace std;

// override operators new and delete, so we can check for memory leaks

int memoryallocated = 0;

const int maxallocations = 1000;
int memoryaddresses[maxallocations + 1];
int sizes[maxallocations + 1];
bool initializedmemory = false;
int numallocates = 0;

void *operator new( size_t size ) {
   if( !initializedmemory ) {
      //cout << "initializing memory " << endl;
      initializedmemory = true;
      for( int i = 0; i < maxallocations; i++ ) {
          memoryaddresses[i] = 0;
          sizes[i] = 0;
      }
   }
   //cout << "operator new( " << size << ")" << endl;
   memoryallocated += size;
   numallocates++;
   void *p_mem = malloc(size);
   int i = 0;
   for( i = 0; i < maxallocations; i++ ) {
      if( memoryaddresses[i] == 0 ) {
         memoryaddresses[i] = (int)p_mem;
         sizes[i] = size;
         break;
      }
   }
   if( i == maxallocations ) {
      cout << "error: no space for more memory allocations" << endl;
      abort();
   }
   //sizebyaddress[(int)p_mem] = size;
   return p_mem;
}

void operator delete( void *p ) {
   int size = 0;
   for( int i = 0; i < maxallocations; i++ ) {
      if( memoryaddresses[i] == (int)p ) {
         size = sizes[i];
         memoryaddresses[i] = 0;
         sizes[i] = 0;
         break;
      }
   }
   //cout << "operator delete()" << endl;
   memoryallocated -= size;
   free( p );
}

class MemoryChecker {
public:
   MemoryChecker();
   ~MemoryChecker();
   int memory;
   int allocates;
};

MemoryChecker::MemoryChecker() {
   this->memory = memoryallocated;
   this->allocates = numallocates;
   //cout << "memorychecker" << endl;
}

MemoryChecker::~MemoryChecker() {
   //cout << "~memorychecker" << endl;
   int memoryused = memoryallocated - memory;
   int allocatesdelta = numallocates - allocates;
   if( memoryused == 0 ) {
      //cout << "Memory check passes.  Total allocates: " << allocatesdelta << endl;
   } else {
      cout << "ERROR: memory leaked: " << memoryused << " allocates: " << allocatesdelta << endl;
   }
}


