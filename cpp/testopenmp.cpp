#include <iostream>
using namespace std;

// this doesn't need to thrash the memory, it simply
// needs two or three floats in cache
// therefore it should be cpu-bound, which is an ok-ish
// test for us
bool isPrime( int n ) {
   int sqrtn = sqrt(n);
   for( int i = 2; i <= sqrtn ) {
      if( n % i == 0 ) {
         return false;
      }
   }
   return true;
}

int main(int argc, char *argv[] ) {
   int N = 100000000;
   bool *isPrime = new bool[N]; // using sieve of aristh.. would be best, but we're trying to test multicore
                                // this array is to prove we can access shared memory across threads
   #pragma omp parallel default(
   

   int count = 0;
   for( int n = 0; n < N; n++ ) {
      // summing them all should be quite quick
      if( isPrime[n] ) {
         count++;
      }
   }
   cout << "numprimes: " << count << endl; // make sure it doesn't just get optimized away..., and check correct
   delete[] isPrime;
   return 0;
}

