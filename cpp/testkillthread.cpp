#include <cstdlib>
#include <signal.h>
#include <pthread.h>
#include <unistd.h>
#include <iostream>
using namespace std;

void *callfunction( void *threadid ) {
   system("sleep 10");
}

int main() {
   pthread_t thread;
   pthread_create(&thread, 0, callfunction, 0 );
//   pthread_kill(thread,9);
   pthread_cancel(thread);
   cout << "sleep a bit" << endl;
   usleep(5 * 1000000);
   cout << "done" << endl;
   return 0;
}

