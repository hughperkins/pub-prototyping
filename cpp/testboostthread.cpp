#include <cstdlib>
#include <signal.h>
#include <pthread.h>
#include <unistd.h>
#include <iostream>
#include <boost/thread.hpp>
using namespace std;

class Foo {
public:
   void go(string message) {
      cout << "start " << message << endl;
      system("sleep 10");
      cout << "end " << message << endl;
   }
};

int main() {
   Foo foo;
   boost::thread mythread(&Foo::go, &foo, "thread1" );
   mythread.interrupt();
   cout << "sleep a bit" << endl;
   usleep(5 * 1000000);
   cout << "done" << endl;
   return 0;
}

