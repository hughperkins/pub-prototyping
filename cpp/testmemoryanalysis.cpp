#include <iostream>
#include <typeinfo>
using namespace std;

class Foo{
public:
   virtual ~Foo(){
   }
};

int main(int argc, char *argv[] ) {
   Foo foo;
   void *ptr = &foo;
   cout << typeid(ptr).name() << endl;
   return 0;
}

