#include <iostream>
#include <vector>
using namespace std;

class Foo {
public:
   string name;
   Foo(string name ) {
       cout << "Foo(" << name << ")" << endl;
       this->name = name;
   }
};

ostream &operator<<( ostream &os, const Foo &foo ) {
   os << "Foo( " << foo.name << " )";
   return os;
}

int main(int argc, char *argv[] ) {
   vector<Foo> foos;
   foos.push_back( Foo("hello") );
   cout << foos[0] << endl;
   return 0;
}

