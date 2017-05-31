#include <tr1/memory>
#include <iostream>
#include <vector>
using namespace std;
using namespace tr1;

#include "utils/memoryanalysis.cpp"

#define safenew0(T ) shared_ptr<T>( new T() )
#define safenew1(T, arg1 ) shared_ptr<T>( new T( arg1 ) )
#define safenew2(T, arg1, a2 ) shared_ptr<T>( new T( arg1, a2 ) )

#define sp(T) shared_ptr<T>

class Foo {
public:
   string name;
   Foo(string name ) { this->name = name; cout << "Foo(" << name << ")" << endl; }
   Foo(string name, int num ) { this->name = name; cout << "Foo(" << name << ", " << num << ")" << endl; }
   ~Foo() { cout << "~Foo()" << endl; }
};

vector< shared_ptr< Foo> > *foos;

void registerFoo( sp(Foo) foo ) {
   foos->push_back(foo );
}

int main(){
   MemoryChecker memoryChecker;
   shared_ptr<Foo> a( new Foo("hello" ) );
   shared_ptr<Foo> b = a;
   cout << b->name << endl;
   cout << a->name << endl;

   foos = new vector<shared_ptr<Foo > >;
   registerFoo( shared_ptr<Foo>( new Foo("blah" ) ) );
   registerFoo( shared_ptr<Foo>( new Foo("bar" ) ) );
   registerFoo( shared_ptr<Foo>( new Foo("foo" ) ) );
   registerFoo( shared_ptr<Foo>( new Foo("foo" ) ) );
   registerFoo( safenew1( Foo, "foobar" ) );
   registerFoo( safenew2( Foo, "foobar", 20 ) );
   delete foos;

   return 0;
}

