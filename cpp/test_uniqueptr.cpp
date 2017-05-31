#include <iostream>
#include <memory>
#include <vector>
using namespace std;


class MyClass {
public:
    MyClass() {
        cout << "MyClass()" << endl;
    }
    ~MyClass() {
        cout << "~MyClass()" << endl;
    }
};

class MyNamedClass {
public:
    string name;
    MyNamedClass(string name) : name(name) {
        cout << "MyNamedClass() " << name << endl;
    }
    ~MyNamedClass() {
        cout << "~MyNamedClass() " << name << endl;
    }
};

void foo(MyClass *p1) {
    cout << "foo >>>" << endl;
    cout << "foo <<<" << endl;
}

unique_ptr<MyClass>createMyClass() {
    unique_ptr<MyClass> c1;
    return c1;
}

void test1() {
    unique_ptr<MyClass> myClass1(new MyClass);
    cout << "before calling foo" << endl;
    // foo(move(myClass1));
    foo(myClass1.get());
    cout << "after calling foo" << endl;
    cout << "before call createmyclass" << endl;
    unique_ptr<MyClass> created = createMyClass();
    cout << "after call createmyclass" << endl;    
}

void test2() {
    vector<unique_ptr<MyNamedClass> > myclasses;
    myclasses.push_back(unique_ptr<MyNamedClass>(new MyNamedClass("foo")));
    myclasses.push_back(unique_ptr<MyNamedClass>(new MyNamedClass("bar")));
}

int main(int argc, char *argv[]) {
    // test1();
    test2();
    return 0;
}
