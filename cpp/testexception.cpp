#include <stdexcept>
#include <iostream>

#include <cstdio>

using namespace std;

class MyBase {
public:
};

class MyClass : public MyBase {
public:
    MyClass(string name) : name(name) {}
    string name;
};

class MyException : public std::exception {
public:
    MyException(MyClass *myclass) {
        this->myclass = myclass;
    }
    virtual const char* what() const throw()
    {
        cout << "what is name " << myclass->name << endl;
        static char buf[1024];
        sprintf(buf, "%s", (string("my custom exception ") + myclass->name).c_str());
        return buf;
    }
    MyClass *myclass;
};

void foo() {
    throw std::runtime_error("some error");
}

void throwcustom() {
    MyClass *myclass = new MyClass("foobar");
    throw MyException(myclass);
}

int main(int argc, char *argv[]) {
    try {
        foo();
    } catch(std::runtime_error& e ){
        std::cout << "caught runtime_error" << std::endl;
    }
    try {
        throwcustom();
    } catch(MyException& e ){
        std::cout << "caught MyException" << std::endl;
        cout << e.what() << endl;
    }
    return 0;
}
