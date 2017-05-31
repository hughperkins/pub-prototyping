#include <iostream>

class Foo {
public:

};

int main(int argc, char *argv[]) {
    typedef Foo *foo;
    foo a = 0;
    foo b = -1;
    std::cout << "done" << std::endl;
}
