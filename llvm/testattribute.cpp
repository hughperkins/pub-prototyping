#include <iostream>
using namespace std;

// __attribute__((myatt))
#pragma my pragma
void foo() {
    cout << "foo()" << endl;
}

int main(int argc, char *argv[]) {
    foo();
    return 0;
}
