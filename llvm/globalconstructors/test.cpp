#include <iostream>
using namespace std;

extern "C" {
    void foo(const char *message);
    void callfoo();
    void callbar();
}

void foo(const char *message) {
    cout << "message: " << message << endl;
}

void callfoo() {
    foo("blah");
}

void callbar() {
    foo("bar");
}

int main(int argc, char *argv[]) {

    return 0;
}
