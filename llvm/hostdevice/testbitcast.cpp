#include <iostream>
using namespace std;

union {
    float floatvalue;
    int intvalue;
} floatintcast;

int main(int argc, char *argv[]) {
    int myint = 123;
    float myfloat = (float)myint;
    cout << "myfloat " << myfloat << endl;

    float myf2 = *(float *)&myint;
    cout << "myf2 " << myf2 << endl;

    int myint2 = *(int *)&myf2;
    cout << "myint2 " << myint2 << endl;

    return 0;
}
