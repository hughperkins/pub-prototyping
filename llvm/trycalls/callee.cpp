#include <iostream>

using namespace std;

#include "callee.h"

void printInt(int someInt) {
    cout << "you said " << someInt << endl;
}

void printInt2(int someInt) {
    cout << "2 you said " << someInt << endl;
}

void printFloat(float val) {
    cout << "you said " << val << endl;
}

void printChars(const char *chars) {
    cout << "you said " << chars << endl;
}

void printChars2(const char *chars) {
    cout << "2 you said " << chars << endl;
}
