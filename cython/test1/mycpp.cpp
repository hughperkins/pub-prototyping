#include <iostream>
using namespace std;

#include "mycpp.h"

void sayName( string name ) {
    cout << "your name is: " << name << endl;
}

MyClass::MyClass() {
    cout << "MyClass()" << endl;
}

MyClass::MyClass(int intarg) {
    cout << "MyClass( intarg )" << endl;
}

std::string MyClass::warpName( std::string inName ) {
    cout << "warping " << inName << " ..." << endl;
    return "You said " + inName;
}

float addFloats( float a, float b ) {
    return a + b;
}

void addToArray( int N, float *array, float amount ) {
    cout << "addToArray N=" << N << " amount=" << amount << endl;
    for( int i = 0; i < N; i++ ) {
        cout << "array[" << i << "]=" << array[i] << endl;
        array[i] += amount;
    }
}

void addToUCArray( int N, unsigned char *array, unsigned char amount ) {
    cout << "addToUCArray N=" << N << " amount=" << amount << endl;
    for( int i = 0; i < N; i++ ) {
        cout << "array[" << i << "]=" << (int)array[i] << endl;
        array[i] += amount;
    }
}

void addToIntArray( int N, int *array, int amount ) {
    cout << "addToIntArray N=" << N << " amount=" << amount << endl;
    for( int i = 0; i < N; i++ ) {
        cout << "array[" << i << "]=" << array[i] << endl;
        array[i] += amount;
    }
}

