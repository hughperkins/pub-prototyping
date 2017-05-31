#include <iostream>
#include <stdexcept>
#include <thread>
#include <chrono>
using namespace std;

#include "mycpp.h"

void sayName( string name ) {
    cout << "your name is: " << name << endl;
}

void throwException( std::string message ) {
    throw runtime_error( "message was: " + message );
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

void MyClass::longOp( int secs ) {
//    std::this_thread::sleep_for (std::chrono::seconds(secs));
// bizarrely, sleep is interruptible, so use infinite loop instead
    double sum = 3.1;
    if( secs == 1 ) {
        throw runtime_error("you chose the magic number!");
    }
    for( int i = 1; i < 1000000000; i++ ) {
         sum *= i % 20;
    }
    cout << sum << endl;
}

float MyClass::getFloat( int *raiseException, string *message ) {
    *raiseException = 1;
    *message = "hey there!";
    return 0;
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

void callMyInterface( MyInterface *myInstance ) {
    cout << "instance returned: " << myInstance->getNumber() << endl;
    float vars[10];
    myInstance->getFloats(vars);
    for( int i = 0; i < 10; i++ ) {
        cout << "vars[" << i << "]=" << vars[i] << " ";
    }
    cout << endl;
}


void testCallback( funcdef ccallback, int value, void *pycallback ) {
    int returnvalue = ccallback( value, pycallback );
    cout << "mycpp.testCallback, received from python: " << returnvalue << endl;
}

void sleepSecs( int secs ) {
//    std::this_thread::sleep_for (std::chrono::seconds(secs));
// bizarrely, sleep is interruptible, so use infinite loop instead
    double sum = 3.1;
    if( secs == 1 ) {
        throw runtime_error("you chose the magic number!");
    }
    for( int i = 1; i < 1000000000; i++ ) {
         sum *= i % 20;
    }
    cout << sum << endl;
}

void cySleepSecs( int secs ) {
    try {
        sleepSecs( secs );
    } catch( runtime_error &e ) {
        raiseException( e.what() );
    }
}

void raiseException( std::string message ) {
    exceptionRaised = 1; 
    exceptionMessage = message;    
}

//float fnRaiseException( float in, int doRaise, int *raiseException, string *message ) {
float fnRaiseException( float in, int doRaise ) {
    if( doRaise ) {
        throw runtime_error("hey there!");
    } else {
        return in + 3.5f;
    }
}

float cyFnRaiseException( float in, int doRaise ) {
    try {
        return fnRaiseException( in, doRaise );
    } catch( runtime_error &e ) {
        raiseException( e.what() );
        return 0;
    }
}

void checkException( int *wasRaised, std::string *message ) {
    *wasRaised = exceptionRaised;
    *message = exceptionMessage;
}

int exceptionRaised = 0;
std::string exceptionMessage = "";

