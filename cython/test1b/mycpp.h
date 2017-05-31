#include <iostream>

//struct _object;
//typedef _object PyObject;

//#include "Python.h"

//#include "pymycpp_api.h"

void sayName( std::string name );
float addFloats( float a, float b );
void addToArray( int N, float *array, float amount );
void addToUCArray( int N, unsigned char *array, unsigned char amount );
void addToIntArray( int N, int *array, int amount );
void throwException( std::string message );

extern int exceptionRaised;
extern std::string exceptionMessage;
void raiseException( std::string message );
void checkException( int *wasRaised, std::string *message );

class MyClass {
public:
    MyClass();
    MyClass(int intarg);
    std::string warpName( std::string inName );   
    float getFloat( int *raiseException, std::string *message );
    void longOp( int secs );
};

class MyInterface {
public:
    virtual int getNumber() = 0;
    virtual void getFloats( float *floats ) = 0;
};

//class CyMyInterface : public MyInterface {
//public:
//    PyObject *pyObject;
//    CyMyInterface( PyObject *pyObject ) :
//            pyObject(pyObject) {
//        Py_XINCREF( pyObject );
//        std::cout << "CyMyinterfaceAdapter(pyObject)" << std::endl;
//    }
//    virtual ~CyMyInterface() {
//        std::cout << "~MyinterfaceAdapter()" << std::endl;
//        Py_XDECREF( pyObject );
//    }
//    virtual int getNumber() { 
//        return cy_MyInterface_getNumber( pyObject ); 
//    }
//};


class CyMyInterface : public MyInterface {
public:
    void *pyObject;

    typedef int(*getNumberDef)(void *pyObject);
    typedef void(*getFloatsDef)(float *floats, void *pyObject);

    getNumberDef cGetNumber;
    getFloatsDef cGetFloats;

    CyMyInterface(void *pyObject) :
            pyObject(pyObject) {
        std::cout << "CyMyinterfaceAdapter(pyObject)" << std::endl;
        cGetNumber = 0;
    }
    void setGetNumberCallback( getNumberDef cGetNumber ) {
        this->cGetNumber = cGetNumber;
    }
    void setGetFloatsCallback( getFloatsDef cGetFloats ) {
        this->cGetFloats = cGetFloats;
    }
    virtual ~CyMyInterface() {
        std::cout << "~CyMyinterfaceAdapter()" << std::endl;
    }
    virtual int getNumber() { 
        int result = cGetNumber( pyObject ); 
        std::cout << "CyMyInterface.getNumber, result received: " << result << std::endl;
        return result;
    }
    virtual void getFloats( float *floats ) {
        cGetFloats( floats, pyObject );
    }
};

void callMyInterface( MyInterface *myinstance );

typedef int (*funcdef)( int someint, void *pycallback);
void testCallback( funcdef ccallback, int value, void *pycallback );

void sleepSecs( int secs );
void cySleepSecs( int secs );
//float fnRaiseException( float in, int doRaise, int *raiseException, std::string *message );

float fnRaiseException( float in, int doRaise );
float cyFnRaiseException( float in, int doRaise );

