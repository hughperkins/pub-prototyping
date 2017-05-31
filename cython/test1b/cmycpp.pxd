from libcpp.string cimport string

cdef extern from "mycpp.h":
    void sayName( string name )
    float addFloats( float a, float b )
    void addToArray( int N, float *array, float amount )
    void addToUCArray( int N, unsigned char *array, unsigned char amount )
    void addToIntArray( int N, int *array, int amount )
    void throwException( string message ) except +
    cdef cppclass MyClass:
        MyClass()
        MyClass(int intarg)
        string warpName( string inName )
        float getFloat( int *raiseError, string *message )
        void longOp( int secs ) nogil
    #cdef cppclass MyInterfaceAdapter:
    #    int getNumber()
    ctypedef int (*funcdef)( int someint, void *pyCallback)
    void testCallback( funcdef cCallback, int value, void *pyCallback )
    void cySleepSecs( int secs ) nogil
    #float fnRaiseException( float invalue, bint doRaise )
    float cyFnRaiseException( float invalue, bint doRaise )
    void checkException( int *wasRaised, string *message )

    ctypedef int(*CyMyInterface_getNumberDef)(void *pyObject)
    ctypedef void(*CyMyInterface_getFloatsDef)(float *floats, void *pyObject)
    cdef cppclass CyMyInterface:
        CyMyInterface(void *pyObject)
        void setGetNumberCallback( CyMyInterface_getNumberDef cGetNumber )
        void setGetFloatsCallback( CyMyInterface_getFloatsDef cGetFloats )
        #int getNumber()
        #void getFloat(float *floats)
    void callMyInterface( CyMyInterface *instance )

