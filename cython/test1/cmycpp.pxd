from libcpp.string cimport string

cdef extern from "mycpp.h":
    void sayName( string name )
    float addFloats( float a, float b )
    void addToArray( int N, float *array, float amount )
    void addToUCArray( int N, unsigned char *array, unsigned char amount )
    void addToIntArray( int N, int *array, int amount )
    cdef cppclass MyClass:
        MyClass()
        MyClass(int intarg)
        string warpName( string inName )

