#include <iostream>

void sayName( std::string name );
float addFloats( float a, float b );
void addToArray( int N, float *array, float amount );
void addToUCArray( int N, unsigned char *array, unsigned char amount );
void addToIntArray( int N, int *array, int amount );

class MyClass {
public:
    MyClass();
    MyClass(int intarg);
    std::string warpName( std::string inName );   
};

