#include <iostream>
using namespace std;

#include "jnitest.h"

void myfunc() {
   cout << "you called myfunc!" << endl;
}

void myfunc_name(const char *name) {
   cout << name << " called myfunc!" << endl;
}

const char * myfunc_name_return(const char *name) {
   cout << name << " called myfunc!" << endl;
   return "This is some message";
}

int *localarray;

void pass_array(int*intarray, int size) {
   for( int i = 0; i < size; i++ ) {
      cout << intarray[i] << " ";
   }
   cout << endl;
}

void pass_array2(int*intarray, int size) {
   for( int i = 0; i < size; i++ ) {
      cout << intarray[i] << " ";
   }
   cout << endl;
   localarray = intarray;
}

void print_array(int size){
   for( int i = 0; i < size; i++ ) {
      cout << localarray[i] << " ";
   }
   cout << endl;
}

int *receive_array() {
   int *myarray = new int[100*1024*1024];
   for( int i = 0; i < 10; i++ ) {
      myarray[i] = 7;
   }
   myarray[0] = 23;
   return myarray;
}

void printSomeStructByValue( SomeStruct someStruct ) {
   cout << "someStruct: " << someStruct.aLong << " " << someStruct.aString << " " << someStruct.aDouble << endl;   
}

void printSomeStruct( SomeStruct *someStruct ) {
   cout << "someStruct: " << someStruct->aLong << " " << someStruct->aString << " " << someStruct->aDouble << endl;
}

void printStructArray( SomeStruct **someStruct, int arraySize ) {
   for( int i = 0; i < arraySize; i++ ) {
      cout << "someStruct: " << someStruct[i]->aLong << " " << someStruct[i]->aString << " " << someStruct[i]->aDouble << endl;
   }
}

void print_pointer_to_int_array(long **ppint, long size) {
   for( int i = 0; i < size; i++ ) {
      cout << (*ppint)[i] << endl;
   }
}

void printStructWithArray(StructWithArray *structWithArray ) {
   cout << structWithArray->size << endl;
   for( int i = 0; i < structWithArray->size; i++ ) {
      cout << structWithArray->array[i].aString << endl;
   }
}

