extern "C" {
   struct SomeStruct {
       long aLong;
       const char *aString;
       double aDouble;
   };
   struct StructWithArray {
      SomeStruct *array;
      int size;
   };

   void myfunc();
   void myfunc_name(const char *name);
   const char * myfunc_name_return(const char *name);
   void pass_array(int *intarray, int size);
   void pass_array2(int *intarray, int size);
   int *receive_array();
   void print_array(int size);
   void printSomeStructByValue( SomeStruct someStruct );
   void printSomeStruct( SomeStruct *someStruct );
   void printStructArray( SomeStruct **someStruct, int arraySize );
   void print_pointer_to_int_array(long **ppint, long size);
   void printStructWithArray(StructWithArray *structWithArray );
}

