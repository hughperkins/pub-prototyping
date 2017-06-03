#include "jnitest.h"

int main(){
   myfunc();
   SomeStruct *psomeStruct = new SomeStruct();
   psomeStruct->aLong = 123456;
   psomeStruct->aString = "hello!";
   psomeStruct->aDouble = 1.234;
   printSomeStruct(psomeStruct);
   SomeStruct someStruct;
   someStruct.aLong = 153456;
   someStruct.aString = "foo!";
   someStruct.aDouble = 4.234;
   printSomeStructByValue(someStruct);
   SomeStruct **sarray = new SomeStruct*[1];
   sarray[0] = psomeStruct;
   printStructArray(sarray,1);

   StructWithArray *pstructWithArray = new StructWithArray();
   pstructWithArray->size = 2;
   pstructWithArray->array = new SomeStruct[2];
   pstructWithArray->array[0].aString = "Hey there!";
   pstructWithArray->array[1].aString = "Cool world!";
   printStructWithArray(pstructWithArray);
   
   delete psomeStruct;
   delete[] sarray;

   return 0;
}

// sourcecode

