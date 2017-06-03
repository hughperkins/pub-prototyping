#include <stdlib.h>

#include "jnitest_c.h"

int main(){
   myfunc();
   SomeStruct *psomeStruct = (SomeStruct *)(malloc(sizeof(SomeStruct)));
   psomeStruct->aLong = 123456;
   psomeStruct->aString = "hello!";
   psomeStruct->aDouble = 1.234;
   printSomeStruct(psomeStruct);
   //SomeStruct someStruct;
   //someStruct.aLong = 153456;
   //someStruct.aString = "foo!";
   //someStruct.aDouble = 4.234;
   //printSomeStructByValue(someStruct);
   SomeStruct **sarray = (SomeStruct **)malloc(8 * 2);
   sarray[0] = psomeStruct;
   printStructArray(sarray,1);

   StructWithArray *pstructWithArray = (StructWithArray *)malloc(sizeof(StructWithArray));
   pstructWithArray->size = 2;
   pstructWithArray->array = (SomeStruct *)malloc(2 * sizeof(SomeStruct));
   pstructWithArray->array[0].aString = "Hey there!";
   pstructWithArray->array[1].aString = "Cool world!";
   printStructWithArray(pstructWithArray);
   
   return 0;
}

// sourcecode

