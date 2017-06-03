#include <stdio.h>
#include <stdlib.h>

#include "jnitest_c.h"

void myfunc() {
   printf("you called myfunc!\n" );
}

void myfunc_name(const char *name) {
   printf("%s called myfunc!\n", name );
}

const char * myfunc_name_return(const char *name) {
   printf("%s called myfunc!\n", name );
   return "This is some message";
}

int *localarray;

void pass_array(int*intarray, int size) {
   int i;
   for( i = 0; i < size; i++ ) {
      printf("%i ", intarray[i]);
   }
   printf("\n");
}

void pass_array2(int*intarray, int size) {
   int i;
   for( i = 0; i < size; i++ ) {
      printf("%i ", intarray[i]);
   }
   printf("\n");
   localarray = intarray;
}

void print_array(int size){
   int i;
   for( i = 0; i < size; i++ ) {
      printf("%i ", localarray[i]);
   }
   printf("\n");
}

int *receive_array() {
   int i;
   int *myarray = (int *)malloc(100*1024*1024 * 4);
   for( i = 0; i < 10; i++ ) {
      myarray[i] = 7;
   }
   myarray[0] = 23;
   return myarray;
}

//void printSomeStructByValue( SomeStruct someStruct ) {
//   printStruct(s
//   cout << "someStruct: " << someStruct.aLong << " " << someStruct.aString << " " << someStruct.aDouble << endl;   
//}

void printSomeStruct( SomeStruct *someStruct ) {
   printf( "someStruct: %ld %s %lf\n", someStruct->aLong, someStruct->aString, someStruct->aDouble );
}

void printStructArray( SomeStruct **someStruct, int arraySize ) {
   int i;
   for( i = 0; i < arraySize; i++ ) {
      printSomeStruct(someStruct[i]);
   }
}

void print_pointer_to_int_array(long **ppint, long size) {
   int i;
   for( i = 0; i < size; i++ ) {
      printf("%ld\n",(*ppint)[i]);
   }
}

void printStructWithArray(StructWithArray *structWithArray ) {
   int i;
   printf("%d\n", structWithArray->size );
   for( i = 0; i < structWithArray->size; i++ ) {
      printf("%s\n", structWithArray->array[i].aString );
   }
}

void printLearnParm(LEARN_PARM *learn_parm ) {
printf("svm c %lf kerneltype %ld \n", 
   learn_parm->svm_c, learn_parm->type
);
printf("totwords %ld\n", learn_parm->totwords );
//printf("svm c %lf cache size %ld kerneltype %ld\n", learn_parm->svm_c, learn_parm->kernel_cache_size, learn_parm->type
//);
}

void printDoc(DOC *doc) {
   printf("doc docnum %ld\n", doc->docnum );
   printf("doc slackid %ld\n", doc->slackid );
   printf("doc queryid %ld\n", doc->queryid );
   printf("doc costfactor %lf\n", doc->costfactor );
   if( doc->fvec != 0 ) {
      printSvector(doc->fvec );
   }
}

void printDocArray(DOC **docs, long numDocs) {
   int i;
   for( i = 0; i < numDocs; i++  ) {
      printDoc(docs[i] );
   }
}

void printWord(WORD *word) {
printf("wnum %ld weight %lf\n", word->wnum, word->weight );
}

void printSvector(SVECTOR *vector) {
   int i;
   i = 0;
   while( vector->words[i].wnum != 0 ) {
      printf("wnum %ld weight %lf\n", vector->words[i].wnum, vector->words[i].weight );
       i++;
   }
}


