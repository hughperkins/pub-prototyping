#include "svm_common.h"

typedef struct SomeStruct {
    long aLong;
    char *aString;
    double aDouble;
} SomeStruct;
typedef struct StructWithArray {
   SomeStruct *array;
   int size;
} StructWithArray;


void myfunc();
void myfunc_name(const char *name);
const char * myfunc_name_return(const char *name);
void pass_array(int *intarray, int size);
void pass_array2(int *intarray, int size);
int *receive_array();
void print_array(int size);
//void printSomeStructByValue( SomeStruct someStruct );
void printSomeStruct( SomeStruct *someStruct );
void printStructArray( SomeStruct **someStruct, int arraySize );
void print_pointer_to_int_array(long **ppint, long size);
void printStructWithArray(StructWithArray *structWithArray );

void printLearnParm(LEARN_PARM *learnParm );
void printDoc(DOC *doc);
void printWord(WORD *word);
void printSvector(SVECTOR *vector);
void printDocArray(DOC **doc, long numDocs);

