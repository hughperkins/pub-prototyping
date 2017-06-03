#define MAX_CLTORCH_DIMS 4

//typedef struct TensorInfoCl {
//  unsigned int sizes[MAX_CLTORCH_DIMS]; // note: this is redundant between a/b
//  unsigned int strides[MAX_CLTORCH_DIMS];
//  int offset;
//  int dims; //redundant
//} TensorInfoCl;

constant int dims = 4;
constant int sizes[4] = {50,400,5,6};
constant int strides[4] = {12000,30,6,1};

kernel void test(int totalElements, global float *data) {
  // just do an apply1 I guess :-)
  int offset = 0;
  int linearId = get_global_id(0);
  if( linearId >= totalElements ) {
    return;
  }
//  if( linearId > 0 ) {
//    return;
//  }
  for( int d = dims - 1; d >= 0; d-- ) {
    int thisSize = sizes[d];
    int thisCoord = linearId % thisSize;
    offset += thisCoord * strides[d];
    linearId /= thisSize;
  }
//  if( linearId == 0 ) {
//    data[0] = 123;
//    data[1] = offset;
//    data[2] = dims;
//  }
  data[offset] = data[offset] + 1;
//  data[offset] = 4.4f;
}

