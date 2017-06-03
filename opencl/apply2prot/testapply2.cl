#define MAX_CLTORCH_DIMS 25

typedef struct TensorInfoCl {
  unsigned int sizes[MAX_CLTORCH_DIMS]; // note: this is redundant between a/b
  unsigned int strides[MAX_CLTORCH_DIMS];
  int offset;
  int dims; //redundant
} TensorInfoCl;


kernel void test(int totalElements, global TensorInfoCl *info, global float *a_data, global float *b_data) {
  int dims = info->dims;
  // just do an apply1 I guess :-)
  int offset = info->offset;
  int linearId = get_global_id(0);
  if( linearId >= totalElements ) {
    return;
  }
//  if( linearId > 0 ) {
//    return;
//  }
  for( int d = dims - 1; d >= 0; d-- ) {
    int thisSize = info->sizes[d];
    int thisCoord = linearId % thisSize;
    offset += thisCoord * info->strides[d];
    linearId /= thisSize;
  }
//  if( linearId == 0 ) {
//    data[0] = 123;
//    data[1] = offset;
//    data[2] = dims;
//  }
  a_data[offset] = a_data[offset] + b_data[offset];
//  data[offset] = 4.4f;
}

