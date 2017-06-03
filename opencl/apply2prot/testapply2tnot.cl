#define MAX_CLTORCH_DIMS 25

typedef struct TensorInfoCl {
  unsigned int sizes[MAX_CLTORCH_DIMS]; // note: this is redundant between a/b
  unsigned int strides[MAX_CLTORCH_DIMS];
  int offset;
  int dims; //redundant
} TensorInfoCl;

void op( global float *out, global float *in1 ) {
  *out += *in1;
}

kernel void test(int totalElements, global TensorInfoCl *a_info, global float *a_data, global TensorInfoCl *b_info, global float *b_data) {
  int dims = a_info->dims;
  // just do an apply1 I guess :-)
  int a_offset = a_info->offset;
  int b_offset = b_info->offset;
  int linearId = get_global_id(0);
  if( linearId >= totalElements ) {
    return;
  }
//  if( linearId > 0 ) {
//    return;
//  }
  for( int d = dims - 1; d >= 0; d-- ) {
    int thisSize = a_info->sizes[d];
    int thisCoord = linearId % thisSize;
    a_offset += thisCoord * a_info->strides[d];
    b_offset += thisCoord * b_info->strides[d];
    linearId /= thisSize;
  }
//  if( linearId == 0 ) {
//    data[0] = 123;
//    data[1] = offset;
//    data[2] = dims;
//  }
//  a_data[a_offset] = a_data[a_offset] + b_data[b_offset];
  op( a_data + a_offset, b_data + b_offset );
//  data[offset] = 4.4f;
}

