kernel void fused(
  int totalElements,
  global const float *d17,
  global const float *d26,
  global const float *d28,
  global const float *d29,
  global const float *d20,
  global float *o1,
  global float *o3
) {
  int n = get_global_id(0);
  if(n >= totalElements) {
    return;
  }
  float d16 = sigmoid(d26[n]);
  float d7 = d17[n] * d16;
  float d18 = sigmoid(d28[n]);
  float d19 = tanh(d29[n]);
  float d8 = d18 * d19;
  float d1 = d7 + d8;
  float d10 = tanh(d1);
  float d9 = sigmoid(d20[n]);
  float d3 = d10 * d9;

// out:
// - d1
// - d3
  o1[n] = d1;
  o3[n] = d3;
}

// old:
//kernel void fused(
//  int totalElements,
//  global float *d36,
//  global float *d37
//) {
//  global float *d33;
//  int linearId = get_global_id(0);
//  if(linearId >= totalElements) {
//    return;
//  }
//  int n = linearId;
//  d33[n] = d36[n] + d37[n];
//  //global float *d27;
//  //global float *d28;
//  global float *d19;
//  global float *d20;
//  int n2 = linearId % NARROW;
//  if(n < NARROW) {
//    d19[n2] = tanh(d33[n2]);
//  } else {
//    d20[n2] = sigmoid(d33[n2 + NARROW]);
//  }
//}

