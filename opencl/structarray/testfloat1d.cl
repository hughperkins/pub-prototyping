kernel void test(int totalElements, global float *data) {
  int offset = 0;
  int linearId = get_global_id(0);
  if( linearId >= totalElements ) {
    return;
  }
  data[linearId] = data[linearId] + 1;
}

