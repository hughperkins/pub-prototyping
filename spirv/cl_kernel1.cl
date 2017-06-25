kernel void mykernel(global char *cmem0, unsigned int offset) {
    global int *data0 = (global int *)(cmem0 + offset);
    int tid = get_global_id(0);
    data0[tid] = tid + 123;
}

