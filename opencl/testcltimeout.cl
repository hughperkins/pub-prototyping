kernel void test_read( const int one,  const int two, global int *out) {
    const int globalid = get_global_id(0);
    int sum = 0;
    int n = 0;
    while( n < 100000 ) {
        sum = (sum + one ) % 1357 * two;
        n++;
    }
    out[globalid] = sum;
}

