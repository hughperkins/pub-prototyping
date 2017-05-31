#include <stdio.h>

extern "C" {
    void hpfn2_(int *a, double *b, double *c);

    void MAIN__(){
    }
}

int main( int argc, char *argv[] ) {
    int a = 23;
    double b = 4.789;
    double c[3];
    c[0] = 1.23;
    c[1] = 2.46;
    c[2] = 3.69;
    hpfn2_(&a, &b, c);
    return 0;
}

