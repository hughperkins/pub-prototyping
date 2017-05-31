#include <stdio.h>

/*void hpfn2_(int *a, double *b, double *c, double **d);*/
void hpfn2_(int *a, double *b, double *c, double *d);

void MAIN__(){
}

int main( int argc, char *argv[] ) {
    int a = 23;
    double b = 4.789;
    double c[3];
    c[0] = 1.23;
    c[1] = 2.46;
    c[2] = 3.69;
/*    double **d = malloc( sizeof(double*)*2);*/
/*    d[0] = malloc(sizeof(double)*2);*/
/*    d[1] = malloc(sizeof(double)*2);*/
/*    double d[2][2];*/
/*    d[0][0] = 3;*/
/*    d[1][0] = 8;*/
/*    d[0][1] = 7;*/
/*    d[1][1] = 6;*/
    double d[4];
    d[0] = 4;
    d[1] = 3;
    d[2] = 2;
    d[3] = 1;
    hpfn2_(&a, &b, c, d);
    return 0;
}

