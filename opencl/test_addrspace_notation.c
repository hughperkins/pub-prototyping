#include <stdio.h>

#define __space(a)

struct MyStruct {
    __space(3:.) float *pFloat;
};

int main(int argc, char *argv[]) {
    __space(0) float a;
    __space(3;0) float *pa;
    __space(3:0) float *pb;
    __space(3:0:0) float **ppb;
    __space(3:0:0) float **ppc;
    __space(0) struct MyStruct myStruct;
    __space(3:0) float *pFloat = myStruct.pFloat;
    return 0;
}
