
#include <iostream>

int crash(int b, int a);

int crash(int b, int a) {
    return b / a;
}

int main(int argc, char *argv[]) {
    crash(5, 0);
}
