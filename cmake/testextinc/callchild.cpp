#include <iostream>
using namespace std;

#include "child.h"

int main( int argc, char *argv[] ) {
    cout << "calling child; child returns " << getName() << endl;
    return 0;
}

