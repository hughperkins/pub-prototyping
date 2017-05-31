#include <iostream>
using namespace std;

#include "mpiwrapper.h"

int main() {
   MPIWrapper_Initialize();
   cout << MPIWrapper_getRank << endl;
   MPIWrapper_Finalize();
   return 0;
}

