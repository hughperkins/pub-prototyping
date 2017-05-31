#include "stdafx.h"
using namespace std;
using namespace Eigen;

int main() {
   string foo = "hello";
   cout << foo << endl;
   MatrixXd a(2,2);
   a << 1,2,3,4;
   cout << a << endl;
   return 0;
}


