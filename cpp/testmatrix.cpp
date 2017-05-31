#include <iostream>
using namespace std;

#include "Matrix.cpp"

int main() {
   Matrix a("1 2; 3 4; 5 6");
   cout << a << endl;

   Matrix b("3 4; 7 8");
   cout << b << endl;

   Matrix r = a * b;
   cout << r << endl;

   Matrix c("3 5 7; 9 1 3");
   cout << c << endl;
   cout << ( a * b ) * c << endl;
   cout << a * ( b * c ) << endl;

   Matrix s(3,3);
   for( int row = 0; row < 3; row++ ) {
      for( int col = 0; col < 3; col++ ) {
         int val = 0;
         for( int r = 0; r < 2; r++ ) {
            for( int s = 0; s < 2; s++ ) {
               val += a.get(row,s) * b.get(s,r) * c.get(r,col);
            }
         }
         s.get(row,col) = val;
      }
   }
   cout << s << endl;

   Matrix x("3; 5; 7");
   Matrix A("2 3 4; 4 7 9; 11 3 5");
   cout << x.getTranspose() * A * x << endl;
   double value = 0;
   for( int r = 0; r < 3; r++ ) {
      for( int s = 0; s < 3; s++ ) {
         value += x.get(r,0) * A.get(r,s) * x.get(s,0 );
      }
   }
   cout << value << endl;

   cout << "=================================================" << endl;
   Matrix X( "1 3 5; 2 5 8; 9 1 4; 1 4 5; 2 5 11");
   Matrix XT = X.getTranspose();
   Matrix BigLambda("0.1 0 0 0 0; 0 0.4 0 0 0; 0 0 20 0 0; 0 0 0 1.3 0; 0 0 0 0 0.72");
   cout << XT * BigLambda * X << endl;

   Matrix resultM(3,3);
   for( int row = 0; row < 3; row++ ) {
      for( int col = 0; col < 3; col++ ) {
         double sum = 0;
         for( int r = 0; r < 5; r++ ) {
            sum += X.get(r,row) * X.get(r,col) * BigLambda.get(r,r);
         }
         resultM.get(row,col) = sum;
      }
   }
   cout << resultM << endl;
   
   return 0;
}

