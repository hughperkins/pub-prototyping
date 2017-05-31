#include <iostream>
#include <vector>
using namespace std;

#include "stringhelper.cpp"

int main(int argc, char *argv[]){
   cout << "trimmed: [" << trim(argv[1]) << "]" << endl;

   vector<string> splitstring = split(argv[1], ";" );
   for(int i = 0; i < (int)splitstring.size(); i++ ) {
      cout << i << ": " << splitstring[i] << endl;
   }
   return 0;
}

