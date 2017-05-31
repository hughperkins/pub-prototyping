#include <string>
#include <iostream>
using namespace std;

#include "child.h"

void child_sayName(string name) {
    cout << "child lib. name is " << name << endl;
}

string getName() {
    return "I am child library";
}

