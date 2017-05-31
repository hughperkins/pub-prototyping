#include "m1.h"

#include <iostream>
#include <vector>

std::vector<std::string> names;

extern "C" {
    void registerName(const char *name);
}

void registerName(const char *name) {
    std::cout << "register " << name << std::endl;
    names.push_back(name);
}

MyClass::MyClass(const char *name) {
    std::cout << "MyClass " << name << std::endl;
    names.push_back(name);
    // registerName(name);
}

int main(int argc, char *argv[]) {
    for(auto it=names.begin(); it != names.end(); it++) {
        std::cout << "registered name: " << *it << std::endl;
    }
    return 0;
}

MyClass obj2("some name m1.cpp");
