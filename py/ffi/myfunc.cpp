#include "myfunc.h"

#include <iostream>

const char * printName(const char * name, int count, float aFloat) {
    std::cout << "printName(name=" << name << ", count=" << count << ", aFloat=" << aFloat << ")" << std::endl;
    return "from myfunc call";
}
