#include <iostream>

class Adder {
public:
    Adder(int amount) : amount(amount) {
    }
    int operator()(int target) {
        return target + amount;
    }
private:
    const int amount;
};

int main(int argc, char *argv[]) {
    Adder add3(3);
    Adder add2(2);

    std::cout << add3(7) << std::endl;
    std::cout << add2(4) << std::endl;

    Adder *

    return 0;
}
