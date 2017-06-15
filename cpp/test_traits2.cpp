#include <iostream>

template< bool b >
struct algorithm_selector {
    template< typename T >
    static void implementation(T& object) {
        object.default_algorithm();
    }
};

template<>
struct algorithm_selector< true > {
    template< typename T >
    static void implementation(T& object) {
        object.optimized_algorithm();
    }
};

template< typename T >
struct supports_optimized_implementation {
    const static bool value = false;
};

class ClassA {
public:
    void default_algorithm() {
        std::cout << "ClassA::default_implementation()" << std::endl;
    }
};

class ClassB {
public:
    void optimized_algorithm() {
        std::cout << "ClassB::optimized_implementation()" << std::endl;
    }
};

template<>
struct supports_optimized_implementation< ClassB > {
    const static bool value = true;
};

template< typename T >
void algorithm( T& target ) {
    algorithm_selector< supports_optimized_implementation< T >::value >::implementation(target);
}

int main(int argc, char *argv[]) {
    ClassA a;
    ClassB b;
    algorithm(a);
    algorithm(b);
    return 0;
}
