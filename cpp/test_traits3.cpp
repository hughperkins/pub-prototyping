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
struct supports_optimized {
    static const bool value = false;
};

template< typename T >
void run_algorithm( T& object ) {
    algorithm_selector< supports_optimized< T >::value >::implementation(object);
}

class ClassA {
public:
    void default_algorithm() {
        std::cout << "ClassA::default_algorithm" << std::endl;
    }
};

class ClassB {
public:
    void optimized_algorithm() {
        std::cout << "ClassB::optimized_algorithm" << std::endl;
    }
};

template<>
struct supports_optimized< ClassB > {
    static const bool value = true;
};

int main(int argc, char *argv[]) {
    ClassA a;
    ClassB b;
    run_algorithm(a);
    run_algorithm(b);
    return 0;
}
