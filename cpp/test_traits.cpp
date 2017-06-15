#include <iostream>

template<typename T>
struct is_void {
    const static bool value = false;
};

template<>
struct is_void<void> {
    const static bool value = true;
};

template<typename T>
struct is_pointer {
    const static bool value = false;
};

template<typename T>
struct is_pointer<T*> {
    const static bool value = true;
};



template< typename T >
struct supports_optimized_implementation{
    const static bool value = false;
};

template<bool b>
struct algorithm_selector {
    template<typename T>
    static void implementation(T &object) {
        std::cout << "algorithm operating on object" << std::endl;
    }
};

template<>
struct algorithm_selector< true > {
    template< typename T >
    static void implementation( T& object ) {
        object.optimized_implementation();
    }
};

template<>
struct algorithm_selector< false > {
    template< typename T >
    static void implementation( T& object ) {
        object.default_implementation();
    }
};

template< typename T >
void algorithm( T& object ) {
    algorithm_selector< supports_optimized_implementation< T >::value >::implementation(object);
}

class ClassB {
public:
    void optimized_implementation() {
        std::cout << "ObjectB::optimized_implementation()" << std::endl;
    }
};

template<>
struct
supports_optimized_implementation< ClassB > {
    static const bool value = true;
};

class ClassA {
public:
    void default_implementation() {
        std::cout << "ClassA::default_implementation()" << std::endl;
    }
};

// template< typename T>
// void algorithm( T& object ) {
//     algorithm_selector< supports_optimized_implementation< ObjectB > {

//     }
// }

int main(int argc, char *argv[]) {
    std::cout << "test_traits.cpp" << std::endl;
    float a;
    std::cout << "is_void" << std::endl;
    std::cout << is_void<float>::value << std::endl;
    std::cout << is_void<void>::value << std::endl;
    std::cout << std::endl;

    std::cout << "is_pointer" << std::endl;
    std::cout << is_pointer<void>::value << std::endl;
    std::cout << is_pointer<void *>::value << std::endl;
    std::cout << is_pointer<float *>::value << std::endl;
    std::cout << is_pointer<float>::value << std::endl;
    std::cout << is_pointer<float **>::value << std::endl;
    std::cout << std::endl;

    ClassA A;
    ClassB B;
    algorithm(A);
    algorithm(B);

    return 0;
}
