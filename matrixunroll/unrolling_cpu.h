#include <iostream>

class UnrollArgs {
public:
    float *input;
    float *unInput;
    int batchSize;
    int inputPlanes;
    int inputSize;
    int filterSize;
    int outputSize;
    void print() {
        std::cout << input << std::endl;
        std::cout << unInput << std::endl;
        std::cout << batchSize << std::endl;
        std::cout << inputPlanes << std::endl;
        std::cout << inputSize << std::endl;
        std::cout << filterSize << std::endl;
        std::cout << outputSize << std::endl;
    }
};

class UnrollOutputArgs {
public:
    float *unOutputs;
    float *outputs;

    int batchSize;
    int outputSize;
    int numFilters;
    
    int unOutputCols;
};

//void unroll_input( float *input, float *unInput, int batchSize, int inputPlanes, int inputSize, int filterSize, int outputSize );
void unroll_input( UnrollArgs unrollArgs );
void unroll_output(UnrollOutputArgs args);


