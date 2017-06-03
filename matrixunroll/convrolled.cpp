#include <iostream>
#include <random>
using namespace std;

#include "stringhelper.h"
#include "Timer.h"

int main( int argc, char * argv[] ) {
    mt19937 random;
//    for( int i = 0; i < 10; i++ ) {
//        cout << random() << endl;
//    }

    #include "dim.h"

    float *input = new float[inputSize * inputSize * inputPlanes * batchSize];
    for( int i = 0; i < inputSize * inputSize * inputPlanes * batchSize; i++ ) {
        input[i] = ( (int)random() % 10000 - 5000 ) / 5000.0f;
    }
    float *filters = new float[filterSize * filterSize * numFilters * inputPlanes];
    for( int i = 0; i < filterSize * filterSize * numFilters * inputPlanes; i++ ) {
        filters[i] = ( ( (int)random() % 10000 ) - 5000 ) / 5000.0f;
//        cout << kernel[i] << endl;
    }
    const int outputSize = inputSize - filterSize + 1;
    float *outputs = new float[outputSize * outputSize * numFilters * batchSize];
    for( int i = 0; i < outputSize * outputSize * numFilters * batchSize; i++ ) {
        outputs[i] = 0;
    }
    Timer timer;
    // i*i*k*k
    const int inputSizeSquared = inputSize * inputSize;
    const int filterSizeSquared = filterSize * filterSize;
    const int outputSizeSquared = outputSize * outputSize;
    for( int n = 0; n < batchSize; n++ ) {
        float *outputCube = outputs + n * numFilters * outputSizeSquared;
        float *inputCube = input + n * inputPlanes * inputSizeSquared;
        for( int filter = 0; filter < numFilters; filter++ ) {
            float *filterCube = filters + filter * filterSize * filterSize * inputPlanes;
            float *outputPlane = outputCube + filter * outputSize * outputSize;
            for( int outRow = 0; outRow < outputSize; outRow++ ) {
                for( int outCol = 0; outCol < outputSize; outCol++ ) {
                    float sum = 0;
                    for( int inputPlaneIdx = 0; inputPlaneIdx < inputPlanes; inputPlaneIdx++ ) {
                        float *inputPlane = inputCube + inputPlaneIdx * inputSize * inputSize;
                        float *filterPlane = filterCube + inputPlaneIdx * filterSizeSquared;
                        for( int kRow = 0; kRow < filterSize; kRow++ ) {
                            int inputRow = outRow + kRow;
                            for( int kCol = 0; kCol < filterSize; kCol++ ) {
                                int inputCol = outCol + kCol;
                                float value = inputPlane[inputRow * inputSize + inputCol];
                                float kernelValue = filterPlane[kRow * filterSize + kCol];
                                sum += value * kernelValue;
                            }
                        }
                    }
                    outputPlane[outRow * outputSize + outCol] = sum;
                }
            }
        }
    }
    timer.timeCheck("done");
    for( int sample = 0; sample < 5; sample++ ) {
        int n = random() % batchSize;
        int outputPlane = random() % numFilters;
        int outputRow = random() % outputSize;
        int outputCol = random() % outputSize;
        int outputOffset = ( ( n
                             * numFilters + outputPlane )
                             * outputSize + outputRow )
                             * outputSize + outputCol;
        cout << "n=" << n << " outputPlane=" << outputPlane << " pos=" << outputRow << "," << outputCol  
            << ": " << outputs[outputOffset] << endl;
    }
//    for( int outRow = 0; outRow <= 5; outRow++ ) {
//        string line = "";
//        for( int outCol = 0; outCol <= 5; outCol++ ) {
//            line += toString( output[outRow * outputSize + outCol] ) + " ";
//        }
//        cout << line << endl;
//    }

    delete[] outputs;
    delete[] filters;
    delete[]input;
}


