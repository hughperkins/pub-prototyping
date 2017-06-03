#include <iostream>
#include <random>
using namespace std;

#include "stringhelper.h"
#include "Timer.h"

#include "unrolling_cpu.h"

void printMatrix( string name, int rows, int cols, float *mat ) {
    cout << name << ":" << endl;
    for( int i = 0; i < rows && i <= 5; i++ ) {
        for( int j = 0; j < cols && j <= 5; j++ ) {
            cout << mat[i * cols + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

int main( int argc, char * argv[] ) {
    mt19937 random;

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
    const int unInputRows = outputSize * outputSize * batchSize;
    const int unInputCols = filterSize * filterSize * inputPlanes;

    const int unFilterRows = unInputCols;
    const int unFilterCols = numFilters;

    float *unInput = new float[unInputRows * unInputCols];
    float *unFilters = new float[unFilterRows * unFilterCols];

    // unroll input
    // o * o * k * k * N
    const int inputSizeSquared = inputSize * inputSize;
    const int filterSizeSquared = filterSize * filterSize;
    const int outputSizeSquared = outputSize * outputSize;

    UnrollArgs unrollArgs;
    unrollArgs.input = input;
    unrollArgs.unInput = unInput;
    unrollArgs.batchSize = batchSize;
    unrollArgs.inputPlanes = inputPlanes;
    unrollArgs.inputSize = inputSize;
    unrollArgs.filterSize = filterSize;
    unrollArgs.outputSize = outputSize;
    unroll_input( unrollArgs );
//    unroll_input( input, unInput, batchSize, inputPlanes, inputSize, filterSize, outputSize );

//    for( int n = 0; n < batchSize; n++ ) {
//        float *inputCube = input + n * inputPlanes * inputSizeSquared;
//        float *unInputCube = unInput + n * outputSizeSquared * unInputCols;
//        for( int outputRow = 0; outputRow < outputSize; outputRow++ ) {
//            for( int outputCol = 0; outputCol < outputSize; outputCol++ ) {
//                int unInRow = outputRow * outputSize + outputCol;
//                for( int inputPlaneIdx = 0; inputPlaneIdx < inputPlanes; inputPlaneIdx++ ) {
//                    //float *filterPlane = filters + filter * filterSize * filterSize;
//                    float *inputPlane = inputCube + inputPlaneIdx * inputSize * inputSize;
//                    for( int kernelRow = 0; kernelRow < filterSize; kernelRow++ ) {
//                        for( int kernelCol = 0; kernelCol < filterSize; kernelCol++ ) {
//                            int unInCol = inputPlaneIdx * filterSizeSquared + kernelRow * filterSize + kernelCol;
//                            int inputRow = outputRow + kernelRow;
//                            int inputCol = outputCol + kernelCol;
//                            unInputCube[unInRow * unInputCols + unInCol] = inputPlane[inputRow * inputSize + inputCol];
//                        }
//                    }
//                }
//            }
//        }
//    }
    timer.timeCheck("unroll input");

    // unroll filters
    // k * k * numFilters
    for( int filter = 0; filter < numFilters; filter++ ) {
        float *filterCube = filters + filter * inputPlanes * filterSizeSquared;
        for( int inputPlaneIdx = 0; inputPlaneIdx < inputPlanes; inputPlaneIdx++ ) {
            float *filterPlane = filterCube + inputPlaneIdx * filterSizeSquared;
            for( int filterRow = 0; filterRow < filterSize; filterRow++ ) {
                for( int filterCol = 0; filterCol < filterSize; filterCol++ ) {
                    int filterLinear = filterRow * filterSize + filterCol;
                    int unFilterRow = inputPlaneIdx * filterSizeSquared + filterLinear;
                    int unFilterCol = filter;
                    unFilters[unFilterRow * unFilterCols + unFilterCol] = 
                        filterPlane[filterLinear];
                }
            }
        }
    }
    timer.timeCheck("unroll filters");

    printMatrix( "input[0]", inputSize, inputSize, input );
    printMatrix( "input[1]", inputSize, inputSize, input + inputSize * inputSize );
    printMatrix( "filters[0]", filterSize, filterSize, filters );
    printMatrix( "filters[1]", filterSize, filterSize, filters + filterSize * filterSize );
    printMatrix( "unInput", unInputRows, unInputCols, unInput );
    printMatrix( "unFilters", unFilterRows, unFilterCols, unFilters );

    //unFilters = kernel;

    // matmult
    // (uninputrows) * (unfiltercols) * (uninputcols)
    // o * o * 1 * k * k * N
    const int unOutputRows = unInputRows;
    const int unOutputCols = unFilterCols;
    float *unOutputs = new float[unOutputRows * unOutputCols];
    for( int i = 0; i < unInputRows; i++ ) {
        float *unInputRow = unInput + i * unInputCols;
        for( int k = 0; k < unFilterCols; k++ ) {
            float sum = 0;
            float *unFilterCol = unFilters + k;
            for( int j = 0; j < unInputCols; j++ ) {
//                float leftValue = unInput[i * unInputCols + j];
                float leftValue = unInputRow[j];
//                float rightValue = unFilters[j * unFilterCols + k];
                float rightValue = unFilterCol[j * unFilterCols];
                sum += leftValue * rightValue;
            }
            unOutputs[i * unFilterCols + k] = sum;
        }
    }
    timer.timeCheck("mat mult");
//    printMatrix("unOutputs", unOutputRows, unOutputCols, unOutputs);

    UnrollOutputArgs unrollOutArgs;
    unrollOutArgs.unOutputs = unOutputs;
    unrollOutArgs.outputs = outputs;

    unrollOutArgs.batchSize = batchSize;
    unrollOutArgs.outputSize = outputSize;
    unrollOutArgs.numFilters = numFilters;
    
    unrollOutArgs.unOutputCols = unOutputCols;
    unroll_output(unrollOutArgs);
    
//    for( int n = 0; n < batchSize; n++ ) {
//        float *unOutputCube = unOutputs + n * unOutputCols * outputSizeSquared;
//        float *outputCube = outputs + n * numFilters * outputSizeSquared;
//        for( int outputRow = 0; outputRow < outputSize; outputRow++ ) {
//            for( int outputCol = 0; outputCol < outputSize; outputCol++ ) {
//                float *unOutputsRow = unOutputCube + ( outputRow * outputSize + outputCol ) * numFilters;
//                const int outputPlanePos = outputRow * outputSize + outputCol;
//                for( int filter = 0; filter < numFilters; filter++ ) {
//                    outputCube[filter * outputSizeSquared + outputPlanePos] =
//                        unOutputsRow[filter];
//                }
//            }
//        }
//    }
    timer.timeCheck("unroll output");
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

//    for( int row = 0; row <= 5; row++ ) {
//        for( int i = 0; i <= 5; i++ ) {
//            cout << unOutput[row * outputSize + i] << " ";
//        }
//        cout << endl;
//    }

    // print output
//    for( int outRow = 0; outRow <= 5; outRow++ ) {
//        string line = "";
//        for( int outCol = 0; outCol <= 5; outCol++ ) {
//            line += toString( output[outRow * outputSize + outCol] ) + " ";
//        }
//        cout << line << endl;
//    }

    delete[] unOutputs;
    delete[] unFilters;
    delete[] unInput;
    delete[] outputs;
    delete[] filters;
    delete[]input;
}


