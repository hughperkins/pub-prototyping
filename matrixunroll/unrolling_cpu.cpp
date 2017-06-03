#include "unrolling_cpu.h"

//void unroll_input( float *input, float *unInput, int batchSize, int inputPlanes, int inputSize, int filterSize, int outputSize ) {
void unroll_input( UnrollArgs args ) {
    args.print();

    const int batchSize = args.batchSize;

    const int inputSize = args.inputSize;
    const int filterSize = args.filterSize;
    const int outputSize = args.outputSize;

    const int inputPlanes = args.inputPlanes;

    float *input = args.input;
    float *unInput = args.unInput;
//    const float *filters = args.filters;

    const int inputSizeSquared = inputSize * inputSize;
    const int filterSizeSquared = filterSize * filterSize;
    const int outputSizeSquared = outputSize * outputSize;
    const int unInputCols = filterSizeSquared * inputPlanes;
    for( int n = 0; n < batchSize; n++ ) {
        float *inputCube = input + n * inputPlanes * inputSizeSquared;
        float *unInputCube = unInput + n * outputSizeSquared * unInputCols;
        for( int outputRow = 0; outputRow < outputSize; outputRow++ ) {
            for( int outputCol = 0; outputCol < outputSize; outputCol++ ) {
                int unInRow = outputRow * outputSize + outputCol;
                for( int inputPlaneIdx = 0; inputPlaneIdx < inputPlanes; inputPlaneIdx++ ) {
                    //float *filterPlane = filters + filter * filterSize * filterSize;
                    float *inputPlane = inputCube + inputPlaneIdx * inputSize * inputSize;
                    for( int kernelRow = 0; kernelRow < filterSize; kernelRow++ ) {
                        for( int kernelCol = 0; kernelCol < filterSize; kernelCol++ ) {
                            int unInCol = inputPlaneIdx * filterSizeSquared + kernelRow * filterSize + kernelCol;
                            int inputRow = outputRow + kernelRow;
                            int inputCol = outputCol + kernelCol;
                            unInputCube[unInRow * unInputCols + unInCol] = inputPlane[inputRow * inputSize + inputCol];
                        }
                    }
                }
            }
        }
    }
}

void unroll_output(UnrollOutputArgs args) {
    float *unOutputs = args.unOutputs;
    float *outputs = args.outputs;

    const int batchSize = args.batchSize;
    const int outputSize = args.outputSize;
    const int numFilters = args.numFilters;
    
    const int unOutputCols = args.numFilters;
    const int outputSizeSquared = outputSize * outputSize;
    for( int n = 0; n < batchSize; n++ ) {
        float *unOutputCube = unOutputs + n * unOutputCols * outputSizeSquared;
        float *outputCube = outputs + n * numFilters * outputSizeSquared;
        for( int outputRow = 0; outputRow < outputSize; outputRow++ ) {
            for( int outputCol = 0; outputCol < outputSize; outputCol++ ) {
                float *unOutputsRow = unOutputCube + ( outputRow * outputSize + outputCol ) * numFilters;
                const int outputPlanePos = outputRow * outputSize + outputCol;
                for( int filter = 0; filter < numFilters; filter++ ) {
                    outputCube[filter * outputSizeSquared + outputPlanePos] =
                        unOutputsRow[filter];
                }
            }
        }
    }
}

