#include <iostream>
#include <random>

#include "OpenCLHelper.h"
#include <clBLAS.h>

#include "stringhelper.h"
#include "Timer.h"
#include "unrolling_cpu.h"

using namespace std;

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

    if( !OpenCLHelper::isOpenCLAvailable() ) {
        cout << "opencl library not found" << endl;
        exit(1);
    }
    cout << "found opencl library" << endl;
    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();

    cl_int err;

    err = clblasSetup();
    if (err != CL_SUCCESS) {
        printf("clblasSetup() failed with %d\n", err);
        delete cl;
        return 1;
    }

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
    UnrollArgs unrollArgs;
    unrollArgs.input = input;
    unrollArgs.unInput = unInput;
    unrollArgs.batchSize = batchSize;
    unrollArgs.inputPlanes = inputPlanes;
    unrollArgs.inputSize = inputSize;
    unrollArgs.filterSize = filterSize;
    unrollArgs.outputSize = outputSize;
    unroll_input( unrollArgs );
    const int filterSizeSquared = filterSize * filterSize;
//    for( int outputRow = 0; outputRow < outputSize; outputRow++ ) {
//        for( int outputCol = 0; outputCol < outputSize; outputCol++ ) {
//            int unInRow = outputRow * outputSize + outputCol;
//            for( int inputPlaneIdx = 0; inputPlaneIdx < inputPlanes; inputPlaneIdx++ ) {
//                //float *filterPlane = filters + filter * filterSize * filterSize;
//                float *inputPlane = input + inputPlaneIdx * inputSize * inputSize;
//                for( int kernelRow = 0; kernelRow < filterSize; kernelRow++ ) {
//                    for( int kernelCol = 0; kernelCol < filterSize; kernelCol++ ) {
//                        int unInCol = inputPlaneIdx * filterSizeSquared + kernelRow * filterSize + kernelCol;
//                        int inputRow = outputRow + kernelRow;
//                        int inputCol = outputCol + kernelCol;
//                        unInput[unInRow * unInputCols + unInCol] = inputPlane[inputRow * inputSize + inputCol];
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

    // matmult
    // (uninputrows) * (unfiltercols) * (uninputcols)
    // o * o * 1 * k * k * N
    const int unOutputRows = unInputRows;
    const int unOutputCols = unFilterCols;
    float *unOutputs = new float[unOutputRows * unOutputCols];


    int M = unInputRows;
    int K = unInputCols;
    int N = unOutputCols;
    cout << "uninput " << unInputRows << " " << unInputCols << endl;
    cout << "unFilters " << unFilterRows << " " << unFilterCols << endl;
    cout << "unOutputs " << unOutputRows << " " << unOutputCols << endl;
    cout << "M=" << M << " K=" << K << " N=" << N << endl;

    CLFloatWrapper *unInputWrap = cl->wrap( unInputRows * unInputCols, unInput );
    CLFloatWrapper *unFiltersWrap = cl->wrap( unFilterRows * unFilterCols, unFilters );
    CLFloatWrapper *unOutputsWrap = cl->wrap( unOutputRows * unOutputCols, unOutputs );
    unInputWrap->copyToDevice();
    unFiltersWrap->copyToDevice();
    unOutputsWrap->createOnDevice();

    size_t lda = K;        /* i.e. lda = K */
    size_t ldb = N;        /* i.e. ldb = N */
    size_t ldc = N;        /* i.e. ldc = N */

    cl_event event = NULL;
    err = clblasSgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans, M, N, K,
                         1, unInputWrap->getBuffer(), 0, lda,
                         unFiltersWrap->getBuffer(), 0, ldb, 0,
                         unOutputsWrap->getBuffer(), 0, ldc,
                         1, cl->queue, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("clblasSgemmEx() failed with %d\n", err);
        delete cl;
        return 1;
    }
    else {
        err = clWaitForEvents(1, &event);
    }
    unOutputsWrap->copyToHost();


//    for( int i = 0; i < unInputRows; i++ ) {
//        float *unInputRow = unInput + i * unInputCols;
//        for( int k = 0; k < unFilterCols; k++ ) {
//            float sum = 0;
//            float *unFilterCol = unFilters + k;
//            for( int j = 0; j < unInputCols; j++ ) {
////                float leftValue = unInput[i * unInputCols + j];
//                float leftValue = unInputRow[j];
////                float rightValue = unFilters[j * unFilterCols + k];
//                float rightValue = unFilterCol[j * unFilterCols];
//                sum += leftValue * rightValue;
//            }
//            unOutputs[i * unFilterCols + k] = sum;
//        }
//    }
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

//    const int outputSizeSquared = outputSize * outputSize;
//    for( int outputRow = 0; outputRow < outputSize; outputRow++ ) {
//        for( int outputCol = 0; outputCol < outputSize; outputCol++ ) {
//            float *unOutputsRow = unOutputs + ( outputRow * outputSize + outputCol ) * numFilters;
//            const int outputPlanePos = outputRow * outputSize + outputCol;
//            for( int filter = 0; filter < numFilters; filter++ ) {
//                outputs[filter * outputSizeSquared + outputPlanePos] =
//                    unOutputsRow[filter];
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

    clblasTeardown();
    delete cl;

    delete[] unOutputs;
    delete[] unFilters;
    delete[] unInput;
    delete[] outputs;
    delete[] filters;
    delete[]input;
}


