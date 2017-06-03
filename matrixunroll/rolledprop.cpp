#include <iostream>
#include <random>
#include "OpenCLHelper.h"

#include "stringhelper.h"
#include "Timer.h"

using namespace std;

int main( int argc, char * argv[] ) {
    mt19937 random;

    if( !OpenCLHelper::isOpenCLAvailable() ) {
        cout << "opencl library not found" << endl;
        exit(1);
    }
    cout << "found opencl library" << endl;
    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();

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
    string options = "-DLINEAR";
    CLKernel *kernel = cl->buildKernel( "../propagate1.cl", "convolve_imagecubes_float2", options );

    Timer timer;

    CLWrapper *inputWrap = cl->wrap( inputSize * inputSize * inputPlanes * batchSize, input );
    CLWrapper *filtersWrap = cl->wrap( filterSize * filterSize * numFilters * inputPlanes, filters );
    CLWrapper *outputsWrap = cl->wrap( outputSize * outputSize * numFilters * batchSize, outputs );
    inputWrap->copyToDevice();
    filtersWrap->copyToDevice();
    outputsWrap->createOnDevice();
    timer.timeCheck("copied to gpu, ish");

    kernel->input(batchSize);
    kernel->input(inputPlanes);
    kernel->input(numFilters);
    kernel->input(inputSize);
    kernel->input(filterSize);
    kernel->input(0);
    kernel->input(inputWrap);
    kernel->input(filtersWrap);
    kernel->output(outputsWrap);

    int globalSize = batchSize * numFilters * outputSize * outputSize;
    int workgroupsize = std::min( globalSize, cl->getMaxWorkgroupSize() );
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
//    cout << "propagate1 globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;
    kernel->run_1d( globalSize, workgroupsize );
    cl->finish();
    timer.timeCheck("cl->finish done");

    outputsWrap->copyToHost();

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

    delete cl;

    delete[] outputs;
    delete[] filters;
    delete[]input;
}


