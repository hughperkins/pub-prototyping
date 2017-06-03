#include "tf_files/philox_random.h"
#include "tf_files/random_distributions.h"

#include "tensorflow/core/platform/types.h"

#include "../third_party/argparsecpp/argparsecpp.h"

#include <iostream>
#include <cassert>

static void fill(tensorflow::random::PhiloxRandom rnd, uint32_t* output, int start, int limit) {
   assert(start % 4 == 0);
   assert(limit % 4 == 0);
   rnd.Skip(start / 4);
   for (int i = start; i < limit; i += 4) {
     auto sample = rnd();
     for(int j = 0; j < 4; j++) {
        output[i + j] = sample[j];
     }
   }
 }

void uniform(int N) {
    // tensorflow::random::PhiloxRandom rand(123);
    tensorflow::random::PhiloxRandom rand(87654321, 123);
    uint32_t numbers[N];
    fill(rand, numbers, 0, N);
    for(int i=0; i < N; i++) {
      std::cout << i << ": " << numbers[i] << " " << tensorflow::random::Uint32ToFloat(numbers[i]) << std::endl;
    }
}

void normal(int N) {
    // tensorflow::random::PhiloxRandom rand(123);
    tensorflow::random::PhiloxRandom gen(87654321, 123);
    tensorflow::random::NormalDistribution<tensorflow::random::PhiloxRandom, float> normal;
    tensorflow::random::Array<float, gen.kResultElementCount> res = normal(&gen);
    for(int i = 0; i < gen.kResultElementCount; i++) {
      std::cout << "  " << i << ": " << res[i] << std::endl;
    }
    // uint32_t numbers[N];
    // fill(gen, numbers, 0, N);
    // for(int i=0; i < N; i++) {
    //   std::cout << i << ": " << numbers[i] << " " << tensorflow::random::Uint32ToFloat(numbers[i]) << std::endl;
    // }
}

int main(int argc, char *argv[]) {
    int N = 0;
    std::string operation = "";

    argparsecpp::ArgumentParser parser;
    parser.add_string_argument("--operation", &operation)->defaultValue("uniform")->help("[uniform|normal]");
    parser.add_int_argument("--count", &N)->defaultValue(12)->help("how many numbers to generate");
    if(!parser.parse_args(argc, argv)) {
        return -1;
    }
    std::cout << "count: " << N << std::endl;

    if(operation == "uniform") {
      uniform(N);
    } else if(operation == "normal") {
      normal(N);
    }
    return 0;
}
