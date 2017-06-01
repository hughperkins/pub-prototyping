#include "from_tf/philox_random.h"

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core//lib/random/random_distributions.h"

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

int main(int argc, char *argv[]) {
    // tensorflow::random::PhiloxRandom rand(123);
    tensorflow::random::PhiloxRandom rand(87654321, 123);
    uint32_t numbers[12];
    fill(rand, numbers, 0, 12);
    for(int i=0; i < 12; i++) {
        std::cout << i << ": " << numbers[i] << " " << tensorflow::random::Uint32ToFloat(numbers[i]) << std::endl;
    }
    return 0;
}
