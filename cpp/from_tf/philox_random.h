/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Implement the Philox algorithm to generate random numbers in parallel.
// Salmon et al. SC 2011. Parallel random numbers: as easy as 1, 2, 3.
//   http://www.thesalmons.org/john/random123/papers/random123sc11.pdf

#ifndef TENSORFLOW_LIB_RANDOM_PHILOX_RANDOM_H_
#define TENSORFLOW_LIB_RANDOM_PHILOX_RANDOM_H_

#include <stdlib.h>

// #include "tensorflow/core/platform/types.h"

typedef uint32_t uint32;
typedef int32_t int32;
typedef uint64_t uint64;

// Function qualifiers that need to work on both CPU and GPU.
// #if defined(__CUDACC__)
// // For nvcc.
// #define PHILOX_DEVICE_FUNC __host__ __device__
// #define PHILOX_INLINE __inline__
// #else
// For non-nvcc.
#define PHILOX_DEVICE_FUNC
#define PHILOX_INLINE inline
// #endif
#define PHILOX_DEVICE_INLINE PHILOX_DEVICE_FUNC PHILOX_INLINE

#include <math.h>
#include <iostream>

namespace tensorflow {
namespace random {

// A class that represents an inline array. It can be used on both CPU and GPU,
// and also trivially copyable between CPU and GPU.
// Arguments:
//   T: the array element type;
//   ElementCount: the fixed size of the array;
template <typename T, int ElementCount>
class Array {
 public:
  PHILOX_DEVICE_INLINE Array() {
    for (int i = 0; i < ElementCount; ++i) {
      data_[i] = T(0);
    }
  }

  PHILOX_DEVICE_INLINE const T& operator[](int index) const {
    return data_[index];
  }

  PHILOX_DEVICE_INLINE T& operator[](int index) { return data_[index]; }

  size_t size() const { return ElementCount; }

 private:
  T data_[ElementCount];
};

// A class that encapsulates all the states for a random number generator using
// the philox_4x32_10 algorithm. Each invocation returns a 128-bit random bits
// in the form of four uint32.
// There are multiple variants of this algorithm, we picked the 4x32_10 version
// that is most suited for our applications.
// Since this class is meant to be copied between CPU to GPU, it maintains a
// value semantics.
//
// For example: To use this class and populate an array of 1024 randoms on CPU
// with two threads,
//
//  void Fill(PhiloxRandom rnd, uint32* output, int start, int limit) {
//    assert(start % 4 == 0);
//    assert(limit % 4 == 0);
//    rnd.Skip(start / 4);
//    for (int i = start; i < limit; i += 4) {
//      auto sample = rnd();
//      ... copy sample[0..3] to output[i..i+3]
//    }
//  }
//
//  PhiloxRandom rng(seed);
//  PhiloxRandom rng_copy = rng;
//  rng.Skip(1000/4);
//
//  ... schedule Fill(rng_copy, output, 0, 512) in thread 1;
//  ... schedule Fill(rng_copy, output, 512, 1024) in thread 2;
//  ... wait for thread 1 & 2 to finish executing Fill().
//
// NOTE:
// 1. PhiloxRandom is trivially copyable.
// 2. PhiloxRandom is compilable by gcc and nvcc.
class PhiloxRandom {
 public:
  typedef Array<uint32, 4> ResultType;
  typedef uint32 ResultElementType;
  // The number of elements that will be returned.
  static const int kResultElementCount = 4;
  // Cost of generation of a single element (in cycles).
  static const int kElementCost = 10;

  PHILOX_DEVICE_INLINE
  PhiloxRandom() {}

  PHILOX_DEVICE_INLINE
  explicit PhiloxRandom(uint64 seed) {
    #ifndef __CUDA_ARCH__
    std::cout << "philox_random.h PhiloxRandom seed " << seed << std::endl;
    #endif
    key_[0] = static_cast<uint32>(seed);
    key_[1] = static_cast<uint32>(seed >> 32);
  }

  PHILOX_DEVICE_INLINE
  explicit PhiloxRandom(uint64 seed_lo, uint64 seed_hi) {
    #ifndef __CUDA_ARCH__
    std::cout << "philox_random.h PhiloxRandom, in pub-prototyping, seed_lo " << seed_lo << " seed_hi " << seed_hi << std::endl;
    #endif
    key_[0] = static_cast<uint32>(seed_lo);
    key_[1] = static_cast<uint32>(seed_lo >> 32);
    counter_[2] = static_cast<uint32>(seed_hi);
    counter_[3] = static_cast<uint32>(seed_hi >> 32);
  }

  // Skip the specified number of samples of 128-bits in the current stream.
  PHILOX_DEVICE_INLINE
  void Skip(uint64 count) {
    #ifndef __CUDA_ARCH__
    std::cout << "philox_random.h skipping " << count << std::endl;
    #endif
    const uint32 count_lo = static_cast<uint32>(count);
    uint32 count_hi = static_cast<uint32>(count >> 32);

    counter_[0] += count_lo;
    if (counter_[0] < count_lo) {
      ++count_hi;
    }

    counter_[1] += count_hi;
    if (counter_[1] < count_hi) {
      if (++counter_[2] == 0) {
        ++counter_[3];
      }
    }
  }

  // Returns a group of four random numbers using the underlying Philox
  // algorithm.
  PHILOX_DEVICE_INLINE ResultType operator()() {
    #ifndef __CUDA_ARCH__
    std::cout << "philox_random.h PhiloxRandom operator() " << std::endl;
    #endif
    ResultType counter = counter_;
    Key key = key_;

    // Run the single rounds for ten times. Manually unrolling the loop
    // for better performance.
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);

    SkipOne();

    return counter;
  }

 private:
  // The type for the 64-bit key stored in the form of two 32-bit uint
  // that are used in the diffusion process.
  typedef Array<uint32, 2> Key;

  // We use the same constants as recommended by the original paper.
  static const uint32 kPhiloxW32A = 0x9E3779B9;
  static const uint32 kPhiloxW32B = 0xBB67AE85;
  static const uint32 kPhiloxM4x32A = 0xD2511F53;
  static const uint32 kPhiloxM4x32B = 0xCD9E8D57;

  // Helper function to skip the next sample of 128-bits in the current stream.
  PHILOX_DEVICE_INLINE void SkipOne() {
    #ifndef __CUDA_ARCH__
    std::cout << "philox_random.h PhiloxRandom SkipOne() " << std::endl;
    #endif
    if (++counter_[0] == 0) {
      if (++counter_[1] == 0) {
        if (++counter_[2] == 0) {
          ++counter_[3];
        }
      }
    }
  }

  // Helper function to return the lower and higher 32-bits from two 32-bit
  // integer multiplications.
  PHILOX_DEVICE_INLINE
  static void MultiplyHighLow(uint32 a, uint32 b, uint32* result_low,
                              uint32* result_high) {
#ifndef __CUDA_ARCH__
    const uint64 product = static_cast<uint64>(a) * b;
    *result_low = static_cast<uint32>(product);
    *result_high = static_cast<uint32>(product >> 32);
#else
    *result_low = a * b;
    *result_high = __umulhi(a, b);
#endif
  }

  // Helper function for a single round of the underlying Philox algorithm.
  PHILOX_DEVICE_INLINE static ResultType ComputeSingleRound(
      const ResultType& counter, const Key& key) {
    std::cout << "PhiloxRandom::ComputeSingleRound(counter, key)" << std::endl;
    uint32 lo0;
    uint32 hi0;

    std::cout << "  counter:";
    for(int i = 0; i < 4; i++) {
      std::cout << " " << counter[i]; 
    }
    std::cout << std::endl;

    std::cout << "  key:";
    for(int i = 0; i < 2; i++) {
      std::cout << " " << key[i]; 
    }
    std::cout << std::endl;

    std::cout << "  pHiloxM4x32A=" << kPhiloxM4x32A << " " << (int)kPhiloxM4x32A <<
       " kPhiloxM4x32B=" << kPhiloxM4x32B << " " << (int)kPhiloxM4x32B << std::endl;

    // std::cout << "  counter[0]=" << counter[0] << std::endl;
    MultiplyHighLow(kPhiloxM4x32A, counter[0], &lo0, &hi0);

    uint32 lo1;
    uint32 hi1;
    // std::cout << "  counter[2]=" << counter[2] << std::endl;
    MultiplyHighLow(kPhiloxM4x32B, counter[2], &lo1, &hi1);

    ResultType result;
    result[0] = hi1 ^ counter[1] ^ key[0];
    result[1] = lo1;
    result[2] = hi0 ^ counter[3] ^ key[1];
    result[3] = lo0;
    std::cout << "  result:";
    for(int i = 0; i < 4; i++) {
      std::cout << " " << result[i];
    }
    std::cout << std::endl;
    return result;
  }

  PHILOX_DEVICE_INLINE void RaiseKey(Key* key) {
    std::cout << "Raisekey()" << std::endl;
    (*key)[0] += kPhiloxW32A;
    (*key)[1] += kPhiloxW32B;
  }

 private:
  ResultType counter_;
  Key key_;
};

}  // namespace random
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_RANDOM_PHILOX_RANDOM_H_
