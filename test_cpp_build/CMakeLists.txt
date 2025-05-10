cmake_minimum_required(VERSION 3.16)
project(SimpleTest)

# set(CMAKE_CXX_COMPILER "/usr/bin/clang++-15" CACHE STRING "c++ compiler")
# set(CMAKE_C_COMPILER "/usr/bin/clang-15" CACHE STRING "c compiler")
# set(CMAKE_CXX_FLAGS "-stdlib=libc++" CACHE STRING "c++ flags")
# set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=lld-15 -lc++ -lc++abi" CACHE STRING "linker flags")

# set(CMAKE_CXX_COMPILER_WORKS 1)  # Skip compiler test
# set(CMAKE_CXX_LINKER_WORKS 1)    # Skip linker test

# set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=lld")
# set(CMAKE_CXX_FLAGS_INIT "-stdlib=libc++")
# set(CMAKE_EXE_LINKER_FLAGS_INIT "-fuse-ld=lld-15 -lc++ -lc++abi") 
# set(CMAKE_CXX_FLAGS "-stdlib=libc++" CACHE STRING "c++ flags" FORCE) 
# set(CMAKE_LINKER "/usr/bin/ld.lld-15" CACHE STRING "c++ linker" FORCE)
# set(CMAKE_C_COMPILER "/usr/bin/clang-15" CACHE STRING "C++ compiler" FORCE)
# set(CMAKE_CXX_COMPILER "/usr/bin/clang++-15" CACHE STRING "C++ compiler" FORCE)

add_executable(test test.cpp)
# target_link_libraries(test PUBLIC stdc++)
# target_link_libraries(test PUBLIC c++ c++abi) 

# set(CMAKE_CXX_COMPILER_WORKS 1)  # Skip compiler test
# set(CMAKE_CXX_LINKER_WORKS 1)    # Skip linker test

