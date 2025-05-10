set(CMAKE_CXX_COMPILER "/usr/bin/clang++-15" CACHE STRING "c++ compiler")
set(CMAKE_C_COMPILER "/usr/bin/clang-15" CACHE STRING "c compiler")
set(CMAKE_CXX_FLAGS "-stdlib=libc++" CACHE STRING "c++ flags")
set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=lld-15 -lc++ -lc++abi" CACHE STRING "linker falgs")

