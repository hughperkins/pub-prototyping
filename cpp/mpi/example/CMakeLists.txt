cmake_minimum_required(VERSION 2.8)

include_directories(/usr/include/mpich2)
include_directories(.)
add_executable( testmpi testmpi.cpp utils/mpi/mpi_helper.cpp utils/stringhelper.cpp )
target_link_libraries( testmpi mpich mpl )

