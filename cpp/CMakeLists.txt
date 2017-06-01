cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-std=c++11 -fPIC -g")

# add_executable(testpthread testpthread.cpp)
# target_link_libraries(testpthread pthread)

# add_executable(testlambda testlambda.cpp)

# add_executable(test_uniqueptr test_uniqueptr.cpp)
# add_executable(testcast testcast.cpp)

add_executable(testphilox testphilox.cpp)
set(TFDIR "CHANGEME" CACHE STRING "cloned tensorflow repo")
set(EIGENDIR "CHANGEME" CACHE STRING "cloned eigen repo")
target_include_directories(testphilox PRIVATE ${TFDIR} ${EIGENDIR})

# add_executable( int int.cpp )

# link_directories(/data/install/svmlight/build)
# include_directories(/data/install/svmlight)

# add_library(svmlightrunner SHARED svmlightrunner.c )

# link_libraries(svmlight)
# link_libraries(m)
# link_libraries(svmlightrunner)

# #add_executable(test_svmlight test_svmlight.c )

# #include_directories(/data/install/autoserial-1.0.1/include)
# #link_directories(/data/install/autoserial-1.0.1/autoserial)
# #link_libraries(autoserial)
# find_package(MPI)
# #include_directories(${MPI_C_INCLUDE_PATH})
# #link_libraries(${MPI_C_LIBRARIES})
# include_directories(/usr/include/mpich2)
# link_directories(/usr/lib/mpich2/lib)

# #link_directories(/usr/lib)
# #link_libraries(/usr/lib/libboost_serialization.so.1.46.1)

# #find_package(Boost 1.46.1 REQUIRED serialization)

# #add_executable(testmpi_boostserialization testmpi_boostserialization.cpp MyException.cpp )

# include_directories(/data/install/eigen)
# add_executable(testEigenSolver testEigenSolver.cpp)

# find_package(Freetype)
# include_directories(${FREETYPE_INCLUDE_DIR_freetype2})
# link_libraries(${FREETYPE_LIBRARY})

# find_package(PNG)
# link_libraries(${PNG_LIBRARY})

# link_libraries(pthread)
# link_libraries(/usr/lib/libboost_thread-mt.so)
# add_executable(testkillthread testkillthread.cpp)
# add_executable(testboostthread testboostthread.cpp)

# add_executable( testfreetype testfreetype.cpp )

# #add_libraries(/usr/include/boost/boo

# include_directories(../../McmcSVM/McmcSVM)

# add_executable( testsharedptr testsharedptr.cpp )
# add_executable( testmpi testmpi.cpp ../../McmcSVM/McmcSVM/utils/mpi/mpi_helper.cpp )
# target_link_libraries(testmpi mpich mpl)
# add_executable( testmpib testmpib.cpp )
# target_link_libraries(testmpib mpich mpl)

# add_executable( mpiparent mpiparent.cpp ../../McmcSVM/McmcSVM/utils/mpi/mpi_reduce.cpp )
# add_executable( mpichild mpichild.cpp ../../McmcSVM/McmcSVM/utils/mpi/mpi_reduce.cpp )

