cmake_minimum_required(VERSION 3.0)

find_library(COCL_RUNTIME_LIBRARY cocl PATHS /usr/local
  HINTS /usr/local/lib)
find_library(CLBLAST_RUNTIME_LIBRARY clblast PATHS /usr/local
  HINTS /usr/local/lib)
find_library(CLEW_RUNTIME_LIBRARY clew PATHS /usr/local
  HINTS /usr/local/lib)
find_library(EASYCL_RUNTIME_LIBRARY easycl PATHS /usr/local
  HINTS /usr/local/lib)

message("COCL ${COCL_RUNTIME_LIBRARY}")
message("CLBLAST ${CLBLAST_RUNTIME_LIBRARY}")
message("EASYCL ${EASYCL_RUNTIME_LIBRARY}")
message("CLEW ${CLEW_RUNTIME_LIBRARY}")

SET(Coriander_LIBRARIES ${COCL_RUNTIME_LIBRARY} ${CLBLAST_RUNTIME_LIBRARY} ${EASYCL_RUNTIME_LIBRARY} ${CLEW_RUNTIME_LIBRARY})
