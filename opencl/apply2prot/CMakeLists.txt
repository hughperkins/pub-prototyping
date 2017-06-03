cmake_minimum_required(VERSION 2.8)

set(HOME "$ENV{HOME}")
# message("HOME ${HOME}")
find_library( EASYCL_LIBRARIES
        NAMES EasyCL
        HINTS
        PATHS
            ${HOME}/git/EasyCL/dist/lib
    )
find_library( CLEW_LIBRARIES
        NAMES clew
        HINTS
        PATHS
            ${HOME}/git/EasyCL/dist/lib
    )

add_executable(testapply2 testapply2.cpp)
target_link_libraries(testapply2 ${EASYCL_LIBRARIES}  ${CLEW_LIBRARIES})
target_link_libraries(testapply2 dl)

add_executable(testapply2t testapply2t.cpp)
target_link_libraries(testapply2t ${EASYCL_LIBRARIES}  ${CLEW_LIBRARIES})
target_link_libraries(testapply2t dl)

add_executable(testapply2tnot testapply2tnot.cpp)
target_link_libraries(testapply2tnot ${EASYCL_LIBRARIES}  ${CLEW_LIBRARIES})
target_link_libraries(testapply2tnot dl)

include_directories("${HOME}/git/EasyCL/dist/include/easycl")
include_directories("${HOME}/git/EasyCL/dist/include")

