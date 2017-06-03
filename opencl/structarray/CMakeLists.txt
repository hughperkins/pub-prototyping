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

add_executable(teststructperf teststructperf.cpp)
target_link_libraries(teststructperf ${EASYCL_LIBRARIES} ${CLEW_LIBRARIES})
target_link_libraries(teststructperf dl)

add_executable(teststructperf2 teststructperf2.cpp)
target_link_libraries(teststructperf2 ${EASYCL_LIBRARIES}  ${CLEW_LIBRARIES})
target_link_libraries(teststructperf2 dl)

add_executable(teststructperf2t teststructperf2t.cpp)
target_link_libraries(teststructperf2t ${EASYCL_LIBRARIES}  ${CLEW_LIBRARIES})
target_link_libraries(teststructperf2t dl)

add_executable(teststructperfnostruct teststructperfnostruct.cpp)
target_link_libraries(teststructperfnostruct ${EASYCL_LIBRARIES}  ${CLEW_LIBRARIES})
target_link_libraries(teststructperfnostruct dl)

add_executable(testfloat4 testfloat4.cpp)
target_link_libraries(testfloat4 ${EASYCL_LIBRARIES}  ${CLEW_LIBRARIES})
target_link_libraries(testfloat4 dl)

add_executable(testfloat1d testfloat1d.cpp)
target_link_libraries(testfloat1d ${EASYCL_LIBRARIES}  ${CLEW_LIBRARIES})
target_link_libraries(testfloat1d dl)

add_executable(testfloat4wrapper testfloat4wrapper.cpp)
target_link_libraries(testfloat4wrapper ${EASYCL_LIBRARIES}  ${CLEW_LIBRARIES})
target_link_libraries(testfloat4wrapper dl)

add_executable(testfloat1wrapper testfloat1wrapper.cpp)
target_link_libraries(testfloat1wrapper ${EASYCL_LIBRARIES}  ${CLEW_LIBRARIES})
target_link_libraries(testfloat1wrapper dl)

include_directories("${HOME}/git/EasyCL/dist/include/easycl")
include_directories("${HOME}/git/EasyCL/dist/include")

