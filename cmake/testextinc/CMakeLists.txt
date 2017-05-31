message(">>>>>PARENT CMAKE START")
cmake_minimum_required(VERSION 2.8)

INCLUDE(ExternalProject)

message("parent prefix ${CMAKE_INSTALL_PREFIX}")

ExternalProject_Add(
    external-childproject
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/childproject
    PREFIX ${CMAKE_BINARY_DIR}/childproject
    CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:STRING=${CMAKE_INSTALL_PREFIX}
)
SET(CHILDPROJECT_INCLUDE_DIRS ${CMAKE_INSTALL_PREFIX}/include)
SET(CHILDPROJECT_LIBRARIES ${CMAKE_INSTALL_PREFIX}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}childproject${CMAKE_SHARED_LIBRARY_SUFFIX})
SET(CHILDPROJECT_FOUND ON)
include_directories( ${CHILDPROJECT_INCLUDE_DIRS})
ADD_LIBRARY(childproject SHARED IMPORTED)

add_executable( callchild callchild.cpp )
target_link_libraries( callchild ${CHILDPROJECT_LIBRARIES} )

ADD_DEPENDENCIES(callchild external-childproject)

INSTALL( TARGETS callchild
    RUNTIME DESTINATION bin )
INSTALL( FILES scripts/activate
    DESTINATION . )
INSTALL( PROGRAMS scripts/run.sh
    DESTINATION . )


message("<<<<PARENT CMAKE FINISH")

