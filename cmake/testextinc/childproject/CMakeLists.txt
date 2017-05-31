cmake_minimum_required(VERSION 2.8)
project(childproject)
message("child prefix ${CMAKE_INSTALL_PREFIX}")

add_library( childproject SHARED child.cpp )

install( TARGETS childproject EXPORT childproject
    LIBRARY DESTINATION lib )
install( FILES child.h
    DESTINATION include)
install(EXPORT childproject DESTINATION lib )

