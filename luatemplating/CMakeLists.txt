cmake_minimum_required(VERSION 2.8)

Find_Package(Lua51)
include_directories(${LUA_INCLUDE_DIR})
link_libraries(${LUA_LIBRARIES})

add_executable( test test.cpp )

