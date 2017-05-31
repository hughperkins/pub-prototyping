cmake_minimum_required(VERSION 3.0)

add_custom_command(
    OUTPUT simple.o
    COMMAND echo building simple.o
    COMMAND echo 'INCLUDE_DIRECTORIES: $<TARGET_PROPERTY:simple,INCLUDE_DIRECTORIES>'
    COMMAND cocl -c -I '$<TARGET_PROPERTY:simple,INCLUDE_DIRECTORIES>' -o ${CMAKE_CURRENT_BINARY_DIR}/simple.o ${CMAKE_CURRENT_SOURCE_DIR}/simple.cu
)

#    COMMAND echo INCLUDE_DIRECTORIES: $<JOIN:$<TARGET_PROPERTY:simple,INCLUDE_DIRECTORIES>, -I>

#    COMMAND echo 'INCLUDE_DIRECTORIES: $<JOIN:$<TARGET_PROPERTY:simple,INCLUDE_DIRECTORIES>, -I>'
# COMMAND echo include_directories: [$<TARGET_PROPERTY:simple,INCLUDE_DIRECTORIES>]

add_custom_target(build_foo
    DEPENDS foo
)

# target_include_directories(build_foo bar)

add_executable(simple simple.o file2.cpp)
set_target_properties(simple PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(simple PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(simple PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include2)
get_target_property(probe simple INCLUDE_DIRECTORIES)
message("probe ${probe}")
