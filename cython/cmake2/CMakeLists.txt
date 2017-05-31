CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX
    "${CMAKE_CURRENT_SOURCE_DIR}/dist" CACHE PATH "Installation prefix, default 'dist'" FORCE
    )
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

include(UseCython)

include_directories(.)

set_source_files_properties(testpylib.pyx PROPERTIES CYTHON_IS_CXX TRUE)
cython_add_module(testpylib testpylib.pyx testcxxmodule.cpp)

install(TARGETS testpylib
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)
install(FILES test_call.py
    DESTINATION bin)

