#----------------------------------------------------------------
# Generated CMake target import file for configuration "".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "childproject" for configuration ""
set_property(TARGET childproject APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(childproject PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libchildproject.so"
  IMPORTED_SONAME_NOCONFIG "libchildproject.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS childproject )
list(APPEND _IMPORT_CHECK_FILES_FOR_childproject "${_IMPORT_PREFIX}/lib/libchildproject.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
