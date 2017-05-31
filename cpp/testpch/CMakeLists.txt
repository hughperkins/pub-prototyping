cmake_minimum_required(VERSION 2.6)

include_directories(/data/install/eigen)
include_directories(.)

MACRO(GCC_SETUP_PCH)
  MESSAGE("creating pch.")
  EXEC_PROGRAM(${CMAKE_CXX_COMPILER} 
    ARGS stdafx.h -g -o stdafx.h.gch
    OUTPUT_VARIABLE CMAKE_C_COMPILER_BUILD_PCH_OUTPUT)
  MESSAGE("${CMAKE_C_COMPILER_BUILD_PCH_OUTPUT}")
ENDMACRO(GCC_SETUP_PCH)

MACRO (ADD_PCH_RULE  _header_filename _src_list)
	SET(_gch_filename "${_header_filename}.gch")
	LIST(APPEND ${_src_list} ${_gch_filename})

   string( TOUPPER "CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}" flags_for_build_name )
   set( compile_flags ${CMAKE_CXX_FLAGS} ${${flags_for_build_name}} )
   MESSAGE("compile_flags: ${compile_flags}")

	SET (_args ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}})
   MESSAGE("${_args}")
   MESSAGE("${COMPILE_FLAGS}")
   SET( _args ${compile_flags})
	LIST(APPEND _args -c ${CMAKE_CURRENT_SOURCE_DIR}/${_header_filename} -o ${CMAKE_CURRENT_SOURCE_DIR}/${_gch_filename})
   MESSAGE("${_args}")
	GET_DIRECTORY_PROPERTY(DIRINC INCLUDE_DIRECTORIES)
	foreach (_inc ${DIRINC})
		LIST(APPEND _args "-I" ${_inc})
	endforeach(_inc ${DIRINC})
	SEPARATE_ARGUMENTS(_args)
	add_custom_command(OUTPUT ${_gch_filename}
		   COMMAND rm -f ${_gch_filename}
		   COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} ${_args}
			    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${_header_filename})
ENDMACRO(ADD_PCH_RULE _header_filename _src_list)

ADD_PCH_RULE(stdafx.h PCHHEADERS)
MESSAGE("${PCHHEADERS}")

#add_executable(mymain mymain.cpp ${PCHHEADERS})
add_executable(mymain mymain.cpp )

