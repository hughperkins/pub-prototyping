add_library( myc SHARED myc.c )

add_executable( testcallmyc testcallmyc.c )
target_link_libraries( testcallmyc myc )

add_library( mycpp SHARED mycpp.cpp ) 

