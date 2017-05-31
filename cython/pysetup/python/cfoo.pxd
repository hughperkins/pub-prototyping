from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "foo.h":
    void sayName( string name )

