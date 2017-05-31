from __future__ import print_function
from libcpp.string cimport string

cdef extern from "testcxxmodule.h":
    cpdef void sayAnythingFromCpp()
    cpdef void sayAStringFromCpp(string mystring)

