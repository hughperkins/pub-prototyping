from cython cimport view
from cpython cimport array as c_array
from array import array
import threading
from libcpp.string cimport string

cimport cfoo

def sayName( name ):
    cfoo.sayName( name )

