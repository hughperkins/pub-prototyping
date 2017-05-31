cimport cmycpp
from cpython cimport array as c_array
from libcpp.string cimport string
#from array import array

def toCppString( pyString ):
    if isinstance( pyString, unicode ):
        return pyString.encode('utf8')
    return pyString

def sayName( name):
    #cdef string namestring = name
    cmycpp.sayName(toCppString(name))

def addFloats( a, b ):
    return cmycpp.addFloats( a, b )

def addToArray( float[:] array, amount ):
    cmycpp.addToArray( array.shape[0], &array[0], amount )

def addToUCArray( unsigned char[:] array, amount ):
    cmycpp.addToUCArray( array.shape[0], &array[0], amount )

def addToIntArray( int[:] array, amount ):
    cmycpp.addToIntArray( array.shape[0], &array[0], amount )

cdef class PyMyClass:
    cdef cmycpp.MyClass *thisptr
     
    def __cinit__(self, intarg = None):
        print('__init__(intarg)')
        if intarg != None:
            self.thisptr = new cmycpp.MyClass(intarg)
        else:
            self.thisptr = new cmycpp.MyClass()

    def warpName(self, inName):
        return self.thisptr.warpName(toCppString(inName))

def some_func(float[:] floats):
    print(floats.shape)
    for f in floats:
        print( f )

