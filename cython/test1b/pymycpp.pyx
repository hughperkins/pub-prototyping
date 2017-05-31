cimport cmycpp
from cpython cimport array as c_array
from libcpp.string cimport string
import thread
import threading
#from array import array

def toCppString( pyString ):
    if isinstance( pyString, unicode ):
        return pyString.encode('utf8')
    return pyString

def sayName( name):
    #cdef string namestring = name
    cmycpp.sayName(toCppString(name))

def throwException( message ):
    cmycpp.throwException( toCppString( message ) )

def addFloats( a, b ):
    return cmycpp.addFloats( a, b )

def addToArray( float[:] array, amount ):
    cmycpp.addToArray( array.shape[0], &array[0], amount )

def addToUCArray( unsigned char[:] array, amount ):
    cmycpp.addToUCArray( array.shape[0], &array[0], amount )

def addToIntArray( int[:] array, amount ):
    cmycpp.addToIntArray( array.shape[0], &array[0], amount )

def interruptableCall( function, args ):
    try:
        mythread = threading.Thread( target=function, args = args )
        mythread.daemon = True
        mythread.start()
        while mythread.isAlive():
            mythread.join(0.1)
            print('join timed out')
    except Exception, errtext:
        print('got exception')
        print(errtext)

cdef void _sleepSecs( int secs ):
    with nogil: 
        cmycpp.cySleepSecs(secs)

def sleepSecs( int secs ):
    interruptableCall( _sleepSecs, [secs] )
    checkException()

def raiseException(float inValue, doRaise ):
    result = cmycpp.cyFnRaiseException(inValue, doRaise)
    checkException()
    return result

def checkException():
    cdef int threwException = 0
    cdef string message = ""
    cmycpp.checkException( &threwException, &message)
    # print('threwException: ' + str(threwException) + ' ' + message ) 
    if threwException:
        raise RuntimeError(message)
    
cdef class PyMyClass:
    cdef cmycpp.MyClass *thisptr
     
    def __cinit__(self, intarg = None):
        print('__init__(intarg)')
        if intarg != None:
            self.thisptr = new cmycpp.MyClass(intarg)
        else:
            self.thisptr = new cmycpp.MyClass()

    def __dealloc__(self):
        del self.thisptr

    def warpName(self, inName):
        return self.thisptr.warpName(toCppString(inName))

    def getFloat(self):
        cdef int threwException = 0
        cdef string message = ""
        result = self.thisptr.getFloat(&threwException, &message)
        print('threwException: ' + str(threwException) + ' ' + message ) 
        if threwException:
            raise RuntimeError(message)
        return result

#    def longOp(self, int secs):
#        with nogil:
#            self.thisptr.longOp(secs)

    def _longOp(self, int secs):
        with nogil:
            self.thisptr.longOp(secs)

    def longOp(self, int secs):
        interruptableCall( self._longOp, [ secs ] )
        checkException()
#        with nogil:
#            self._longOp2(secs)
            #self.thisptr.longOp(secs)

def some_func(float[:] floats):
    print(floats.shape)
    for f in floats:
        print( f )

cdef int cycallback( int num, void *pyCallback ):
    return (<object>pyCallback)(num)

def testCallback(somenum, pyCallback):
    cmycpp.testCallback( cycallback, somenum, <void *>pyCallback )


cdef int MyInterface_getNumber( void *pyObject ):
    return (<object>pyObject).getNumber()

cdef void MyInterface_getFloats( float *floats, void *pyObject ):
#    cdef float[:]floatsMv = floats
    pyFloats = (<object>pyObject).getFloats()
    for i in range(len(pyFloats)):
        floats[i] = pyFloats[i]

cdef class MyInterface:
    cdef cmycpp.CyMyInterface *thisptr
    def __cinit__(self):
        self.thisptr = new cmycpp.CyMyInterface(<void *>self )
        self.thisptr.setGetNumberCallback( MyInterface_getNumber )
        self.thisptr.setGetFloatsCallback( MyInterface_getFloats )
    def __dealloc__(self):
        del self.thisptr
    def getNumber(self):
        return 0 # placeholder
    def getFloats(self):
        return [] # placeholder

def callMyInterface( MyInterface instance ):
    cmycpp.callMyInterface( instance.thisptr )


