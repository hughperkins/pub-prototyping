# from __future__ import print_function

#import libc.stdlib
    
cdef double f(double x):
    return x**2 - x

def integrate_f(double a, double b, int N):
    cdef int i
    cdef double s, dx
    s = 0
    dx = (b-a)/N
    for i in range(N):
        s += f(a+i*dx)
    return s * dx

def say_hello(name):
    print('Hello %s!' % name)

cdef extern from "myc.h":
    double doSomething( double inValue )

def pyDoSomething( inValue ):
    return doSomething( inValue )

