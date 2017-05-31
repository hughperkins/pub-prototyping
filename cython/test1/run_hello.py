#!/usr/bin/python

from __future__ import print_function

#import hello
import array

#hello.say_hello('foo')

#print( hello.integrate_f(0.5,1,10000000) ) 

#print( hello.pyDoSomething( 6 ) )

#import pymyc
#print( pymyc.doSomething(5) )

import pymycpp

#print( type( 'foo') )
#print( isinstance( 'foo', str ) )
#print( isinstance( u'foo', str ) )
#print( isinstance( 'foo', unicode ) )
#print( isinstance( u'foo', unicode ) )

pymycpp.sayName('foo')

a = pymycpp.PyMyClass() 
print(a.warpName('blah'))

b = pymycpp.PyMyClass(12) 

print( pymycpp.addFloats(3.5,2.1) )
print( pymycpp.addFloats(1.0,2.0) )

print( pymycpp.some_func( array.array('f', [1,2,3] ) )  )

myarray = array.array('f', [4,2,7] )
print(myarray)
pymycpp.addToArray( myarray, 2.5 )
print(myarray)


myucarray = array.array('B', [4,2,7] )
print(myucarray)
pymycpp.addToUCArray( myucarray, 11 )
print(myucarray)

myintarray = array.array('i', [4,2,7] )
print(myintarray)
pymycpp.addToIntArray( myintarray, 4 )
print(myintarray)



