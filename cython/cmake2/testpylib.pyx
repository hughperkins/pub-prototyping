cimport testpylib2

def saySomething(name):
    print('You said ' + name)

def sayAnythingFromCpp():
    testpylib2.sayAnythingFromCpp()

include 'testpylibinc.pyx'

